//! A fitted model and its inference operations.
//!
//! `Fitted<E>` is produced by [`Hmm::fit`](super::hmm::Hmm::fit) or
//! [`Hmm::into_fitted`](super::hmm::Hmm::into_fitted). Wrapping the model in a
//! distinct type makes "must be fitted" a compile-time property for `score`,
//! `decode`, `sample`, and friends.

use crate::core::algorithms::viterbi;
use crate::core::emission::EmissionModel;
use crate::core::hmm::{cumsum, cumsum_axis, first_gt, FbMode, Hmm};
use crate::core::inference::Inference;
use crate::core::monitor::ConvergenceMonitor;
use crate::core::params::{DecoderAlgorithm, Implementation};
use crate::error::Result;
use crate::linalg::stationary_distribution;
use crate::rng::NumpyRandomState;
use crate::util::split_lengths;
use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, Axis};

/// A trained Hidden Markov Model.
#[derive(Clone)]
pub struct Fitted<E: EmissionModel>(Hmm<E>);

impl<E: EmissionModel> Fitted<E> {
    /// Wrap a trained model as a [`Fitted`].
    ///
    /// # Arguments
    /// * `hmm` — the model whose parameters are treated as fitted.
    ///
    /// # Returns
    /// The wrapped model.
    pub(crate) fn new(hmm: Hmm<E>) -> Self {
        Fitted(hmm)
    }

    /// Recover the underlying model to continue fitting (one EM iteration at a
    /// time, as `assert_log_likelihood_increasing` does).
    #[cfg(test)]
    pub(crate) fn into_inner(self) -> Hmm<E> {
        self.0
    }

    /// Number of hidden states.
    pub fn n_components(&self) -> usize {
        self.0.n_components()
    }
    /// The fitted initial-state distribution.
    pub fn start_prob(&self) -> &Array1<f64> {
        self.0.inference.start_prob()
    }
    /// The fitted transition matrix.
    pub fn trans_mat(&self) -> &Array2<f64> {
        self.0.inference.trans_mat()
    }
    /// The fitted emission model.
    pub fn emission(&self) -> &E {
        &self.0.emission
    }
    /// The convergence monitor after fitting.
    pub fn monitor(&self) -> &ConvergenceMonitor {
        &self.0.monitor
    }

    /// Log-likelihood of `X` and per-sample state posteriors.
    ///
    /// # Arguments
    /// * `x` — concatenated observations, `(n_samples, n_features)`.
    /// * `lengths` — per-sequence lengths summing to `n_samples`; `None` is one
    ///   sequence.
    ///
    /// # Returns
    /// `(log_prob, posteriors)`: the total log-likelihood and the
    /// `(n_samples, n_components)` state posteriors.
    ///
    /// # Errors
    /// [`LengthsMismatch`](crate::error::HmmError::LengthsMismatch) or
    /// [`ScalingUnderflow`](crate::error::HmmError::ScalingUnderflow) (scaling
    /// path).
    ///
    /// # Panics
    /// If the per-sequence posterior blocks cannot be concatenated (shape
    /// disagreement); does not occur for well-formed inputs.
    pub fn score_samples(
        &self,
        x: ArrayView2<f64>,
        lengths: Option<&[usize]>,
    ) -> Result<(f64, Array2<f64>)> {
        self.score_impl(x, lengths, true)
    }

    /// Log-likelihood of `X` under the model.
    ///
    /// # Arguments
    /// * `x` — concatenated observations, `(n_samples, n_features)`.
    /// * `lengths` — per-sequence lengths summing to `n_samples`; `None` is one
    ///   sequence.
    ///
    /// # Returns
    /// The total log-likelihood of `X`.
    ///
    /// # Errors
    /// [`LengthsMismatch`](crate::error::HmmError::LengthsMismatch) or
    /// [`ScalingUnderflow`](crate::error::HmmError::ScalingUnderflow) (scaling
    /// path).
    pub fn score(&self, x: ArrayView2<f64>, lengths: Option<&[usize]>) -> Result<f64> {
        Ok(self.score_impl(x, lengths, false)?.0)
    }

    /// Shared implementation of [`score`](Self::score) and
    /// [`score_samples`](Self::score_samples).
    ///
    /// # Arguments
    /// * `x` — concatenated observations, `(n_samples, n_features)`.
    /// * `lengths` — per-sequence lengths summing to `n_samples`; `None` is one
    ///   sequence.
    /// * `compute_posteriors` — whether to collect and return posteriors; when
    ///   `false` an empty `(0, n_components)` array is returned.
    ///
    /// # Returns
    /// `(log_prob, posteriors)`: the total log-likelihood and either the stacked
    /// posteriors or an empty array.
    ///
    /// # Errors
    /// [`LengthsMismatch`](crate::error::HmmError::LengthsMismatch) or
    /// [`ScalingUnderflow`](crate::error::HmmError::ScalingUnderflow) (scaling
    /// path).
    ///
    /// # Panics
    /// If the per-sequence posterior blocks cannot be concatenated
    /// (`concatenate(...).unwrap()`); does not occur for well-formed inputs since
    /// every block has `n_components` columns.
    fn score_impl(
        &self,
        x: ArrayView2<f64>,
        lengths: Option<&[usize]>,
        compute_posteriors: bool,
    ) -> Result<(f64, Array2<f64>)> {
        let h = &self.0;
        let start = h.inference.start_prob().clone();
        let trans = h.inference.trans_mat().clone();
        let ranges = split_lengths(x.nrows(), lengths)?;
        let mut log_prob = 0.0;
        let mut posts: Vec<Array2<f64>> = Vec::new();
        for (s_idx, e_idx) in ranges {
            let sub = x.slice(s![s_idx..e_idx, ..]);
            let fb = h.forward_backward(&start, &trans, sub, FbMode::Score)?;
            log_prob += fb.log_prob;
            if compute_posteriors {
                posts.push(fb.posteriors);
            }
        }
        let posteriors = if compute_posteriors {
            let views: Vec<ArrayView2<f64>> = posts.iter().map(|p| p.view()).collect();
            ndarray::concatenate(Axis(0), &views).unwrap()
        } else {
            Array2::zeros((0, self.n_components()))
        };
        Ok((log_prob, posteriors))
    }

    /// Most likely state sequence for `X` (uses the model's default algorithm
    /// unless `algorithm` is given).
    ///
    /// Viterbi decodes each sequence independently; MAP takes the per-sample
    /// argmax of the posteriors.
    ///
    /// # Arguments
    /// * `x` — concatenated observations, `(n_samples, n_features)`.
    /// * `lengths` — per-sequence lengths summing to `n_samples`; `None` is one
    ///   sequence.
    /// * `algorithm` — decoder to use; `None` uses the model's default.
    ///
    /// # Returns
    /// `(log_prob, state_sequence)`: the decoded path log-probability (summed
    /// over sequences) and the state indices, length `n_samples`.
    ///
    /// # Errors
    /// [`LengthsMismatch`](crate::error::HmmError::LengthsMismatch), or
    /// [`ScalingUnderflow`](crate::error::HmmError::ScalingUnderflow) via the MAP
    /// path on the scaling implementation.
    ///
    /// # Panics
    /// If the per-sequence Viterbi state sequences cannot be concatenated
    /// (`concatenate(...).unwrap()`); does not occur for well-formed inputs.
    pub fn decode(
        &self,
        x: ArrayView2<f64>,
        lengths: Option<&[usize]>,
        algorithm: Option<DecoderAlgorithm>,
    ) -> Result<(f64, Array1<usize>)> {
        let h = &self.0;
        let algorithm = algorithm.unwrap_or(h.algorithm);
        match algorithm {
            DecoderAlgorithm::Viterbi => {
                let start = h.inference.start_prob().clone();
                let trans = h.inference.trans_mat().clone();
                let ranges = split_lengths(x.nrows(), lengths)?;
                let mut log_prob = 0.0;
                let mut seqs: Vec<Array1<usize>> = Vec::new();
                for (s_idx, e_idx) in ranges {
                    let sub = x.slice(s![s_idx..e_idx, ..]);
                    let lfp = h.emission.log_likelihood(sub);
                    let (lp, seq) = viterbi(start.view(), trans.view(), lfp.view());
                    log_prob += lp;
                    seqs.push(seq);
                }
                let views: Vec<ArrayView1<usize>> = seqs.iter().map(|s| s.view()).collect();
                Ok((log_prob, ndarray::concatenate(Axis(0), &views).unwrap()))
            }
            DecoderAlgorithm::Map => {
                let (_, post) = self.score_samples(x, lengths)?;
                let mut log_prob = 0.0;
                let mut seq = Array1::<usize>::zeros(post.nrows());
                for (t, row) in post.rows().into_iter().enumerate() {
                    let (idx, max) = argmax_row(row);
                    log_prob += max;
                    seq[t] = idx;
                }
                Ok((log_prob, seq))
            }
        }
    }

    /// State sequence for `X` (Viterbi/MAP per the model default).
    ///
    /// # Arguments
    /// * `x` — concatenated observations, `(n_samples, n_features)`.
    /// * `lengths` — per-sequence lengths summing to `n_samples`; `None` is one
    ///   sequence.
    ///
    /// # Returns
    /// The decoded state indices, length `n_samples`.
    ///
    /// # Errors
    /// [`LengthsMismatch`](crate::error::HmmError::LengthsMismatch) or
    /// [`ScalingUnderflow`](crate::error::HmmError::ScalingUnderflow).
    pub fn predict(&self, x: ArrayView2<f64>, lengths: Option<&[usize]>) -> Result<Array1<usize>> {
        Ok(self.decode(x, lengths, None)?.1)
    }

    /// Per-sample state posteriors for `X`.
    ///
    /// # Arguments
    /// * `x` — concatenated observations, `(n_samples, n_features)`.
    /// * `lengths` — per-sequence lengths summing to `n_samples`; `None` is one
    ///   sequence.
    ///
    /// # Returns
    /// The `(n_samples, n_components)` state posteriors.
    ///
    /// # Errors
    /// [`LengthsMismatch`](crate::error::HmmError::LengthsMismatch) or
    /// [`ScalingUnderflow`](crate::error::HmmError::ScalingUnderflow).
    pub fn predict_proba(
        &self,
        x: ArrayView2<f64>,
        lengths: Option<&[usize]>,
    ) -> Result<Array2<f64>> {
        Ok(self.score_samples(x, lengths)?.1)
    }

    /// Generate `n_samples` observations and their hidden states.
    ///
    /// The first state is drawn from the start distribution (unless `currstate`
    /// pins it); subsequent states follow the transition CDF.
    ///
    /// # Arguments
    /// * `n_samples` — number of observations to generate.
    /// * `random_state` — RNG seed; `None` falls back to the model's seed, then 0.
    /// * `currstate` — optional fixed initial state.
    ///
    /// # Returns
    /// `(x, states)`: the `(n_samples, n_features)` observations and the length-
    /// `n_samples` hidden-state sequence.
    ///
    /// # Panics
    /// If `n_samples` is 0 (the feature count is read from the first drawn row,
    /// `rows[0]`).
    pub fn sample(
        &self,
        n_samples: usize,
        random_state: Option<u32>,
        currstate: Option<usize>,
    ) -> (Array2<f64>, Array1<usize>) {
        let h = &self.0;
        let seed = random_state.or(h.random_state).unwrap_or(0);
        let mut rng = NumpyRandomState::new(seed);
        let transmat_cdf = cumsum_axis(h.inference.trans_mat().view(), 1);

        let mut state = match currstate {
            Some(c) => c,
            None => first_gt(
                cumsum(h.inference.start_prob().view()).view(),
                rng.random_sample(),
            ),
        };

        let mut states = Vec::with_capacity(n_samples);
        let mut rows: Vec<Array1<f64>> = Vec::with_capacity(n_samples);
        for t in 0..n_samples {
            if t > 0 {
                state = first_gt(transmat_cdf.row(state), rng.random_sample());
            }
            states.push(state);
            rows.push(h.emission.sample_state(state, &mut rng));
        }

        let n_features = rows[0].len();
        let mut x = Array2::<f64>::zeros((n_samples, n_features));
        for (t, row) in rows.iter().enumerate() {
            x.row_mut(t).assign(row);
        }
        (x, Array1::from(states))
    }

    /// Akaike information criterion: `-2·logL + 2·k`.
    ///
    /// `k` is the total number of free scalar parameters.
    ///
    /// # Arguments
    /// * `x` — concatenated observations, `(n_samples, n_features)`.
    /// * `lengths` — per-sequence lengths summing to `n_samples`; `None` is one
    ///   sequence.
    ///
    /// # Returns
    /// The AIC (lower is better).
    ///
    /// # Errors
    /// [`LengthsMismatch`](crate::error::HmmError::LengthsMismatch) or
    /// [`ScalingUnderflow`](crate::error::HmmError::ScalingUnderflow) from scoring.
    pub fn aic(&self, x: ArrayView2<f64>, lengths: Option<&[usize]>) -> Result<f64> {
        let k = self.0.total_fit_scalars() as f64;
        Ok(-2.0 * self.score(x, lengths)? + 2.0 * k)
    }

    /// Bayesian information criterion: `-2·logL + k·ln(n_samples)`.
    ///
    /// `k` is the total number of free scalar parameters.
    ///
    /// # Arguments
    /// * `x` — concatenated observations, `(n_samples, n_features)`.
    /// * `lengths` — per-sequence lengths summing to `n_samples`; `None` is one
    ///   sequence.
    ///
    /// # Returns
    /// The BIC (lower is better).
    ///
    /// # Errors
    /// [`LengthsMismatch`](crate::error::HmmError::LengthsMismatch) or
    /// [`ScalingUnderflow`](crate::error::HmmError::ScalingUnderflow) from scoring.
    pub fn bic(&self, x: ArrayView2<f64>, lengths: Option<&[usize]>) -> Result<f64> {
        let k = self.0.total_fit_scalars() as f64;
        Ok(-2.0 * self.score(x, lengths)? + k * (x.nrows() as f64).ln())
    }

    /// Stationary distribution of the transition matrix.
    ///
    /// # Returns
    /// The length-`n_components` left-eigenvector of the transition matrix for
    /// eigenvalue 1, normalized to sum to 1.
    pub fn stationary_distribution(&self) -> Array1<f64> {
        stationary_distribution(self.trans_mat().view())
    }

    /// The numerical strategy used for forward-backward.
    pub fn implementation(&self) -> Implementation {
        self.0.implementation
    }
}

/// `(argmax_index, max_value)` of a row; first maximum wins (like `np.argmax`).
///
/// # Arguments
/// * `row` — values to scan.
///
/// # Returns
/// The index and value of the first maximum entry.
fn argmax_row(row: ArrayView1<f64>) -> (usize, f64) {
    let mut idx = 0;
    let mut max = f64::NEG_INFINITY;
    for (i, &v) in row.iter().enumerate() {
        if v > max {
            max = v;
            idx = i;
        }
    }
    (idx, max)
}
