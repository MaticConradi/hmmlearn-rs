//! The generic HMM core: shared parameters and the universal EM fit loop.
//!
//! `Hmm<E>` owns one emission `E` and its paired inference strategy
//! `E::Inference`. Construction and fitting live here; read-only operations on a
//! fitted model live on [`Fitted`].

use crate::core::algorithms::{
    backward_log, backward_scaling, compute_log_xi_sum, compute_posteriors_log,
    compute_posteriors_scaling, compute_scaling_xi_sum, forward_log, forward_scaling,
};
use crate::core::emission::EmissionModel;
use crate::core::fitted::Fitted;
use crate::core::inference::{start_trans_fit_scalars, CoreStats, Inference};
use crate::core::params::{DecoderAlgorithm, Implementation, Param, ParamSet};
use crate::core::ConvergenceMonitor;
use crate::error::Result;
use crate::util::split_lengths;
use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, Axis};

/// Which likelihood a forward-backward pass should use: the sub-normalized one
/// (variational fitting) or the ordinary one (scoring).
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum FbMode {
    /// Fitting pass: use the (possibly sub-normalized) fit likelihoods.
    Fit,
    /// Scoring pass: use the ordinary emission likelihoods.
    Score,
}

/// One sequence's forward-backward results.
pub(crate) struct ForwardBackward {
    /// Frame likelihood (scaling) or log-likelihood (log),
    /// `(n_samples, n_components)`.
    pub frameprob: Array2<f64>,
    /// Sequence log-likelihood.
    pub log_prob: f64,
    /// Per-sample state posteriors, `(n_samples, n_components)`.
    pub posteriors: Array2<f64>,
    /// Forward lattice, `(n_samples, n_components)`.
    pub fwd: Array2<f64>,
    /// Backward lattice, `(n_samples, n_components)`.
    pub bwd: Array2<f64>,
}

/// A Hidden Markov Model: an emission family `E` over a shared start/transition core.
#[derive(Clone)]
pub struct Hmm<E: EmissionModel> {
    /// The emission distribution family.
    pub(crate) emission: E,
    /// The start/transition inference strategy paired with `E`.
    pub(crate) inference: E::Inference,
    /// Default decoding algorithm for `decode`/`predict`.
    pub(crate) algorithm: DecoderAlgorithm,
    /// Numerical strategy (log or scaling) for forward-backward.
    pub(crate) implementation: Implementation,
    /// Parameter groups updated during fitting.
    pub(crate) params: ParamSet,
    /// Parameter groups initialized before fitting.
    pub(crate) init_params: ParamSet,
    /// Maximum number of EM iterations.
    pub(crate) n_iter: usize,
    /// Optional RNG seed for initialization and sampling.
    pub(crate) random_state: Option<u32>,
    /// EM convergence monitor.
    pub(crate) monitor: ConvergenceMonitor,
}

impl<E: EmissionModel> Hmm<E> {
    /// Number of hidden states.
    pub fn n_components(&self) -> usize {
        self.inference.n_components()
    }

    /// Free scalars across the parameter groups in `params`.
    ///
    /// # Arguments
    /// * `params` — which parameter groups to count.
    ///
    /// # Returns
    /// The start/transition free scalars plus the emission's free scalars for
    /// the selected groups.
    fn fit_scalars(&self, params: ParamSet) -> usize {
        let nc = self.n_components();
        start_trans_fit_scalars(nc, params) + self.emission.n_fit_scalars(nc, params)
    }

    /// Free scalars across every parameter group (for AIC/BIC).
    ///
    /// # Returns
    /// The total free-scalar count over start, transition, and all emission
    /// parameter groups, regardless of `params`.
    pub(crate) fn total_fit_scalars(&self) -> usize {
        let mut all = ParamSet::from_codes("st");
        for &p in E::emission_params() {
            all.insert(p);
        }
        self.fit_scalars(all)
    }

    /// The frame log-likelihood for the current pass.
    ///
    /// # Arguments
    /// * `sub` — one sequence's observations, `(n_samples, n_features)`.
    /// * `mode` — `Fit` selects the (sub-normalized) fit likelihood, `Score` the
    ///   ordinary emission likelihood.
    ///
    /// # Returns
    /// The `(n_samples, n_components)` emission log-likelihood.
    fn frame_log_likelihood(&self, sub: ArrayView2<f64>, mode: FbMode) -> Array2<f64> {
        match mode {
            FbMode::Fit => self.emission.fit_log_likelihood(sub),
            FbMode::Score => self.emission.log_likelihood(sub),
        }
    }

    /// The frame likelihood for the current pass (scaling implementation).
    ///
    /// # Arguments
    /// * `sub` — one sequence's observations, `(n_samples, n_features)`.
    /// * `mode` — `Fit` selects the (sub-normalized) fit likelihood, `Score` the
    ///   ordinary emission likelihood.
    ///
    /// # Returns
    /// The `(n_samples, n_components)` emission likelihood.
    fn frame_likelihood(&self, sub: ArrayView2<f64>, mode: FbMode) -> Array2<f64> {
        match mode {
            FbMode::Fit => self.emission.fit_likelihood(sub),
            FbMode::Score => self.emission.likelihood(sub),
        }
    }

    /// Run forward-backward for one sequence using `(start, trans)`.
    ///
    /// Dispatches on the model's [`Implementation`]: the log path uses
    /// log-likelihoods, the scaling path uses likelihoods plus per-step scaling.
    ///
    /// # Arguments
    /// * `start` — initial-state parameters to drive the recursions.
    /// * `trans` — transition parameters to drive the recursions.
    /// * `sub` — one sequence's observations, `(n_samples, n_features)`.
    /// * `mode` — whether to use fit or scoring likelihoods.
    ///
    /// # Returns
    /// A [`ForwardBackward`] with the frame probabilities, log-likelihood,
    /// posteriors, and forward/backward lattices.
    ///
    /// # Errors
    /// [`ScalingUnderflow`](crate::error::HmmError::ScalingUnderflow) (scaling
    /// path only) if a forward lattice row sums below `1e-300`.
    pub(crate) fn forward_backward(
        &self,
        start: &Array1<f64>,
        trans: &Array2<f64>,
        sub: ArrayView2<f64>,
        mode: FbMode,
    ) -> Result<ForwardBackward> {
        match self.implementation {
            Implementation::Log => {
                let frameprob = self.frame_log_likelihood(sub, mode);
                let (log_prob, fwd) = forward_log(start.view(), trans.view(), frameprob.view());
                let bwd = backward_log(start.view(), trans.view(), frameprob.view());
                let posteriors = compute_posteriors_log(fwd.view(), bwd.view());
                Ok(ForwardBackward {
                    frameprob,
                    log_prob,
                    posteriors,
                    fwd,
                    bwd,
                })
            }
            Implementation::Scaling => {
                let frameprob = self.frame_likelihood(sub, mode);
                let (log_prob, fwd, scaling) =
                    forward_scaling(start.view(), trans.view(), frameprob.view())?;
                let bwd =
                    backward_scaling(start.view(), trans.view(), frameprob.view(), scaling.view());
                let posteriors = compute_posteriors_scaling(fwd.view(), bwd.view());
                Ok(ForwardBackward {
                    frameprob,
                    log_prob,
                    posteriors,
                    fwd,
                    bwd,
                })
            }
        }
    }

    /// Accumulate the shared start/transition statistics from one sequence.
    ///
    /// Adds the first-frame posterior into `start` and, for sequences longer than
    /// one sample, the xi-sum transition counts into `trans`.
    ///
    /// # Arguments
    /// * `stats` — running [`CoreStats`], updated in place.
    /// * `fb` — this sequence's [`ForwardBackward`] results.
    /// * `trans` — the transition matrix used in the E-step (sub-normalized for
    ///   variational inference), matching hmmlearn's xi-sum.
    fn accumulate_core(&self, stats: &mut CoreStats, fb: &ForwardBackward, trans: &Array2<f64>) {
        stats.nobs += 1;
        if self.params.contains(Param::Start) {
            stats.start += &fb.posteriors.row(0);
        }
        // A length-1 sequence has no transitions to accumulate.
        if self.params.contains(Param::Trans) && fb.posteriors.nrows() > 1 {
            let xi = match self.implementation {
                Implementation::Log => compute_log_xi_sum(
                    fb.fwd.view(),
                    trans.view(),
                    fb.bwd.view(),
                    fb.frameprob.view(),
                )
                .mapv(f64::exp),
                Implementation::Scaling => compute_scaling_xi_sum(
                    fb.fwd.view(),
                    trans.view(),
                    fb.bwd.view(),
                    fb.frameprob.view(),
                ),
            };
            stats.trans += &xi;
        }
    }

    /// Estimate model parameters by (variational) Expectation–Maximization.
    ///
    /// Initializes the selected parameters, then iterates the E-step (accumulate
    /// statistics over all sequences) and M-step (update start, transition, and
    /// emission parameters) until the monitor reports convergence or `n_iter` is
    /// reached. Emits a stderr note if the data has fewer points than free
    /// scalar parameters.
    ///
    /// # Arguments
    /// * `x` — concatenated observations, `(n_samples, n_features)`.
    /// * `lengths` — per-sequence lengths summing to `n_samples`; `None` treats
    ///   all rows as one sequence.
    ///
    /// # Returns
    /// A [`Fitted`] model wrapping the trained parameters.
    ///
    /// # Errors
    /// [`LengthsMismatch`](crate::error::HmmError::LengthsMismatch) if `lengths`
    /// does not sum to `n_samples`;
    /// [`ScalingUnderflow`](crate::error::HmmError::ScalingUnderflow) on the
    /// scaling path; plus any error from feature validation, parameter
    /// validation, or an emission M-step.
    pub fn fit(mut self, x: ArrayView2<f64>, lengths: Option<&[usize]>) -> Result<Fitted<E>> {
        let sequences = split_lengths(x.nrows(), lengths)?;

        self.emission.check_and_set_n_features(x)?;
        self.inference.init(
            self.init_params,
            self.random_state,
            sequences.len(),
            x.nrows(),
        );
        self.emission.init(x, self.init_params, self.random_state)?;

        let n_scalars = self.fit_scalars(self.params);
        if x.len() < n_scalars {
            eprintln!(
                "Fitting a model with {n_scalars} free scalar parameters with only {} data \
                 points will result in a degenerate solution.",
                x.len()
            );
        }

        self.inference.check()?;
        self.emission.check(self.n_components())?;
        self.monitor.reset();

        for _ in 0..self.n_iter {
            // E-step: accumulate statistics over all sequences.
            self.emission.estep_begin();
            self.inference.estep_begin();
            let (start, trans) = self.inference.estep_start_trans();

            let mut core = CoreStats::zeros(self.n_components());
            let mut emission_stats = self.emission.init_stats();
            let mut curr_logprob = 0.0;
            for &(begin, end) in &sequences {
                let sub = x.slice(s![begin..end, ..]);
                let fb = self.forward_backward(&start, &trans, sub, FbMode::Fit)?;
                self.accumulate_core(&mut core, &fb, &trans);
                self.emission.accumulate(
                    &mut emission_stats,
                    sub,
                    fb.posteriors.view(),
                    self.params,
                );
                curr_logprob += fb.log_prob;
            }

            // Lower bound uses the pre-update parameters, then M-step.
            let lower_bound =
                self.emission.lower_bound_contribution() + self.inference.lower_bound(curr_logprob);
            self.inference.mstep(&core, self.params);
            self.emission.mstep(&emission_stats, self.params)?;

            self.monitor.report(lower_bound);
            if self.monitor.converged() {
                break;
            }
        }
        Ok(Fitted::new(self))
    }

    /// Treat the configured parameters as fitted, after validation — the
    /// "set attributes then score" path, without running EM.
    ///
    /// # Returns
    /// A [`Fitted`] model over the current parameters.
    ///
    /// # Errors
    /// [`DimensionMismatch`](crate::error::HmmError::DimensionMismatch) or
    /// [`InvalidParameter`](crate::error::HmmError::InvalidParameter) if the
    /// start/transition or emission parameters fail validation.
    pub fn into_fitted(self) -> Result<Fitted<E>> {
        self.inference.check()?;
        self.emission.check(self.n_components())?;
        Ok(Fitted::new(self))
    }
}

/// Cumulative sum of a 1-D array.
///
/// # Arguments
/// * `a` — input array.
///
/// # Returns
/// A new array whose `i`-th entry is the sum of `a[0..=i]`.
pub(crate) fn cumsum(a: ArrayView1<f64>) -> Array1<f64> {
    let mut out = a.to_owned();
    out.accumulate_axis_inplace(Axis(0), |&prev, cur| *cur += prev);
    out
}

/// Cumulative sum along `axis` of a 2-D array.
///
/// # Arguments
/// * `a` — input matrix.
/// * `axis` — axis along which to accumulate.
///
/// # Returns
/// A new matrix of running sums along `axis`.
pub(crate) fn cumsum_axis(a: ArrayView2<f64>, axis: usize) -> Array2<f64> {
    let mut out = a.to_owned();
    out.accumulate_axis_inplace(Axis(axis), |&prev, cur| *cur += prev);
    out
}

/// First index `i` where `cdf[i] > u` — NumPy's `(cdf > u).argmax()` over a CDF.
///
/// # Arguments
/// * `cdf` — cumulative distribution, non-decreasing.
/// * `u` — threshold, typically a uniform draw in `[0, 1)`.
///
/// # Returns
/// The first index whose CDF value exceeds `u`, or the last index if none does.
pub(crate) fn first_gt(cdf: ArrayView1<f64>, u: f64) -> usize {
    cdf.iter().position(|&c| c > u).unwrap_or(cdf.len() - 1)
}
