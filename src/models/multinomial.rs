//! Multinomial HMM — port of `hmmlearn.hmm.MultinomialHMM`.
//!
//! Each observation is a vector of counts summing to `n_trials`. With
//! `n_trials = 1` (one-hot rows) it reduces to [`CategoricalHmm`](super::CategoricalHmm).

use crate::core::emission::EmissionModel;
use crate::core::hmm::Hmm;
use crate::core::inference::Em;
use crate::core::params::{DecoderAlgorithm, Implementation, Param, ParamSet};
use crate::core::{ConvergenceMonitor, Fitted};
use crate::error::{HmmError, Result};
use crate::rng::NumpyRandomState;
use crate::special::ln_gamma;
use crate::util::normalize_axis;
use ndarray::{Array1, Array2, ArrayView2};

/// Computes scipy's `xlogy`: `x * ln(y)`, defined as `0` when `x == 0`.
///
/// # Arguments
/// * `x` — the multiplier; `x == 0` short-circuits to `0` (even for `y <= 0`).
/// * `y` — the value whose natural log is taken.
///
/// # Returns
/// `x * ln(y)`, or `0.0` when `x == 0`.
fn xlogy(x: f64, y: f64) -> f64 {
    if x == 0.0 {
        0.0
    } else {
        x * y.ln()
    }
}

/// Sufficient statistics for multinomial emissions.
pub struct MultinomialStats {
    /// Accumulated posterior-weighted counts per state and symbol, `(n_components, n_features)`.
    obs: Array2<f64>,
}

/// Multinomial emission model.
#[derive(Clone)]
pub struct MultinomialEm {
    /// Number of hidden states.
    n_components: usize,
    /// Number of count categories (features); `None` until inferred or set.
    n_features: Option<usize>,
    /// Emission matrix `P(symbol | state)`, `(n_components, n_features)`; `None` until initialized.
    emissionprob: Option<Array2<f64>>,
    /// Whether `emissionprob` was caller-supplied (preset) rather than randomly initialized.
    emissionprob_preset: bool,
    /// Fixed number of trials each observation's counts sum to; `None` until inferred or set.
    n_trials: Option<i64>,
    /// Whether `n_trials` was inferred from data rather than supplied.
    n_trials_inferred: bool,
}

impl MultinomialEm {
    /// The fitted emission probabilities, shape `(n_components, n_features)`.
    ///
    /// # Panics
    /// If the emission matrix has not been initialized.
    pub fn emissionprob(&self) -> &Array2<f64> {
        self.emissionprob
            .as_ref()
            .expect("emissionprob not initialized")
    }

    /// The number of count categories (features).
    ///
    /// # Panics
    /// If `n_features` has not been inferred or set.
    fn nf(&self) -> usize {
        self.n_features.expect("n_features not set")
    }

    /// Sums row `t` of `x` across all feature columns (its total count).
    ///
    /// # Arguments
    /// * `x` — observations, shape `(n_samples, n_features)`.
    /// * `t` — row index.
    ///
    /// # Returns
    /// The total count `sum_f x[t, f]` for sample `t`.
    fn row_sum(x: ArrayView2<f64>, t: usize) -> f64 {
        (0..x.ncols()).map(|f| x[[t, f]]).sum()
    }
}

impl EmissionModel for MultinomialEm {
    type Inference = Em;
    type Stats = MultinomialStats;

    fn emission_params() -> &'static [Param] {
        &[Param::Emit]
    }

    fn n_features(&self) -> usize {
        self.nf()
    }

    fn check_and_set_n_features(&mut self, x: ArrayView2<f64>) -> Result<()> {
        let nf = x.ncols();
        match self.n_features {
            Some(existing) if existing != nf => {
                return Err(HmmError::DimensionMismatch(format!(
                    "Unexpected number of dimensions, got {nf} but expected {existing}"
                )));
            }
            _ => self.n_features = Some(nf),
        }
        for &v in x.iter() {
            if v.fract() != 0.0 || v < 0.0 {
                return Err(HmmError::InvalidParameter(
                    "Symbol counts should be nonnegative integers".into(),
                ));
            }
        }
        match self.n_trials {
            Some(nt) => {
                for t in 0..x.nrows() {
                    if Self::row_sum(x, t) as i64 != nt {
                        return Err(HmmError::InvalidParameter(
                            "Total count for each sample should add up to the number of trials"
                                .into(),
                        ));
                    }
                }
            }
            None => self.n_trials_inferred = true,
        }
        Ok(())
    }

    fn init(&mut self, _x: ArrayView2<f64>, init: ParamSet, seed: Option<u32>) -> Result<()> {
        if init.contains(Param::Emit) || !self.emissionprob_preset {
            let nc = self.n_components;
            let nf = self.nf();
            let mut rng = NumpyRandomState::new(seed.unwrap_or(0));
            let mut ep = Array2::from_shape_vec((nc, nf), rng.random_sample_n(nc * nf).to_vec())
                .expect("shape");
            normalize_axis(&mut ep, 1);
            self.emissionprob = Some(ep);
        }
        self.emissionprob_preset = true;
        Ok(())
    }

    fn check(&self, n_components: usize) -> Result<()> {
        let ep = self.emissionprob.as_ref().ok_or(HmmError::NotFitted)?;
        if ep.dim() != (n_components, self.nf()) {
            return Err(HmmError::DimensionMismatch(
                "emissionprob_ must have shape (n_components, n_features)".into(),
            ));
        }
        if self.n_trials.is_none() && !self.n_trials_inferred {
            return Err(HmmError::InvalidParameter("n_trials must be set".into()));
        }
        Ok(())
    }

    fn n_fit_scalars(&self, n_components: usize, params: ParamSet) -> usize {
        if params.contains(Param::Emit) {
            n_components * (self.nf() - 1)
        } else {
            0
        }
    }

    fn log_likelihood(&self, x: ArrayView2<f64>) -> Array2<f64> {
        let ep = self.emissionprob();
        let nc = self.n_components;
        let nf = self.nf();
        let ns = x.nrows();
        let mut out = Array2::zeros((ns, nc));
        for t in 0..ns {
            let n: f64 = Self::row_sum(x, t);
            let ln_n_fac = ln_gamma(n + 1.0);
            let sum_ln_x_fac: f64 = (0..nf).map(|f| ln_gamma(x[[t, f]] + 1.0)).sum();
            for c in 0..nc {
                let dot: f64 = (0..nf).map(|f| xlogy(x[[t, f]], ep[[c, f]])).sum();
                out[[t, c]] = ln_n_fac - sum_ln_x_fac + dot;
            }
        }
        out
    }

    fn init_stats(&self) -> MultinomialStats {
        MultinomialStats {
            obs: Array2::zeros((self.n_components, self.nf())),
        }
    }

    fn accumulate(
        &self,
        stats: &mut MultinomialStats,
        x: ArrayView2<f64>,
        posteriors: ArrayView2<f64>,
        params: ParamSet,
    ) {
        if params.contains(Param::Emit) {
            stats.obs += &posteriors.t().dot(&x);
        }
    }

    fn mstep(&mut self, stats: &MultinomialStats, params: ParamSet) -> Result<()> {
        if params.contains(Param::Emit) {
            let mut ep = stats.obs.clone();
            normalize_axis(&mut ep, 1);
            self.emissionprob = Some(ep);
        }
        Ok(())
    }

    /// Draws one multinomial count vector of `n_trials` draws from `state`'s row.
    ///
    /// # Panics
    /// If `n_trials` is unset — sampling requires a single fixed trial count.
    fn sample_state(&self, state: usize, rng: &mut NumpyRandomState) -> Array1<f64> {
        let n = self
            .n_trials
            .expect("a single n_trials must be set for sampling");
        let ep = self.emissionprob();
        let counts = rng.multinomial(n, ep.row(state));
        counts.mapv(|v| v as f64)
    }
}

/// Builder for [`MultinomialEm`] HMMs (`hmmlearn.hmm.MultinomialHMM`).
#[derive(Clone)]
pub struct MultinomialHmm {
    /// Number of hidden states.
    n_components: usize,
    /// Number of count categories (features); `None` lets fitting infer it.
    n_features: Option<usize>,
    /// Fixed per-sample trial count each observation's counts sum to; `None` infers it.
    n_trials: Option<i64>,
    /// Decoding algorithm (Viterbi or MAP).
    algorithm: DecoderAlgorithm,
    /// Forward-backward implementation (log-space or scaling).
    implementation: Implementation,
    /// Which parameters EM updates each iteration.
    params: ParamSet,
    /// Which parameters are (re)initialized before fitting.
    init_params: ParamSet,
    /// Maximum number of EM iterations.
    n_iter: usize,
    /// Convergence threshold on the per-iteration log-likelihood gain.
    tol: f64,
    /// Whether to print per-iteration convergence diagnostics.
    verbose: bool,
    /// Seed for the RNG used in initialization and sampling; `None` is unseeded.
    random_state: Option<u32>,
    /// Dirichlet concentration prior on the initial-state distribution.
    startprob_prior: f64,
    /// Dirichlet concentration prior on each transition-matrix row.
    transmat_prior: f64,
    /// Optional preset initial-state distribution.
    start_prob: Option<Array1<f64>>,
    /// Optional preset transition matrix.
    trans_mat: Option<Array2<f64>>,
    /// Optional preset emission matrix.
    emissionprob: Option<Array2<f64>>,
}

impl MultinomialHmm {
    /// A model with `n_components` states and hmmlearn's defaults.
    ///
    /// # Arguments
    /// * `n_components` — number of hidden states.
    ///
    /// # Returns
    /// A builder carrying hmmlearn's defaults: Viterbi decoding, log-space
    /// implementation, `params`/`init_params` = `"ste"`, `n_iter` = 10,
    /// `tol` = 1e-2, both Dirichlet priors = 1.0, `verbose` = false, and no
    /// preset parameters or `n_trials`.
    pub fn new(n_components: usize) -> Self {
        MultinomialHmm {
            n_components,
            n_features: None,
            n_trials: None,
            algorithm: DecoderAlgorithm::Viterbi,
            implementation: Implementation::Log,
            params: ParamSet::from_codes("ste"),
            init_params: ParamSet::from_codes("ste"),
            n_iter: 10,
            tol: 1e-2,
            verbose: false,
            random_state: None,
            startprob_prior: 1.0,
            transmat_prior: 1.0,
            start_prob: None,
            trans_mat: None,
            emissionprob: None,
        }
    }

    /// Sets the number of count categories (features) the model emits.
    pub fn n_features(mut self, nf: usize) -> Self {
        self.n_features = Some(nf);
        self
    }
    /// Sets the fixed number of trials each observation's counts must sum to.
    pub fn n_trials(mut self, n: i64) -> Self {
        self.n_trials = Some(n);
        self
    }
    /// Sets the maximum number of EM iterations.
    pub fn n_iter(mut self, n: usize) -> Self {
        self.n_iter = n;
        self
    }
    /// Sets the convergence threshold on the per-iteration log-likelihood gain.
    pub fn tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }
    /// Sets the decoding algorithm (Viterbi or MAP).
    pub fn algorithm(mut self, a: DecoderAlgorithm) -> Self {
        self.algorithm = a;
        self
    }
    /// Sets the forward-backward implementation (log-space or scaling).
    pub fn implementation(mut self, i: Implementation) -> Self {
        self.implementation = i;
        self
    }
    /// Sets which parameters EM updates, from letter codes (`s`/`t`/`e`).
    pub fn params(mut self, codes: &str) -> Self {
        self.params = ParamSet::from_codes(codes);
        self
    }
    /// Sets which parameters are initialized before fitting, from letter codes (`s`/`t`/`e`).
    pub fn init_params(mut self, codes: &str) -> Self {
        self.init_params = ParamSet::from_codes(codes);
        self
    }
    /// Sets the RNG seed for initialization and sampling.
    pub fn random_state(mut self, seed: u32) -> Self {
        self.random_state = Some(seed);
        self
    }
    /// Sets whether per-iteration convergence diagnostics are printed.
    pub fn verbose(mut self, v: bool) -> Self {
        self.verbose = v;
        self
    }
    /// Preset the initial-state distribution (skips its initialization).
    pub fn start_prob(mut self, sp: Array1<f64>) -> Self {
        self.start_prob = Some(sp);
        self
    }
    /// Preset the transition matrix.
    pub fn trans_mat(mut self, tm: Array2<f64>) -> Self {
        self.trans_mat = Some(tm);
        self
    }
    /// Preset the emission probabilities.
    pub fn emissionprob(mut self, ep: Array2<f64>) -> Self {
        self.emissionprob = Some(ep);
        self
    }

    /// Assembles the configured builder into an unfitted [`Hmm`].
    ///
    /// # Returns
    /// An [`Hmm`] wrapping a [`MultinomialEm`] and its EM inference core. If
    /// `n_features` was not set, it is taken from the preset emission matrix's
    /// column count when one is given.
    fn build(self) -> Hmm<MultinomialEm> {
        let n_features = self
            .n_features
            .or_else(|| self.emissionprob.as_ref().map(|e| e.ncols()));
        let inference = Em::new(
            self.n_components,
            self.startprob_prior,
            self.transmat_prior,
            self.start_prob,
            self.trans_mat,
        );
        let emission = MultinomialEm {
            n_components: self.n_components,
            n_features,
            emissionprob_preset: self.emissionprob.is_some(),
            emissionprob: self.emissionprob,
            n_trials: self.n_trials,
            n_trials_inferred: false,
        };
        Hmm {
            emission,
            inference,
            algorithm: self.algorithm,
            implementation: self.implementation,
            params: self.params,
            init_params: self.init_params,
            n_iter: self.n_iter,
            random_state: self.random_state,
            monitor: ConvergenceMonitor::new(self.tol, self.n_iter, self.verbose),
        }
    }

    /// Fit the model by EM.
    ///
    /// # Arguments
    /// * `x` — observations, shape `(n_samples, n_features)`; each row is a
    ///   non-negative integer count vector summing to `n_trials`.
    /// * `lengths` — lengths of the individual sequences concatenated in `x`;
    ///   `None` treats `x` as a single sequence.
    ///
    /// # Returns
    /// The [`Fitted`] model.
    ///
    /// # Errors
    /// [`HmmError`] variants from input validation (non-integer or negative
    /// counts, feature-count mismatch, rows not summing to `n_trials`) or from a
    /// numerical failure during EM.
    pub fn fit(
        self,
        x: ArrayView2<f64>,
        lengths: Option<&[usize]>,
    ) -> Result<Fitted<MultinomialEm>> {
        self.build().fit(x, lengths)
    }

    /// Treat the configured (preset) parameters as fitted, after validation.
    ///
    /// # Returns
    /// The [`Fitted`] model built from the preset parameters.
    ///
    /// # Errors
    /// [`HmmError::NotFitted`] if the emission matrix is missing,
    /// [`HmmError::InvalidParameter`] if `n_trials` was neither set nor inferred,
    /// or another [`HmmError`] variant on a shape mismatch.
    pub fn into_fitted(self) -> Result<Fitted<MultinomialEm>> {
        self.build().into_fitted()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::CategoricalHmm;
    use crate::testutil::{assert_close, assert_ll_increasing};
    use ndarray::array;
    use std::collections::HashSet;

    const IMPLS: [Implementation; 2] = [Implementation::Log, Implementation::Scaling];

    fn new_hmm(imp: Implementation) -> MultinomialHmm {
        MultinomialHmm::new(2)
            .n_trials(5)
            .implementation(imp)
            .start_prob(array![0.6, 0.4])
            .trans_mat(array![[0.8, 0.2], [0.2, 0.8]])
            .emissionprob(array![[0.5, 0.3, 0.1, 0.1], [0.1, 0.1, 0.4, 0.4]])
    }

    #[test]
    fn attributes_validation() {
        assert!(MultinomialHmm::new(2)
            .n_trials(5)
            .n_features(4)
            .start_prob(array![0.6, 0.4])
            .trans_mat(array![[0.8, 0.2], [0.2, 0.8]])
            .emissionprob(Array2::zeros((0, 4)))
            .into_fitted()
            .is_err());
        // n_trials unset (and not inferred) is rejected.
        assert!(MultinomialHmm::new(2)
            .n_features(4)
            .start_prob(array![0.6, 0.4])
            .trans_mat(array![[0.8, 0.2], [0.2, 0.8]])
            .emissionprob(array![[0.5, 0.3, 0.1, 0.1], [0.1, 0.1, 0.4, 0.4]])
            .into_fitted()
            .is_err());
    }

    #[test]
    fn score_samples_shape_and_normalization() {
        let x = array![
            [1.0, 1.0, 3.0, 0.0],
            [3.0, 1.0, 1.0, 0.0],
            [3.0, 0.0, 2.0, 0.0],
            [2.0, 2.0, 0.0, 1.0],
            [2.0, 2.0, 0.0, 1.0],
            [0.0, 1.0, 1.0, 3.0],
            [1.0, 0.0, 3.0, 1.0],
            [2.0, 0.0, 1.0, 2.0],
            [0.0, 2.0, 1.0, 2.0],
            [1.0, 0.0, 1.0, 3.0]
        ];
        for imp in IMPLS {
            let h = new_hmm(imp).into_fitted().unwrap();
            let (_ll, post) = h.score_samples(x.view(), None).unwrap();
            assert_eq!(post.dim(), (10, 2));
            for r in post.rows() {
                assert!((r.sum() - 1.0).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn sample_respects_n_trials() {
        for imp in IMPLS {
            let h = new_hmm(imp).into_fitted().unwrap();
            let (x, states) = h.sample(1000, Some(0), None);
            assert_eq!(x.nrows(), 1000);
            assert_eq!(states.len(), 1000);
            let unique: HashSet<i64> = x.iter().map(|&v| v as i64).collect();
            assert_eq!(unique.len(), 6); // counts 0..=5
            for r in x.rows() {
                assert_eq!(r.iter().sum::<f64>() as i64, 5);
            }
        }
    }

    #[test]
    fn check_and_set_n_trials() {
        let mut em = MultinomialEm {
            n_components: 2,
            n_features: None,
            emissionprob: None,
            emissionprob_preset: false,
            n_trials: None,
            n_trials_inferred: false,
        };
        // infers n_trials = 5
        assert!(em
            .check_and_set_n_features(array![[0.0, 2.0, 3.0, 0.0], [1.0, 0.0, 2.0, 2.0]].view())
            .is_ok());
        // wrong number of features
        assert!(em
            .check_and_set_n_features(array![[0.0, 0.0, 2.0, 1.0, 3.0, 1.0, 1.0]].view())
            .is_err());
        // non-integral
        let mut em2 = MultinomialEm {
            n_components: 2,
            n_features: Some(4),
            emissionprob: None,
            emissionprob_preset: false,
            n_trials: Some(5),
            n_trials_inferred: false,
        };
        assert!(em2
            .check_and_set_n_features(array![[0.0, 2.0, 0.0, 3.0], [0.0, 2.5, 2.5, 0.0]].view())
            .is_err());
        // does not sum to n_trials
        let mut em3 = MultinomialEm {
            n_components: 2,
            n_features: Some(4),
            emissionprob: None,
            emissionprob_preset: false,
            n_trials: Some(5),
            n_trials_inferred: false,
        };
        assert!(em3
            .check_and_set_n_features(array![[0.0, 0.0, 1.0, 1.0], [3.0, 1.0, 1.0, 0.0]].view())
            .is_err());
    }

    #[test]
    fn fit_increases_log_likelihood() {
        for imp in IMPLS {
            let (x, _) = new_hmm(imp)
                .into_fitted()
                .unwrap()
                .sample(100, Some(1), None);
            let lengths = vec![10usize; 10];
            let model = MultinomialHmm::new(2)
                .n_trials(5)
                .implementation(imp)
                .random_state(2)
                .build();
            assert_ll_increasing(model, x.view(), Some(&lengths), 5);
        }
    }

    #[test]
    fn compare_with_categorical_hmm() {
        let startprob = array![0.6, 0.4];
        let transmat = array![[0.7, 0.3], [0.4, 0.6]];
        let emissionprob = array![[0.1, 0.4, 0.5], [0.6, 0.3, 0.1]];
        let x1 = array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let x2 = array![[0.0], [1.0], [2.0]];
        let expected = array![
            [0.23170303, 0.76829697],
            [0.62406281, 0.37593719],
            [0.86397706, 0.13602294]
        ];
        for imp in IMPLS {
            let h1 = MultinomialHmm::new(2)
                .n_trials(1)
                .implementation(imp)
                .start_prob(startprob.clone())
                .trans_mat(transmat.clone())
                .emissionprob(emissionprob.clone())
                .into_fitted()
                .unwrap();
            let h2 = CategoricalHmm::new(2)
                .n_features(3)
                .implementation(imp)
                .start_prob(startprob.clone())
                .trans_mat(transmat.clone())
                .emissionprob(emissionprob.clone())
                .into_fitted()
                .unwrap();
            let (lp1, seq1) = h1
                .decode(x1.view(), None, Some(DecoderAlgorithm::Viterbi))
                .unwrap();
            let (lp2, seq2) = h2
                .decode(x2.view(), None, Some(DecoderAlgorithm::Viterbi))
                .unwrap();
            assert!((lp1.exp() - 0.01344).abs() < 5e-6);
            assert!((lp2.exp() - 0.01344).abs() < 5e-6);
            assert_eq!(seq1.to_vec(), vec![1, 0, 0]);
            assert_eq!(seq2.to_vec(), vec![1, 0, 0]);
            assert_close(&h1.predict_proba(x1.view(), None).unwrap(), &expected, 1e-6);
            assert_close(&h2.predict_proba(x2.view(), None).unwrap(), &expected, 1e-6);
        }
    }
}
