//! Categorical (discrete-symbol) HMM — port of `hmmlearn.hmm.CategoricalHMM`.
//!
//! Observations are non-negative integer symbols stored as a single `f64`
//! column; `n_features` is the number of distinct symbols.

use crate::core::emission::EmissionModel;
use crate::core::hmm::{cumsum, first_gt, Hmm};
use crate::core::inference::Em;
use crate::core::params::{DecoderAlgorithm, Implementation, Param, ParamSet};
use crate::core::{ConvergenceMonitor, Fitted};
use crate::error::{HmmError, Result};
use crate::rng::NumpyRandomState;
use crate::util::normalize_axis;
use ndarray::{Array1, Array2, ArrayView2};

/// Emission sufficient statistics: per-state, per-symbol posterior mass.
pub struct CategoricalStats {
    /// Accumulated posterior mass per state and symbol, `(n_components, n_features)`.
    obs: Array2<f64>,
}

/// Categorical emission model.
#[derive(Clone)]
pub struct CategoricalEm {
    /// Number of hidden states.
    n_components: usize,
    /// Number of distinct symbols; `None` until inferred from data or set.
    n_features: Option<usize>,
    /// Emission matrix `P(symbol | state)`, `(n_components, n_features)`; `None` until initialized.
    emissionprob: Option<Array2<f64>>,
    /// Whether `emissionprob` was caller-supplied (preset) rather than randomly initialized.
    emissionprob_preset: bool,
    /// Dirichlet concentration prior on each state's symbol distribution.
    emissionprob_prior: f64,
}

impl CategoricalEm {
    /// Builds a categorical emission model.
    ///
    /// # Arguments
    /// * `n_components` — number of hidden states.
    /// * `n_features` — number of distinct symbols; if `None`, taken from
    ///   `emissionprob`'s column count when a preset is given, else left unset
    ///   until data is seen.
    /// * `emissionprob` — optional preset emission matrix `(n_components, n_features)`;
    ///   when present it is treated as caller-supplied (preset).
    /// * `emissionprob_prior` — Dirichlet concentration prior on each state's
    ///   symbol distribution.
    ///
    /// # Returns
    /// A `CategoricalEm` ready for initialization or fitting.
    fn new(
        n_components: usize,
        n_features: Option<usize>,
        emissionprob: Option<Array2<f64>>,
        emissionprob_prior: f64,
    ) -> Self {
        let preset = emissionprob.is_some();
        let n_features = n_features.or_else(|| emissionprob.as_ref().map(|e| e.ncols()));
        CategoricalEm {
            n_components,
            n_features,
            emissionprob,
            emissionprob_preset: preset,
            emissionprob_prior,
        }
    }

    /// The fitted emission probabilities, shape `(n_components, n_features)`.
    ///
    /// # Panics
    /// If the emission matrix has not been initialized.
    pub fn emissionprob(&self) -> &Array2<f64> {
        self.emissionprob
            .as_ref()
            .expect("emissionprob not initialized")
    }

    /// The number of symbols.
    ///
    /// # Panics
    /// If `n_features` has not been inferred or set.
    pub fn n_symbols(&self) -> usize {
        self.n_features.expect("n_features not set")
    }

    /// Per-frame emission probabilities `P(X[t] | state)`, shape `(ns, nc)`.
    ///
    /// Reads each row's integer symbol from column 0 and looks it up in every
    /// state's row of the emission matrix.
    ///
    /// # Arguments
    /// * `x` — observations, shape `(ns, 1)`; column 0 holds integer symbols.
    ///
    /// # Returns
    /// A `(ns, nc)` matrix whose `[t, c]` entry is `emissionprob[c, x[t, 0]]`,
    /// where `ns` = number of samples and `nc` = `n_components`.
    ///
    /// # Panics
    /// If the emission matrix or `n_features` is unset, or a symbol lies outside
    /// `0..n_features`.
    fn frame_likelihood(&self, x: ArrayView2<f64>) -> Array2<f64> {
        let ep = self.emissionprob();
        let ns = x.nrows();
        let nc = self.n_components;
        let mut out = Array2::zeros((ns, nc));
        for t in 0..ns {
            let sym = x[[t, 0]] as usize;
            for c in 0..nc {
                out[[t, c]] = ep[[c, sym]];
            }
        }
        out
    }
}

impl EmissionModel for CategoricalEm {
    type Inference = Em;
    type Stats = CategoricalStats;

    fn emission_params() -> &'static [Param] {
        &[Param::Emit]
    }

    fn n_features(&self) -> usize {
        self.n_symbols()
    }

    fn check_and_set_n_features(&mut self, x: ArrayView2<f64>) -> Result<()> {
        for &v in x.iter() {
            if v.fract() != 0.0 {
                return Err(HmmError::InvalidParameter(
                    "Symbols should be integers".into(),
                ));
            }
            if v < 0.0 {
                return Err(HmmError::InvalidParameter(
                    "Symbols should be nonnegative".into(),
                ));
            }
        }
        let max_symbol = x.iter().cloned().fold(0.0_f64, f64::max) as usize;
        match self.n_features {
            Some(nf) => {
                if nf <= max_symbol {
                    return Err(HmmError::InvalidParameter(format!(
                        "Largest symbol is {max_symbol} but the model only emits symbols up to {}",
                        nf - 1
                    )));
                }
            }
            None => self.n_features = Some(max_symbol + 1),
        }
        Ok(())
    }

    fn init(&mut self, _x: ArrayView2<f64>, init: ParamSet, seed: Option<u32>) -> Result<()> {
        if init.contains(Param::Emit) || !self.emissionprob_preset {
            let nc = self.n_components;
            let nf = self.n_symbols();
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
        let nf = self.n_symbols();
        if ep.dim() != (n_components, nf) {
            return Err(HmmError::DimensionMismatch(format!(
                "emissionprob_ must have shape ({n_components}, {nf})"
            )));
        }
        for (i, row) in ep.rows().into_iter().enumerate() {
            if (row.sum() - 1.0).abs() > 1e-5 {
                return Err(HmmError::InvalidParameter(format!(
                    "emissionprob_ row {i} must sum to 1 (got {:.4})",
                    row.sum()
                )));
            }
        }
        Ok(())
    }

    fn n_fit_scalars(&self, n_components: usize, params: ParamSet) -> usize {
        if params.contains(Param::Emit) {
            n_components * (self.n_symbols() - 1)
        } else {
            0
        }
    }

    fn log_likelihood(&self, x: ArrayView2<f64>) -> Array2<f64> {
        self.frame_likelihood(x).mapv(f64::ln)
    }

    fn likelihood(&self, x: ArrayView2<f64>) -> Array2<f64> {
        self.frame_likelihood(x)
    }

    fn init_stats(&self) -> CategoricalStats {
        CategoricalStats {
            obs: Array2::zeros((self.n_components, self.n_symbols())),
        }
    }

    fn accumulate(
        &self,
        stats: &mut CategoricalStats,
        x: ArrayView2<f64>,
        posteriors: ArrayView2<f64>,
        params: ParamSet,
    ) {
        if !params.contains(Param::Emit) {
            return;
        }
        for t in 0..x.nrows() {
            let sym = x[[t, 0]] as usize;
            for c in 0..self.n_components {
                stats.obs[[c, sym]] += posteriors[[t, c]];
            }
        }
    }

    fn mstep(&mut self, stats: &CategoricalStats, params: ParamSet) -> Result<()> {
        if params.contains(Param::Emit) {
            let prior = self.emissionprob_prior;
            let mut ep = stats.obs.mapv(|o| (prior - 1.0 + o).max(0.0));
            normalize_axis(&mut ep, 1);
            self.emissionprob = Some(ep);
        }
        Ok(())
    }

    fn sample_state(&self, state: usize, rng: &mut NumpyRandomState) -> Array1<f64> {
        let cdf = cumsum(self.emissionprob().row(state));
        let symbol = first_gt(cdf.view(), rng.random_sample());
        Array1::from(vec![symbol as f64])
    }
}

/// Builder for [`CategoricalEm`] HMMs (`hmmlearn.hmm.CategoricalHMM`).
#[derive(Clone)]
pub struct CategoricalHmm {
    /// Number of hidden states.
    n_components: usize,
    /// Number of distinct symbols; `None` lets fitting infer it from the data.
    n_features: Option<usize>,
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
    /// Dirichlet concentration prior on each state's symbol distribution.
    emissionprob_prior: f64,
    /// Optional preset initial-state distribution.
    start_prob: Option<Array1<f64>>,
    /// Optional preset transition matrix.
    trans_mat: Option<Array2<f64>>,
    /// Optional preset emission matrix.
    emissionprob: Option<Array2<f64>>,
}

impl CategoricalHmm {
    /// A model with `n_components` states and hmmlearn's defaults.
    ///
    /// # Arguments
    /// * `n_components` — number of hidden states.
    ///
    /// # Returns
    /// A builder carrying hmmlearn's defaults: Viterbi decoding, log-space
    /// implementation, `params`/`init_params` = `"ste"`, `n_iter` = 10,
    /// `tol` = 1e-2, all Dirichlet priors = 1.0, `verbose` = false, and no
    /// preset parameters.
    pub fn new(n_components: usize) -> Self {
        CategoricalHmm {
            n_components,
            n_features: None,
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
            emissionprob_prior: 1.0,
            start_prob: None,
            trans_mat: None,
            emissionprob: None,
        }
    }

    /// Sets the number of distinct symbols the model emits.
    pub fn n_features(mut self, nf: usize) -> Self {
        self.n_features = Some(nf);
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
    /// Sets the Dirichlet concentration prior on the initial-state distribution.
    pub fn startprob_prior(mut self, p: f64) -> Self {
        self.startprob_prior = p;
        self
    }
    /// Sets the Dirichlet concentration prior on each transition-matrix row.
    pub fn transmat_prior(mut self, p: f64) -> Self {
        self.transmat_prior = p;
        self
    }
    /// Sets the Dirichlet concentration prior on each state's symbol distribution.
    pub fn emissionprob_prior(mut self, p: f64) -> Self {
        self.emissionprob_prior = p;
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
    /// An [`Hmm`] wrapping a [`CategoricalEm`] and its EM inference core, carrying
    /// the configured presets, priors, and convergence monitor.
    fn build(self) -> Hmm<CategoricalEm> {
        let inference = Em::new(
            self.n_components,
            self.startprob_prior,
            self.transmat_prior,
            self.start_prob,
            self.trans_mat,
        );
        let emission = CategoricalEm::new(
            self.n_components,
            self.n_features,
            self.emissionprob,
            self.emissionprob_prior,
        );
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
    /// * `x` — observations, shape `(n_samples, 1)`; column 0 holds integer symbols.
    /// * `lengths` — lengths of the individual sequences concatenated in `x`;
    ///   `None` treats `x` as a single sequence.
    ///
    /// # Returns
    /// The [`Fitted`] model.
    ///
    /// # Errors
    /// [`HmmError`] variants from input validation (non-integer or out-of-range
    /// symbols, shape mismatches) or from a numerical failure during EM.
    pub fn fit(
        self,
        x: ArrayView2<f64>,
        lengths: Option<&[usize]>,
    ) -> Result<Fitted<CategoricalEm>> {
        self.build().fit(x, lengths)
    }

    /// Treat the configured (preset) parameters as fitted, after validation.
    ///
    /// # Returns
    /// The [`Fitted`] model built from the preset parameters.
    ///
    /// # Errors
    /// [`HmmError::NotFitted`] if a required parameter is missing, or another
    /// [`HmmError`] variant if a preset fails validation (wrong shape, or a row
    /// that does not sum to 1).
    pub fn into_fitted(self) -> Result<Fitted<CategoricalEm>> {
        self.build().into_fitted()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{assert_close, assert_ll_increasing};
    use ndarray::array;
    use std::collections::HashSet;

    const IMPLS: [Implementation; 2] = [Implementation::Log, Implementation::Scaling];

    /// The Wikipedia categorical model, fitted from preset parameters.
    fn wiki(imp: Implementation) -> Fitted<CategoricalEm> {
        CategoricalHmm::new(2)
            .n_features(3)
            .implementation(imp)
            .start_prob(array![0.6, 0.4])
            .trans_mat(array![[0.7, 0.3], [0.4, 0.6]])
            .emissionprob(array![[0.1, 0.4, 0.5], [0.6, 0.3, 0.1]])
            .into_fitted()
            .unwrap()
    }

    #[test]
    fn wikipedia_decode_viterbi() {
        let x = array![[0.0], [1.0], [2.0]];
        for imp in IMPLS {
            let h = wiki(imp);
            let (log_prob, seq) = h
                .decode(x.view(), None, Some(DecoderAlgorithm::Viterbi))
                .unwrap();
            assert!((log_prob.exp() - 0.01344).abs() < 5e-6);
            assert_eq!(seq.to_vec(), vec![1, 0, 0]);
        }
    }

    #[test]
    fn wikipedia_decode_map() {
        let x = array![[0.0], [1.0], [2.0]];
        for imp in IMPLS {
            let (_lp, seq) = wiki(imp)
                .decode(x.view(), None, Some(DecoderAlgorithm::Map))
                .unwrap();
            assert_eq!(seq.to_vec(), vec![1, 0, 0]);
        }
    }

    #[test]
    fn wikipedia_predict_and_proba() {
        let x = array![[0.0], [1.0], [2.0]];
        let expected = array![
            [0.23170303, 0.76829697],
            [0.62406281, 0.37593719],
            [0.86397706, 0.13602294]
        ];
        for imp in IMPLS {
            let h = wiki(imp);
            assert_eq!(h.predict(x.view(), None).unwrap().to_vec(), vec![1, 0, 0]);
            let proba = h.predict_proba(x.view(), None).unwrap();
            assert_close(&proba, &expected, 1e-6);
        }
    }

    #[test]
    fn n_features_inferred_and_respected() {
        for imp in IMPLS {
            let (seqs, _) = wiki(imp).sample(500, Some(42), None);
            let fitted = CategoricalHmm::new(2)
                .implementation(imp)
                .random_state(1)
                .n_iter(5)
                .fit(seqs.view(), Some(&[500]))
                .unwrap();
            assert_eq!(fitted.emission().n_symbols(), 3);

            let fitted5 = CategoricalHmm::new(2)
                .implementation(imp)
                .n_features(5)
                .random_state(1)
                .n_iter(5)
                .fit(seqs.view(), Some(&[500]))
                .unwrap();
            assert_eq!(fitted5.emission().n_symbols(), 5);

            let model = CategoricalHmm::new(2)
                .implementation(imp)
                .random_state(1)
                .build();
            assert_ll_increasing(model, seqs.view(), Some(&[500]), 10);
        }
    }

    #[test]
    fn bad_emissionprob_rejected() {
        let built = CategoricalHmm::new(2)
            .n_features(3)
            .start_prob(array![0.6, 0.4])
            .trans_mat(array![[0.7, 0.3], [0.4, 0.6]])
            .emissionprob(Array2::zeros((0, 3)))
            .into_fitted();
        assert!(built.is_err());
    }

    #[test]
    fn score_samples_shape_and_normalization() {
        let mut rng = NumpyRandomState::new(7);
        let x: Array2<f64> =
            Array2::from_shape_vec((20, 1), rng.randint(0, 3, 20).mapv(|v| v as f64).to_vec())
                .unwrap();
        for imp in IMPLS {
            let (_ll, post) = wiki(imp).score_samples(x.view(), None).unwrap();
            assert_eq!(post.dim(), (20, 2));
            for row in post.rows() {
                assert!((row.sum() - 1.0).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn sample_shape_and_symbol_coverage() {
        for imp in IMPLS {
            let (x, states) = wiki(imp).sample(1000, Some(42), None);
            assert_eq!(x.ndim(), 2);
            assert_eq!(x.nrows(), 1000);
            assert_eq!(states.len(), 1000);
            let unique: HashSet<i64> = x.iter().map(|&v| v as i64).collect();
            assert_eq!(unique.len(), 3);
        }
    }

    #[test]
    fn fit_increases_log_likelihood() {
        for imp in IMPLS {
            let (x, _) = wiki(imp).sample(100, Some(13), None);
            let lengths = vec![10usize; 10];
            // params = "ste" (all)
            let model = CategoricalHmm::new(2)
                .implementation(imp)
                .random_state(2)
                .build();
            assert_ll_increasing(model, x.view(), Some(&lengths), 5);
            // params = "e" (only emission)
            let model_e = CategoricalHmm::new(2)
                .implementation(imp)
                .params("e")
                .random_state(2)
                .build();
            assert_ll_increasing(model_e, x.view(), Some(&lengths), 5);
        }
    }

    #[test]
    fn base_attributes_validation() {
        // startprob_ that does not sum to 1 is rejected.
        assert!(CategoricalHmm::new(2)
            .n_features(2)
            .start_prob(array![1.2, 0.8])
            .trans_mat(array![[0.5, 0.5], [0.5, 0.5]])
            .emissionprob(array![[0.5, 0.5], [0.5, 0.5]])
            .into_fitted()
            .is_err());
        // transmat_ with the wrong shape is rejected.
        assert!(CategoricalHmm::new(2)
            .n_features(2)
            .start_prob(array![0.5, 0.5])
            .trans_mat(Array2::zeros((0, 2)))
            .emissionprob(array![[0.5, 0.5], [0.5, 0.5]])
            .into_fitted()
            .is_err());
    }

    #[test]
    fn sample_continuation_matches_currstate() {
        // test_base.py::test_generate_samples
        let h = wiki(Implementation::Log);
        let (_x0, z0) = h.sample(10, Some(5), None);
        let last = z0[z0.len() - 1];
        let (_x, z) = h.sample(10, Some(5), Some(last));
        assert_eq!(z0.len(), 10);
        assert_eq!(z.len(), 10);
        assert_eq!(z[0], last);
    }

    #[test]
    fn uniform_transmat_map_decode_is_per_row_argmax() {
        // test_base.py::TestBaseConsistentWithGMM::test_decode — with uniform
        // start/trans, MAP decoding reduces to the per-frame argmax.
        let h = CategoricalHmm::new(2)
            .n_features(3)
            .start_prob(Array1::from_elem(2, 0.5))
            .trans_mat(Array2::from_elem((2, 2), 0.5))
            .emissionprob(array![[0.1, 0.4, 0.5], [0.6, 0.3, 0.1]])
            .into_fitted()
            .unwrap();
        let x = array![[0.0], [1.0], [2.0]];
        let (_lp, seq) = h
            .decode(x.view(), None, Some(DecoderAlgorithm::Map))
            .unwrap();
        assert_eq!(seq.to_vec(), vec![1, 0, 0]);
    }

    #[test]
    fn check_and_set_n_features_validation() {
        // valid non-negative integers
        let mut em = CategoricalEm::new(2, None, None, 1.0);
        assert!(em
            .check_and_set_n_features(
                array![[0.0], [0.0], [2.0], [1.0], [3.0], [1.0], [1.0]].view()
            )
            .is_ok());
        assert_eq!(em.n_symbols(), 4);
        // genuinely non-integral value (the f64 API checks values, not dtype)
        let mut em2 = CategoricalEm::new(2, None, None, 1.0);
        assert!(em2
            .check_and_set_n_features(array![[0.0], [2.0], [1.0], [3.5]].view())
            .is_err());
        // negative
        let mut em3 = CategoricalEm::new(2, None, None, 1.0);
        assert!(em3
            .check_and_set_n_features(array![[0.0], [-2.0], [1.0]].view())
            .is_err());
    }
}
