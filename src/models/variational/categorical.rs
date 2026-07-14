//! Variational-Bayes categorical HMM — port of
//! `hmmlearn.vhmm.VariationalCategoricalHMM`.
//!
//! Emission probabilities are Dirichlet-distributed (prior + posterior). The
//! E-step uses the sub-normalized log-likelihood (digamma of the posterior);
//! scoring uses the normalized posterior median.

use crate::core::emission::EmissionModel;
use crate::core::hmm::{cumsum, first_gt, Hmm};
use crate::core::inference::Variational;
use crate::core::params::{DecoderAlgorithm, Implementation, Param, ParamSet};
use crate::core::{ConvergenceMonitor, Fitted};
use crate::error::{HmmError, Result};
use crate::kl::kl_dirichlet;
use crate::rng::NumpyRandomState;
use crate::special::digamma;
use crate::util::normalize_axis;
use ndarray::{Array1, Array2, ArrayView2};

/// Emission sufficient statistics: per-state, per-symbol posterior mass.
pub struct VCatStats {
    /// Per-state, per-symbol posterior mass `Σₜ γ(t,c)·[xₜ = symbol]`, shape
    /// `(n_components, n_features)`.
    obs: Array2<f64>,
}

/// Variational categorical emission model.
#[derive(Clone)]
pub struct VariationalCategoricalEm {
    n_components: usize,
    n_features: Option<usize>,
    emissionprob: Option<Array2<f64>>, // normalized posterior median (scoring/sampling)
    prior: Option<Array2<f64>>,
    posterior: Option<Array2<f64>>,
    log_subnorm: Option<Array2<f64>>, // digamma terms (fitting)
    emissionprob_prior_scalar: Option<f64>,
    posterior_preset: bool,
}

impl VariationalCategoricalEm {
    /// The number of categorical symbols.
    ///
    /// # Panics
    /// Panics if `n_features` has not been set (no data seen / not initialized).
    fn nf(&self) -> usize {
        self.n_features.expect("n_features not set")
    }
    /// The Dirichlet posterior over emission probabilities, shape
    /// `(n_components, n_features)`.
    ///
    /// # Panics
    /// Panics if the posterior has not been initialized.
    fn posterior(&self) -> &Array2<f64> {
        self.posterior.as_ref().expect("posterior not initialized")
    }
    /// The normalized emission probabilities (posterior median).
    pub fn emissionprob(&self) -> &Array2<f64> {
        self.emissionprob.as_ref().expect("emissionprob not set")
    }
    /// The Dirichlet posterior over emission probabilities.
    pub fn emissionprob_posterior(&self) -> &Array2<f64> {
        self.posterior()
    }

    /// Recompute the normalized emission probabilities (the posterior median):
    /// each state's Dirichlet posterior row rescaled to sum to one.
    ///
    /// # Panics
    /// Panics if the posterior has not been initialized.
    fn update_normalized(&mut self) {
        let mut ep = self.posterior().clone();
        normalize_axis(&mut ep, 1);
        self.emissionprob = Some(ep);
    }
}

impl EmissionModel for VariationalCategoricalEm {
    type Inference = Variational;
    type Stats = VCatStats;

    fn emission_params() -> &'static [Param] {
        &[Param::Emit]
    }

    fn n_features(&self) -> usize {
        self.nf()
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
                        "Largest symbol is {max_symbol} but the model only emits up to {}",
                        nf - 1
                    )));
                }
            }
            None => self.n_features = Some(max_symbol + 1),
        }
        Ok(())
    }

    fn init(&mut self, x: ArrayView2<f64>, init: ParamSet, seed: Option<u32>) -> Result<()> {
        if init.contains(Param::Emit) || !self.posterior_preset {
            let nc = self.n_components;
            let nf = self.nf();
            let ep_init = self.emissionprob_prior_scalar.unwrap_or(1.0 / nf as f64);
            self.prior = Some(Array2::from_elem((nc, nf), ep_init));
            let alpha = Array1::from_elem(nf, ep_init);
            let mut rng = NumpyRandomState::new(seed.unwrap_or(0));
            let scale = x.nrows() as f64 / nc as f64;
            let mut post = Array2::zeros((nc, nf));
            for c in 0..nc {
                post.row_mut(c)
                    .assign(&(rng.dirichlet(alpha.view()) * scale));
            }
            self.posterior = Some(post);
        }
        self.posterior_preset = true;
        self.update_normalized();
        Ok(())
    }

    fn check(&self, n_components: usize) -> Result<()> {
        let post = self.posterior.as_ref().ok_or(HmmError::NotFitted)?;
        let prior = self.prior.as_ref().ok_or(HmmError::NotFitted)?;
        if post.dim() != prior.dim() {
            return Err(HmmError::DimensionMismatch(
                "emissionprob prior/posterior must have the same shape".into(),
            ));
        }
        if post.dim() != (n_components, self.nf()) {
            return Err(HmmError::DimensionMismatch(
                "emissionprob_ must have shape (n_components, n_features)".into(),
            ));
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

    fn estep_begin(&mut self) {
        let post = self.posterior();
        let (nc, nf) = post.dim();
        let mut ls = Array2::zeros((nc, nf));
        for c in 0..nc {
            let dg_sum = digamma(post.row(c).sum());
            for f in 0..nf {
                ls[[c, f]] = digamma(post[[c, f]]) - dg_sum;
            }
        }
        self.log_subnorm = Some(ls);
    }

    /// Scoring: log of the normalized emission probabilities.
    fn log_likelihood(&self, x: ArrayView2<f64>) -> Array2<f64> {
        let ep = self.emissionprob();
        let ns = x.nrows();
        let nc = self.n_components;
        Array2::from_shape_fn((ns, nc), |(t, c)| ep[[c, x[[t, 0]] as usize]].ln())
    }

    /// Fitting: the sub-normalized log-likelihood.
    ///
    /// # Panics
    /// Panics if `estep_begin` has not populated the sub-normalized
    /// log-emission table for the current posteriors.
    fn fit_log_likelihood(&self, x: ArrayView2<f64>) -> Array2<f64> {
        let ls = self.log_subnorm.as_ref().expect("estep_begin not called");
        let ns = x.nrows();
        let nc = self.n_components;
        Array2::from_shape_fn((ns, nc), |(t, c)| ls[[c, x[[t, 0]] as usize]])
    }

    /// Scaling-path fitting likelihood: the sub-normalized likelihood.
    /// (The default would exponentiate the *normalized* scoring likelihood.)
    fn fit_likelihood(&self, x: ArrayView2<f64>) -> Array2<f64> {
        self.fit_log_likelihood(x).mapv(f64::exp)
    }

    fn init_stats(&self) -> VCatStats {
        VCatStats {
            obs: Array2::zeros((self.n_components, self.nf())),
        }
    }

    fn accumulate(
        &self,
        stats: &mut VCatStats,
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

    fn mstep(&mut self, stats: &VCatStats, params: ParamSet) -> Result<()> {
        if params.contains(Param::Emit) {
            let prior = self.prior.as_ref().expect("prior set");
            self.posterior = Some(prior + &stats.obs);
            self.update_normalized();
        }
        Ok(())
    }

    fn lower_bound_contribution(&self) -> f64 {
        let post = self.posterior();
        let prior = self.prior.as_ref().expect("prior set");
        let mut lb = 0.0;
        for i in 0..self.n_components {
            lb -= kl_dirichlet(post.row(i), prior.row(i));
        }
        lb
    }

    fn sample_state(&self, state: usize, rng: &mut NumpyRandomState) -> Array1<f64> {
        let cdf = cumsum(self.emissionprob().row(state));
        Array1::from(vec![first_gt(cdf.view(), rng.random_sample()) as f64])
    }
}

/// Builder for [`VariationalCategoricalEm`] HMMs
/// (`hmmlearn.vhmm.VariationalCategoricalHMM`).
///
/// Unlike the EM builders there is no `into_fitted`: a variational model is
/// always obtained by fitting.
#[derive(Clone)]
pub struct VariationalCategoricalHmm {
    n_components: usize,
    n_features: Option<usize>,
    algorithm: DecoderAlgorithm,
    implementation: Implementation,
    params: ParamSet,
    init_params: ParamSet,
    n_iter: usize,
    tol: f64,
    verbose: bool,
    random_state: Option<u32>,
    startprob_prior: Option<f64>,
    transmat_prior: Option<f64>,
    emissionprob_prior: Option<f64>,
    startprob_prior_arr: Option<Array1<f64>>,
    startprob_posterior: Option<Array1<f64>>,
    transmat_prior_arr: Option<Array2<f64>>,
    transmat_posterior: Option<Array2<f64>>,
}

impl VariationalCategoricalHmm {
    /// A model with `n_components` states and hmmlearn's variational defaults.
    ///
    /// Matches `VariationalCategoricalHMM`'s defaults: `n_iter = 100`,
    /// `tol = 1e-6`, log-space forward-backward, all parameters (`ste`) both
    /// initialized and updated, symbol count inferred from the data, and no
    /// random seed. The Dirichlet priors are left unset and filled at `init`.
    ///
    /// # Arguments
    /// * `n_components` — number of hidden states.
    ///
    /// # Returns
    /// An unfitted builder.
    pub fn new(n_components: usize) -> Self {
        VariationalCategoricalHmm {
            n_components,
            n_features: None,
            algorithm: DecoderAlgorithm::Viterbi,
            implementation: Implementation::Log,
            params: ParamSet::from_codes("ste"),
            init_params: ParamSet::from_codes("ste"),
            n_iter: 100,
            tol: 1e-6,
            verbose: false,
            random_state: None,
            startprob_prior: None,
            transmat_prior: None,
            emissionprob_prior: None,
            startprob_prior_arr: None,
            startprob_posterior: None,
            transmat_prior_arr: None,
            transmat_posterior: None,
        }
    }

    /// Sets the number of categorical symbols (otherwise inferred from data).
    pub fn n_features(mut self, nf: usize) -> Self {
        self.n_features = Some(nf);
        self
    }
    /// Sets the Dirichlet concentration prior on the initial-state distribution.
    pub fn startprob_prior(mut self, p: Array1<f64>) -> Self {
        self.startprob_prior_arr = Some(p);
        self
    }
    /// Presets the Dirichlet posterior over the initial-state distribution.
    pub fn startprob_posterior(mut self, p: Array1<f64>) -> Self {
        self.startprob_posterior = Some(p);
        self
    }
    /// Sets the Dirichlet concentration prior on the transition-matrix rows.
    pub fn transmat_prior(mut self, p: Array2<f64>) -> Self {
        self.transmat_prior_arr = Some(p);
        self
    }
    /// Presets the Dirichlet posterior over the transition-matrix rows.
    pub fn transmat_posterior(mut self, p: Array2<f64>) -> Self {
        self.transmat_posterior = Some(p);
        self
    }
    /// Sets the maximum number of variational EM iterations.
    pub fn n_iter(mut self, n: usize) -> Self {
        self.n_iter = n;
        self
    }
    /// Sets the convergence threshold on the per-iteration lower-bound gain.
    pub fn tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }
    /// Sets the forward-backward implementation (log-space or scaling).
    pub fn implementation(mut self, i: Implementation) -> Self {
        self.implementation = i;
        self
    }
    /// Sets which parameters are updated each iteration (`s`/`t`/`e`).
    pub fn params(mut self, codes: &str) -> Self {
        self.params = ParamSet::from_codes(codes);
        self
    }
    /// Sets which parameters are initialized before fitting (`s`/`t`/`e`).
    pub fn init_params(mut self, codes: &str) -> Self {
        self.init_params = ParamSet::from_codes(codes);
        self
    }
    /// Sets the random seed used for initialization.
    pub fn random_state(mut self, seed: u32) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Assemble the configured builder into an unfitted [`Hmm`].
    ///
    /// Packages the Dirichlet variational inference core (from the scalar or
    /// array start/transition priors and any preset posteriors) with the
    /// categorical emission model.
    ///
    /// # Returns
    /// An unfitted [`Hmm`] wrapping a [`VariationalCategoricalEm`].
    pub(crate) fn build(self) -> Hmm<VariationalCategoricalEm> {
        let inference = Variational::new(
            self.n_components,
            self.startprob_prior,
            self.transmat_prior,
            self.startprob_prior_arr,
            self.startprob_posterior,
            self.transmat_prior_arr,
            self.transmat_posterior,
        );
        let emission = VariationalCategoricalEm {
            n_components: self.n_components,
            n_features: self.n_features,
            emissionprob: None,
            prior: None,
            posterior: None,
            log_subnorm: None,
            emissionprob_prior_scalar: self.emissionprob_prior,
            posterior_preset: false,
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

    /// Fit the model by variational EM.
    ///
    /// There is no `into_fitted` counterpart (unlike the EM builders): a
    /// variational model is always obtained by fitting.
    ///
    /// # Arguments
    /// * `x` — observation matrix of symbol indices, `(n_samples, 1)`.
    /// * `lengths` — optional per-sequence lengths summing to `n_samples`;
    ///   `None` treats `x` as a single sequence.
    ///
    /// # Returns
    /// The [`Fitted`] variational model.
    ///
    /// # Errors
    /// Propagates any error from initialization, parameter validation, or the
    /// variational EM loop.
    pub fn fit(
        self,
        x: ArrayView2<f64>,
        lengths: Option<&[usize]>,
    ) -> Result<Fitted<VariationalCategoricalEm>> {
        self.build().fit(x, lengths)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::CategoricalHmm;
    use crate::testutil::assert_ll_increasing;
    use ndarray::array;

    const IMPLS: [Implementation; 2] = [Implementation::Log, Implementation::Scaling];

    /// Data from a 3-state categorical HMM (a Beal-style cyclic model).
    fn beal_data(seed: u32, n_seq: usize, length: usize) -> (Array2<f64>, Vec<usize>) {
        let gen = CategoricalHmm::new(3)
            .n_features(3)
            .start_prob(array![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
            .trans_mat(array![[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
            .emissionprob(array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
            .into_fitted()
            .unwrap();
        let mut rows: Vec<f64> = Vec::new();
        let mut lengths = Vec::new();
        for i in 0..n_seq {
            let (x, _) = gen.sample(length, Some(seed + i as u32), None);
            rows.extend(x.iter().copied());
            lengths.push(length);
        }
        let total: usize = lengths.iter().sum();
        (Array2::from_shape_vec((total, 1), rows).unwrap(), lengths)
    }

    #[test]
    fn fit_increases_lower_bound() {
        for imp in IMPLS {
            let (x, lengths) = beal_data(1984, 7, 100);
            let model = VariationalCategoricalHmm::new(4)
                .implementation(imp)
                .random_state(1984)
                .build();
            assert_ll_increasing(model, x.view(), Some(&lengths), 10);
        }
    }

    #[test]
    fn n_features_inferred() {
        let (x, lengths) = beal_data(7, 5, 50);
        let fitted = VariationalCategoricalHmm::new(3)
            .random_state(1)
            .n_iter(20)
            .fit(x.view(), Some(&lengths))
            .unwrap();
        assert_eq!(fitted.emission().emissionprob().ncols(), 3);
    }

    #[test]
    fn log_and_scaling_agree() {
        // Both implementations use the sub-normalized likelihood in the E-step,
        // so a fit from identical initial conditions must reach the same score.
        let (x, lengths) = beal_data(2024, 5, 60);
        let score = |imp| {
            VariationalCategoricalHmm::new(3)
                .implementation(imp)
                .random_state(11)
                .n_iter(30)
                .fit(x.view(), Some(&lengths))
                .unwrap()
                .score(x.view(), Some(&lengths))
                .unwrap()
        };
        let log = score(Implementation::Log);
        let scaling = score(Implementation::Scaling);
        assert!(
            (log - scaling).abs() < 1e-6,
            "log {log} vs scaling {scaling}"
        );
    }

    #[test]
    fn fit_length_one_sequences() {
        // Single-sample sequences must not break the fit.
        let (x, _) = beal_data(3, 1, 30);
        let lengths = vec![1usize; 30];
        assert!(VariationalCategoricalHmm::new(2)
            .random_state(1)
            .n_iter(10)
            .fit(x.view(), Some(&lengths))
            .is_ok());
    }

    /// Port of `test_fit_and_compare_with_em`: four components with uniform
    /// variational priors fit to data from a 3-state model. One state is left
    /// "unused" (its posteriors relax to the uniform prior), and the posterior
    /// point estimate scores/decodes/samples identically to the equivalent EM
    /// categorical model.
    #[test]
    fn matches_em_model_at_posterior() {
        use crate::core::params::DecoderAlgorithm::{Map, Viterbi};
        for imp in IMPLS {
            let (x, lengths) = beal_data(1984, 7, 100);
            let total: usize = lengths.iter().sum();
            let nc = 4;
            let uniform = 1.0 / nc as f64;
            let vi = VariationalCategoricalHmm::new(nc)
                .n_features(3)
                .implementation(imp)
                .init_params("e")
                .random_state(1984)
                .n_iter(500)
                .startprob_prior(Array1::from_elem(nc, uniform))
                .startprob_posterior(Array1::from_elem(nc, uniform * lengths.len() as f64))
                .transmat_prior(Array2::from_elem((nc, nc), uniform))
                .transmat_posterior(Array2::from_elem((nc, nc), uniform * total as f64))
                .fit(x.view(), Some(&lengths))
                .unwrap();

            // Automatic relevance determination leaves at least one state unused:
            // its emission row relaxes to the uniform posterior median.
            let unused = (0..nc).any(|i| {
                vi.emission()
                    .emissionprob()
                    .row(i)
                    .iter()
                    .all(|&p| (p - 1.0 / 3.0).abs() < 1e-2)
            });
            assert!(unused, "{imp:?}: expected an unused (uniform) state");

            let em = CategoricalHmm::new(nc)
                .n_features(3)
                .start_prob(vi.start_prob().clone())
                .trans_mat(vi.trans_mat().clone())
                .emissionprob(vi.emission().emissionprob().clone())
                .into_fitted()
                .unwrap();

            let vs = vi.score(x.view(), Some(&lengths)).unwrap();
            let es = em.score(x.view(), Some(&lengths)).unwrap();
            assert!((vs - es).abs() < 1e-9, "{imp:?}: score {vs} vs {es}");
            for algo in [Viterbi, Map] {
                let (vlp, vpath) = vi.decode(x.view(), Some(&lengths), Some(algo)).unwrap();
                let (elp, epath) = em.decode(x.view(), Some(&lengths), Some(algo)).unwrap();
                assert!((vlp - elp).abs() < 1e-9, "{imp:?}/{algo:?}");
                assert_eq!(vpath, epath, "{imp:?}/{algo:?}");
            }
            let vp = vi.predict_proba(x.view(), Some(&lengths)).unwrap();
            let ep = em.predict_proba(x.view(), Some(&lengths)).unwrap();
            crate::testutil::assert_close(&vp, &ep, 1e-9);
            let (vo, vst) = vi.sample(100, Some(42), None);
            let (eo, est) = em.sample(100, Some(42), None);
            assert_eq!(vst, est, "{imp:?} sample states");
            crate::testutil::assert_close(&vo, &eo, 0.0);
        }
    }
}
