//! Poisson HMM — port of `hmmlearn.hmm.PoissonHMM`.
//!
//! Emissions are independent Poisson counts per feature. Rates are initialized
//! by the method of moments (a Gamma draw) and updated with a Gamma prior.

use crate::core::emission::EmissionModel;
use crate::core::hmm::Hmm;
use crate::core::inference::Em;
use crate::core::params::{DecoderAlgorithm, Implementation, Param, ParamSet};
use crate::core::{ConvergenceMonitor, Fitted};
use crate::error::{HmmError, Result};
use crate::rng::NumpyRandomState;
use crate::special::ln_gamma;
use ndarray::{Array1, Array2, ArrayView2, Axis};

/// Sufficient statistics for Poisson emissions.
pub struct PoissonStats {
    /// Per-state posterior mass `Σₜ γₜ(c)`, shape `(n_components,)`.
    post: Array1<f64>,
    /// Posterior-weighted sum of counts `Σₜ γₜ(c)·xₜ`, shape `(n_components, n_features)`.
    obs: Array2<f64>,
}

/// Poisson emission model.
#[derive(Clone)]
pub struct PoissonEm {
    /// Number of hidden states.
    n_components: usize,
    /// Observation dimensionality; set on the first fit/init.
    n_features: Option<usize>,
    /// Per-state Poisson rates, shape `(n_components, n_features)`.
    lambdas: Option<Array2<f64>>,
    /// Whether the rates were supplied (skips re-initialization).
    lambdas_preset: bool,
    /// Gamma-prior shape `α` on the rates.
    lambdas_prior: f64,
    /// Gamma-prior rate `β` on the rates.
    lambdas_weight: f64,
}

impl PoissonEm {
    /// The fitted rate parameters, shape `(n_components, n_features)`.
    ///
    /// # Panics
    /// If the rates have not been initialized (model not fitted).
    pub fn lambdas(&self) -> &Array2<f64> {
        self.lambdas.as_ref().expect("lambdas not initialized")
    }

    /// The observation dimensionality.
    ///
    /// # Panics
    /// If `n_features` has not been set (no data seen and no preset params).
    fn nf(&self) -> usize {
        self.n_features.expect("n_features not set")
    }
}

impl EmissionModel for PoissonEm {
    type Inference = Em;
    type Stats = PoissonStats;

    fn emission_params() -> &'static [Param] {
        &[Param::Lambdas]
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
        Ok(())
    }

    fn init(&mut self, x: ArrayView2<f64>, init: ParamSet, seed: Option<u32>) -> Result<()> {
        if init.contains(Param::Lambdas) || !self.lambdas_preset {
            let nc = self.n_components;
            let nf = self.nf();
            let mean_x = x.mean().unwrap();
            let var_x = x.var(0.0); // population variance (ddof=0)
            let shape = mean_x * mean_x / var_x;
            let scale = var_x / mean_x;
            let mut rng = NumpyRandomState::new(seed.unwrap_or(0));
            let lambdas = Array2::from_shape_fn((nc, nf), |_| rng.gamma(shape, scale));
            self.lambdas = Some(lambdas);
        }
        self.lambdas_preset = true;
        Ok(())
    }

    fn check(&self, n_components: usize) -> Result<()> {
        let lambdas = self.lambdas.as_ref().ok_or(HmmError::NotFitted)?;
        if lambdas.dim() != (n_components, self.nf()) {
            return Err(HmmError::DimensionMismatch(
                "lambdas_ must have shape (n_components, n_features)".into(),
            ));
        }
        Ok(())
    }

    fn n_fit_scalars(&self, n_components: usize, params: ParamSet) -> usize {
        if params.contains(Param::Lambdas) {
            n_components * self.nf()
        } else {
            0
        }
    }

    fn log_likelihood(&self, x: ArrayView2<f64>) -> Array2<f64> {
        let lambdas = self.lambdas();
        let nc = self.n_components;
        let nf = self.nf();
        let ns = x.nrows();
        let mut out = Array2::zeros((ns, nc));
        for t in 0..ns {
            for c in 0..nc {
                let mut s = 0.0;
                for f in 0..nf {
                    let k = x[[t, f]];
                    let lam = lambdas[[c, f]];
                    // poisson.logpmf(k, lam) = k*ln(lam) - lam - ln(k!)
                    s += k * lam.ln() - lam - ln_gamma(k + 1.0);
                }
                out[[t, c]] = s;
            }
        }
        out
    }

    fn init_stats(&self) -> PoissonStats {
        PoissonStats {
            post: Array1::zeros(self.n_components),
            obs: Array2::zeros((self.n_components, self.nf())),
        }
    }

    fn accumulate(
        &self,
        stats: &mut PoissonStats,
        x: ArrayView2<f64>,
        posteriors: ArrayView2<f64>,
        params: ParamSet,
    ) {
        if params.contains(Param::Lambdas) {
            stats.post += &posteriors.sum_axis(Axis(0));
            stats.obs += &posteriors.t().dot(&x);
        }
    }

    fn mstep(&mut self, stats: &PoissonStats, params: ParamSet) -> Result<()> {
        if params.contains(Param::Lambdas) {
            let nc = self.n_components;
            let nf = self.nf();
            let alphas = self.lambdas_prior;
            let betas = self.lambdas_weight;
            let n: f64 = stats.post.sum();
            let mut lambdas = Array2::zeros((nc, nf));
            for c in 0..nc {
                for f in 0..nf {
                    let y_bar = stats.obs[[c, f]] / stats.post[c];
                    lambdas[[c, f]] = (alphas + n * y_bar) / (betas + n);
                }
            }
            self.lambdas = Some(lambdas);
        }
        Ok(())
    }

    fn sample_state(&self, state: usize, rng: &mut NumpyRandomState) -> Array1<f64> {
        let lambdas = self.lambdas();
        Array1::from_shape_fn(self.nf(), |f| rng.poisson(lambdas[[state, f]]) as f64)
    }
}

/// Builder for [`PoissonEm`] HMMs (`hmmlearn.hmm.PoissonHMM`).
#[derive(Clone)]
pub struct PoissonHmm {
    /// Number of hidden states.
    n_components: usize,
    /// Decoder algorithm (Viterbi or MAP).
    algorithm: DecoderAlgorithm,
    /// Forward/backward arithmetic (log or scaling).
    implementation: Implementation,
    /// Parameters updated during EM.
    params: ParamSet,
    /// Parameters initialized before EM.
    init_params: ParamSet,
    /// Maximum number of EM iterations.
    n_iter: usize,
    /// Convergence threshold on the per-iteration log-likelihood gain.
    tol: f64,
    /// Whether to print per-iteration convergence output.
    verbose: bool,
    /// Optional RNG seed for reproducibility.
    random_state: Option<u32>,
    /// Dirichlet concentration prior on the initial-state distribution.
    startprob_prior: f64,
    /// Dirichlet concentration prior on each transition-matrix row.
    transmat_prior: f64,
    /// Gamma-prior shape `α` on the rates.
    lambdas_prior: f64,
    /// Gamma-prior rate `β` on the rates.
    lambdas_weight: f64,
    /// Optional preset initial-state probabilities.
    start_prob: Option<Array1<f64>>,
    /// Optional preset transition matrix.
    trans_mat: Option<Array2<f64>>,
    /// Optional preset rates.
    lambdas: Option<Array2<f64>>,
}

impl PoissonHmm {
    /// A model with `n_components` states and hmmlearn's defaults.
    ///
    /// Defaults match `PoissonHMM`: Viterbi decoding, log-space arithmetic,
    /// `params`/`init_params` = `"stl"`, `n_iter = 10`, `tol = 1e-2`, non-verbose,
    /// no seed, `startprob_prior = transmat_prior = 1.0`, and
    /// `lambdas_prior = lambdas_weight = 0.0` (rates initialized by method of moments).
    ///
    /// # Arguments
    /// * `n_components` — the number of hidden states.
    ///
    /// # Returns
    /// An unfitted builder carrying the defaults above.
    pub fn new(n_components: usize) -> Self {
        PoissonHmm {
            n_components,
            algorithm: DecoderAlgorithm::Viterbi,
            implementation: Implementation::Log,
            params: ParamSet::from_codes("stl"),
            init_params: ParamSet::from_codes("stl"),
            n_iter: 10,
            tol: 1e-2,
            verbose: false,
            random_state: None,
            startprob_prior: 1.0,
            transmat_prior: 1.0,
            lambdas_prior: 0.0,
            lambdas_weight: 0.0,
            start_prob: None,
            trans_mat: None,
            lambdas: None,
        }
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
    /// Sets the decoder algorithm (Viterbi or MAP).
    pub fn algorithm(mut self, a: DecoderAlgorithm) -> Self {
        self.algorithm = a;
        self
    }
    /// Sets the forward/backward arithmetic (log or scaling).
    pub fn implementation(mut self, i: Implementation) -> Self {
        self.implementation = i;
        self
    }
    /// Sets which parameters are updated during EM (from letter codes, e.g. `"stl"`).
    pub fn params(mut self, codes: &str) -> Self {
        self.params = ParamSet::from_codes(codes);
        self
    }
    /// Sets which parameters are initialized before EM (from letter codes).
    pub fn init_params(mut self, codes: &str) -> Self {
        self.init_params = ParamSet::from_codes(codes);
        self
    }
    /// Sets the RNG seed for reproducible initialization and sampling.
    pub fn random_state(mut self, seed: u32) -> Self {
        self.random_state = Some(seed);
        self
    }
    /// Sets whether to print per-iteration convergence output.
    pub fn verbose(mut self, v: bool) -> Self {
        self.verbose = v;
        self
    }
    /// Sets the Gamma-prior shape `α` on the emission rates.
    pub fn lambdas_prior(mut self, p: f64) -> Self {
        self.lambdas_prior = p;
        self
    }
    /// Sets the Gamma-prior rate `β` on the emission rates.
    pub fn lambdas_weight(mut self, w: f64) -> Self {
        self.lambdas_weight = w;
        self
    }
    /// Presets the initial-state probabilities.
    pub fn start_prob(mut self, sp: Array1<f64>) -> Self {
        self.start_prob = Some(sp);
        self
    }
    /// Presets the transition matrix.
    pub fn trans_mat(mut self, tm: Array2<f64>) -> Self {
        self.trans_mat = Some(tm);
        self
    }
    /// Presets the Poisson emission rates.
    pub fn lambdas(mut self, l: Array2<f64>) -> Self {
        self.lambdas = Some(l);
        self
    }

    /// Assembles the configured builder into an unfitted [`Hmm`].
    ///
    /// Infers `n_features` from the preset rates when supplied.
    ///
    /// # Returns
    /// An [`Hmm`] carrying the emission model, EM inference, and convergence monitor.
    fn build(self) -> Hmm<PoissonEm> {
        let n_features = self.lambdas.as_ref().map(|l| l.ncols());
        let inference = Em::new(
            self.n_components,
            self.startprob_prior,
            self.transmat_prior,
            self.start_prob,
            self.trans_mat,
        );
        let emission = PoissonEm {
            n_components: self.n_components,
            n_features,
            lambdas_preset: self.lambdas.is_some(),
            lambdas: self.lambdas,
            lambdas_prior: self.lambdas_prior,
            lambdas_weight: self.lambdas_weight,
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

    /// Fit the model to observations by EM.
    ///
    /// # Arguments
    /// * `x` — count observations, shape `(n_samples, n_features)`.
    /// * `lengths` — optional per-sequence lengths partitioning the rows of `x` into
    ///   independent sequences; `None` treats all rows as a single sequence.
    ///
    /// # Returns
    /// The [`Fitted`] model after EM.
    ///
    /// # Errors
    /// Forwards errors from initialization, validation, and the EM loop, e.g.
    /// [`HmmError::DimensionMismatch`] or [`HmmError::ScalingUnderflow`].
    pub fn fit(self, x: ArrayView2<f64>, lengths: Option<&[usize]>) -> Result<Fitted<PoissonEm>> {
        self.build().fit(x, lengths)
    }

    /// Treat the configured (preset) parameters as already fitted, after validation.
    ///
    /// # Returns
    /// The [`Fitted`] model wrapping the preset parameters.
    ///
    /// # Errors
    /// [`HmmError::NotFitted`] if a required parameter was not preset, or
    /// [`HmmError::DimensionMismatch`] if a preset has the wrong shape.
    pub fn into_fitted(self) -> Result<Fitted<PoissonEm>> {
        self.build().into_fitted()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::assert_ll_increasing;
    use ndarray::array;

    const IMPLS: [Implementation; 2] = [Implementation::Log, Implementation::Scaling];

    fn new_hmm(imp: Implementation) -> PoissonHmm {
        PoissonHmm::new(2)
            .implementation(imp)
            .random_state(0)
            .start_prob(array![0.6, 0.4])
            .trans_mat(array![[0.7, 0.3], [0.4, 0.6]])
            .lambdas(array![[3.1, 1.4, 4.5], [1.6, 5.3, 0.1]])
    }

    #[test]
    fn attributes_validation() {
        // lambdas with the wrong number of rows is rejected.
        assert!(PoissonHmm::new(2)
            .start_prob(array![0.6, 0.4])
            .trans_mat(array![[0.7, 0.3], [0.4, 0.6]])
            .lambdas(Array2::zeros((0, 3)))
            .into_fitted()
            .is_err());
    }

    #[test]
    fn score_samples_shape_and_normalization() {
        for imp in IMPLS {
            let h = new_hmm(imp).into_fitted().unwrap();
            let (x, states) = h.sample(1000, Some(0), None);
            assert_eq!(x.ndim(), 2);
            assert_eq!(x.nrows(), 1000);
            assert_eq!(states.len(), 1000);
            let (_ll, post) = h.score_samples(x.view(), None).unwrap();
            assert_eq!(post.dim(), (1000, 2));
            for r in post.rows() {
                assert!((r.sum() - 1.0).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn fit_increases_log_likelihood() {
        for imp in IMPLS {
            let (x, _) = new_hmm(imp)
                .into_fitted()
                .unwrap()
                .sample(100, Some(1), None);
            let lengths = vec![10usize; 10];
            let model = PoissonHmm::new(2)
                .implementation(imp)
                .random_state(2)
                .build();
            assert_ll_increasing(model, x.view(), Some(&lengths), 5);
            let model_l = PoissonHmm::new(2)
                .implementation(imp)
                .params("l")
                .random_state(2)
                .build();
            assert_ll_increasing(model_l, x.view(), Some(&lengths), 5);
        }
    }

    #[test]
    fn criterion_aic_bic_finite() {
        for imp in IMPLS {
            let (x, _) = new_hmm(imp)
                .into_fitted()
                .unwrap()
                .sample(500, Some(412), None);
            for n in [2usize, 3, 4] {
                let h = PoissonHmm::new(n)
                    .implementation(imp)
                    .n_iter(20)
                    .random_state(412)
                    .fit(x.view(), None)
                    .unwrap();
                assert!(h.aic(x.view(), None).unwrap().is_finite());
                assert!(h.bic(x.view(), None).unwrap().is_finite());
            }
        }
    }
}
