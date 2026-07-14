//! Gaussian HMM — port of `hmmlearn.hmm.GaussianHMM`.
//!
//! Supports the four covariance parameterizations. Means are initialized with
//! k-means++ and covariances from the sample covariance (`distribute_covar`).
//! The M-step follows Huang/Acero/Hon with Normal/Inverse-Wishart priors.

use crate::cluster::kmeans;
use crate::core::emission::EmissionModel;
use crate::core::hmm::Hmm;
use crate::core::inference::Em;
use crate::core::params::{DecoderAlgorithm, Implementation, Param, ParamSet};
use crate::core::{ConvergenceMonitor, Fitted};
use crate::covariance::{distribute_covar, validate_covars, CovarStore, CovarianceType};
use crate::error::{HmmError, Result};
use crate::rng::NumpyRandomState;
use crate::stats::{log_multivariate_normal_density, sample_covariance};
use ndarray::{Array1, Array2, Array3, ArrayView2, Axis};

/// Sufficient statistics for Gaussian emissions.
pub struct GaussianStats {
    /// Per-state posterior mass `Σₜ γₜ(c)`, shape `(n_components,)`.
    post: Array1<f64>,
    /// Posterior-weighted sum of observations `Σₜ γₜ(c)·xₜ`, shape `(n_components, n_features)`.
    obs: Array2<f64>,
    /// Posterior-weighted sum of squared observations (spherical/diag only), shape `(n_components, n_features)`.
    obs2: Array2<f64>,
    /// Posterior-weighted outer products `Σₜ γₜ(c)·xₜxₜᵀ` (tied/full only), shape `(n_components, n_features, n_features)`.
    obs_obs_t: Option<Array3<f64>>,
}

/// Gaussian emission model.
#[derive(Clone)]
pub struct GaussianEm {
    /// Number of hidden states.
    n_components: usize,
    /// Observation dimensionality; set on the first fit/init.
    n_features: Option<usize>,
    /// Covariance parameterization (spherical, diagonal, tied, or full).
    covariance_type: CovarianceType,
    /// State means, shape `(n_components, n_features)`.
    means: Option<Array2<f64>>,
    /// State covariances in compressed form.
    covars: Option<CovarStore>,
    /// Whether the means were supplied (skips re-initialization).
    means_preset: bool,
    /// Whether the covariances were supplied (skips re-initialization).
    covars_preset: bool,
    /// Floor added to the covariance diagonal at initialization.
    min_covar: f64,
    /// Scalar Normal-prior mean `μ₀` (broadcast when no per-state array is given).
    means_prior: f64,
    /// Optional per-state Normal-prior means, shape `(n_components, n_features)`.
    means_prior_arr: Option<Array2<f64>>,
    /// Normal-prior precision weight on the means.
    means_weight: f64,
    /// Inverse-gamma/Wishart prior scale on the covariances.
    covars_prior: f64,
    /// Prior weight (degrees of freedom) on the covariances.
    covars_weight: f64,
}

impl GaussianEm {
    /// The fitted means, shape `(n_components, n_features)`.
    ///
    /// # Panics
    /// If the means have not been initialized (model not fitted).
    pub fn means(&self) -> &Array2<f64> {
        self.means.as_ref().expect("means not initialized")
    }

    /// The compressed covariances (the `_covars_` form).
    ///
    /// # Panics
    /// If the covariances have not been initialized (model not fitted).
    pub fn covars(&self) -> &CovarStore {
        self.covars.as_ref().expect("covars not initialized")
    }

    /// The dense `(n_components, n_features, n_features)` covariances (`covars_`).
    ///
    /// # Panics
    /// If the covariances or `n_features` have not been initialized (model not fitted).
    pub fn covars_full(&self) -> Array3<f64> {
        self.covars()
            .full(self.n_components, self.n_features.unwrap())
    }

    /// The observation dimensionality.
    ///
    /// # Panics
    /// If `n_features` has not been set (no data seen and no preset params).
    fn nf(&self) -> usize {
        self.n_features.expect("n_features not set")
    }

    /// The per-state Normal-prior mean matrix, shape `(n_components, n_features)`.
    ///
    /// Returns the configured `means_prior_arr` when present, otherwise the scalar
    /// `means_prior` broadcast to every state and feature.
    ///
    /// # Panics
    /// If `n_features` has not been set (broadcast branch only).
    fn means_prior_arr(&self) -> Array2<f64> {
        match &self.means_prior_arr {
            Some(m) => m.clone(),
            None => Array2::from_elem((self.n_components, self.nf()), self.means_prior),
        }
    }
}

impl EmissionModel for GaussianEm {
    type Inference = Em;
    type Stats = GaussianStats;

    fn emission_params() -> &'static [Param] {
        &[Param::Means, Param::Covars]
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
        let nc = self.n_components;
        let nf = self.nf();
        if init.contains(Param::Means) || !self.means_preset {
            self.means = Some(kmeans(x, nc, 10, 300, seed));
        }
        if init.contains(Param::Covars) || !self.covars_preset {
            let mut cv = sample_covariance(x);
            for d in 0..nf {
                cv[[d, d]] += self.min_covar;
            }
            self.covars = Some(distribute_covar(cv.view(), self.covariance_type, nc));
        }
        self.means_preset = true;
        self.covars_preset = true;
        Ok(())
    }

    fn check(&self, n_components: usize) -> Result<()> {
        let means = self.means.as_ref().ok_or(HmmError::NotFitted)?;
        if means.nrows() != n_components {
            return Err(HmmError::DimensionMismatch(
                "means_ must have n_components rows".into(),
            ));
        }
        let covars = self.covars.as_ref().ok_or(HmmError::NotFitted)?;
        validate_covars(covars, n_components)
    }

    fn n_fit_scalars(&self, n_components: usize, params: ParamSet) -> usize {
        let nf = self.nf();
        let mut n = 0;
        if params.contains(Param::Means) {
            n += n_components * nf;
        }
        if params.contains(Param::Covars) {
            n += match self.covariance_type {
                CovarianceType::Spherical => n_components,
                CovarianceType::Diag => n_components * nf,
                CovarianceType::Full => n_components * nf * (nf + 1) / 2,
                CovarianceType::Tied => nf * (nf + 1) / 2,
            };
        }
        n
    }

    fn log_likelihood(&self, x: ArrayView2<f64>) -> Array2<f64> {
        log_multivariate_normal_density(x, self.means().view(), self.covars())
    }

    fn init_stats(&self) -> GaussianStats {
        let nc = self.n_components;
        let nf = self.nf();
        let obs_obs_t = matches!(
            self.covariance_type,
            CovarianceType::Tied | CovarianceType::Full
        )
        .then(|| Array3::zeros((nc, nf, nf)));
        GaussianStats {
            post: Array1::zeros(nc),
            obs: Array2::zeros((nc, nf)),
            obs2: Array2::zeros((nc, nf)),
            obs_obs_t,
        }
    }

    fn accumulate(
        &self,
        stats: &mut GaussianStats,
        x: ArrayView2<f64>,
        posteriors: ArrayView2<f64>,
        params: ParamSet,
    ) {
        let nc = self.n_components;
        let nf = self.nf();
        let ns = x.nrows();
        let needs_mean = params.contains(Param::Means);
        let needs_covar = params.contains(Param::Covars);
        if needs_mean {
            stats.post += &posteriors.sum_axis(Axis(0));
            stats.obs += &posteriors.t().dot(&x);
        }
        if needs_covar {
            match self.covariance_type {
                CovarianceType::Spherical | CovarianceType::Diag => {
                    let x2 = &x * &x;
                    stats.obs2 += &posteriors.t().dot(&x2);
                }
                CovarianceType::Tied | CovarianceType::Full => {
                    let oot = stats.obs_obs_t.as_mut().unwrap();
                    for t in 0..ns {
                        for c in 0..nc {
                            let p = posteriors[[t, c]];
                            if p == 0.0 {
                                continue;
                            }
                            for k in 0..nf {
                                let xk = x[[t, k]];
                                for l in 0..nf {
                                    oot[[c, k, l]] += p * xk * x[[t, l]];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    fn mstep(&mut self, stats: &GaussianStats, params: ParamSet) -> Result<()> {
        // Means are updated first; the covariance update reads the new means.
        if params.contains(Param::Means) {
            self.means = Some(self.update_means(stats));
        }
        if params.contains(Param::Covars) {
            self.covars = Some(self.update_covars(stats));
        }
        Ok(())
    }

    fn sample_state(&self, state: usize, rng: &mut NumpyRandomState) -> Array1<f64> {
        let cov = self.covars().covariance_of(state, self.nf());
        rng.multivariate_normal(self.means().row(state), cov.view())
    }
}

impl GaussianEm {
    /// Normal-prior mean update, per state and feature.
    ///
    /// Computes `(means_weight·μ₀ + Σ post·x) / (means_weight + Σ post)` with `μ₀`
    /// the prior mean, following Huang/Acero/Hon.
    ///
    /// # Arguments
    /// * `stats` — the accumulated sufficient statistics.
    ///
    /// # Returns
    /// The updated means, shape `(n_components, n_features)`.
    fn update_means(&self, stats: &GaussianStats) -> Array2<f64> {
        let prior = self.means_prior_arr();
        let weight = self.means_weight;
        Array2::from_shape_fn((self.n_components, self.nf()), |(c, f)| {
            (weight * prior[[c, f]] + stats.obs[[c, f]]) / (weight + stats.post[c])
        })
    }

    /// Covariance M-step: dispatches to the diagonal or dense update by covariance type.
    ///
    /// # Arguments
    /// * `stats` — the accumulated sufficient statistics.
    ///
    /// # Returns
    /// The updated covariances in compressed [`CovarStore`] form.
    fn update_covars(&self, stats: &GaussianStats) -> CovarStore {
        match self.covariance_type {
            CovarianceType::Spherical | CovarianceType::Diag => self.update_covars_diagonal(stats),
            CovarianceType::Tied | CovarianceType::Full => self.update_covars_dense(stats),
        }
    }

    /// Per-feature variance update for spherical/diagonal covariances (inverse-gamma prior).
    ///
    /// For each state/feature computes
    /// `(covars_prior + Nₙ) / max(max(covars_weight − 1, 0) + post, 1e-5)`, where
    /// `Nₙ = means_weight·(μ − μ₀)² + Σpost·x² − 2μ·Σpost·x + μ²·post`. Spherical
    /// covariances then average the per-feature variances across features.
    ///
    /// # Arguments
    /// * `stats` — the accumulated sufficient statistics.
    ///
    /// # Returns
    /// The updated covariances as [`CovarStore::Spherical`] or [`CovarStore::Diag`].
    fn update_covars_diagonal(&self, stats: &GaussianStats) -> CovarStore {
        let (nc, nf) = (self.n_components, self.nf());
        let means = self.means();
        let mean_prior = self.means_prior_arr();
        let (mean_weight, prior, weight) =
            (self.means_weight, self.covars_prior, self.covars_weight);

        let variance = Array2::from_shape_fn((nc, nf), |(c, f)| {
            let post = stats.post[c];
            let mean_diff = means[[c, f]] - mean_prior[[c, f]];
            let numer = mean_weight * mean_diff * mean_diff + stats.obs2[[c, f]]
                - 2.0 * means[[c, f]] * stats.obs[[c, f]]
                + means[[c, f]] * means[[c, f]] * post;
            let denom = ((weight - 1.0).max(0.0) + post).max(1e-5);
            (prior + numer) / denom
        });

        match self.covariance_type {
            CovarianceType::Spherical => {
                CovarStore::Spherical(variance.mean_axis(Axis(1)).unwrap())
            }
            _ => CovarStore::Diag(variance),
        }
    }

    /// Scatter-matrix update for tied/full covariances (inverse-Wishart prior).
    ///
    /// Builds the per-state scatter `means_weight·(μ − μ₀)(μ − μ₀)ᵀ + Σpost·xxᵀ
    /// − Σpost·x·μᵀ − μ·(Σpost·x)ᵀ + post·μμᵀ`, then divides
    /// `(covars_prior + scatter)` by `(max(covars_weight − n_features, 0) + post)`.
    /// Tied covariances pool the scatter and posterior mass across all states into
    /// a single shared matrix.
    ///
    /// # Arguments
    /// * `stats` — the accumulated sufficient statistics.
    ///
    /// # Returns
    /// The updated covariances as [`CovarStore::Tied`] or [`CovarStore::Full`].
    fn update_covars_dense(&self, stats: &GaussianStats) -> CovarStore {
        let (nc, nf) = (self.n_components, self.nf());
        let means = self.means();
        let mean_prior = self.means_prior_arr();
        let cross = stats.obs_obs_t.as_ref().expect("obs_obs_t for tied/full");
        let mean_weight = self.means_weight;
        let prior = self.covars_prior;
        let weight = (self.covars_weight - nf as f64).max(0.0);

        // Per-state scatter numerator: Σ post·(x - μ)(x - μ)ᵀ plus the prior term.
        let scatter = Array3::from_shape_fn((nc, nf, nf), |(c, k, l)| {
            let dk = means[[c, k]] - mean_prior[[c, k]];
            let dl = means[[c, l]] - mean_prior[[c, l]];
            mean_weight * dk * dl + cross[[c, k, l]]
                - stats.obs[[c, k]] * means[[c, l]]
                - stats.obs[[c, l]] * means[[c, k]]
                + means[[c, k]] * means[[c, l]] * stats.post[c]
        });

        match self.covariance_type {
            CovarianceType::Tied => {
                let post_sum = stats.post.sum();
                let tied = Array2::from_shape_fn((nf, nf), |(k, l)| {
                    let summed: f64 = (0..nc).map(|c| scatter[[c, k, l]]).sum();
                    (prior + summed) / (weight + post_sum)
                });
                CovarStore::Tied(tied)
            }
            _ => {
                let full = Array3::from_shape_fn((nc, nf, nf), |(c, k, l)| {
                    (prior + scatter[[c, k, l]]) / (weight + stats.post[c])
                });
                CovarStore::Full(full)
            }
        }
    }
}

/// Builder for [`GaussianEm`] HMMs (`hmmlearn.hmm.GaussianHMM`).
#[derive(Clone)]
pub struct GaussianHmm {
    /// Number of hidden states.
    n_components: usize,
    /// Covariance parameterization.
    covariance_type: CovarianceType,
    /// Floor added to the covariance diagonal at initialization.
    min_covar: f64,
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
    /// Scalar Normal-prior mean `μ₀` for the state means.
    means_prior: f64,
    /// Optional per-state Normal-prior means, shape `(n_components, n_features)`.
    means_prior_arr: Option<Array2<f64>>,
    /// Normal-prior precision weight on the means.
    means_weight: f64,
    /// Inverse-gamma/Wishart prior scale on the covariances.
    covars_prior: f64,
    /// Prior weight (degrees of freedom) on the covariances.
    covars_weight: f64,
    /// Optional preset initial-state probabilities.
    start_prob: Option<Array1<f64>>,
    /// Optional preset transition matrix.
    trans_mat: Option<Array2<f64>>,
    /// Optional preset means.
    means: Option<Array2<f64>>,
    /// Optional preset covariances.
    covars: Option<CovarStore>,
}

impl GaussianHmm {
    /// A model with `n_components` states, diagonal covariance, and hmmlearn's defaults.
    ///
    /// Defaults match `GaussianHMM`: diagonal covariance, `min_covar = 1e-3`, Viterbi
    /// decoding, log-space arithmetic, `params`/`init_params` = `"stmc"`, `n_iter = 10`,
    /// `tol = 1e-2`, non-verbose, no seed, `startprob_prior = transmat_prior = 1.0`,
    /// `means_prior = means_weight = 0.0`, `covars_prior = 1e-2`, `covars_weight = 1.0`.
    ///
    /// # Arguments
    /// * `n_components` — the number of hidden states.
    ///
    /// # Returns
    /// An unfitted builder carrying the defaults above.
    pub fn new(n_components: usize) -> Self {
        GaussianHmm {
            n_components,
            covariance_type: CovarianceType::Diag,
            min_covar: 1e-3,
            algorithm: DecoderAlgorithm::Viterbi,
            implementation: Implementation::Log,
            params: ParamSet::from_codes("stmc"),
            init_params: ParamSet::from_codes("stmc"),
            n_iter: 10,
            tol: 1e-2,
            verbose: false,
            random_state: None,
            startprob_prior: 1.0,
            transmat_prior: 1.0,
            means_prior: 0.0,
            means_prior_arr: None,
            means_weight: 0.0,
            covars_prior: 1e-2,
            covars_weight: 1.0,
            start_prob: None,
            trans_mat: None,
            means: None,
            covars: None,
        }
    }

    /// Sets the covariance parameterization (spherical, diagonal, tied, or full).
    pub fn covariance_type(mut self, ct: CovarianceType) -> Self {
        self.covariance_type = ct;
        self
    }
    /// Sets the floor added to the covariance diagonal at initialization.
    pub fn min_covar(mut self, v: f64) -> Self {
        self.min_covar = v;
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
    /// Sets which parameters are updated during EM (from letter codes, e.g. `"stmc"`).
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
    /// Sets the Normal-prior precision weight on the means.
    pub fn means_weight(mut self, w: f64) -> Self {
        self.means_weight = w;
        self
    }
    /// Sets the scalar Normal-prior mean `μ₀` for the state means.
    pub fn means_prior(mut self, p: f64) -> Self {
        self.means_prior = p;
        self
    }
    /// Sets per-state Normal-prior means, shape `(n_components, n_features)`.
    pub fn means_prior_array(mut self, p: Array2<f64>) -> Self {
        self.means_prior_arr = Some(p);
        self
    }
    /// Sets the inverse-gamma/Wishart prior scale on the covariances.
    pub fn covars_prior(mut self, p: f64) -> Self {
        self.covars_prior = p;
        self
    }
    /// Sets the prior weight (degrees of freedom) on the covariances.
    pub fn covars_weight(mut self, w: f64) -> Self {
        self.covars_weight = w;
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
    /// Presets the state means.
    pub fn means(mut self, m: Array2<f64>) -> Self {
        self.means = Some(m);
        self
    }
    /// Presets the state covariances.
    pub fn covars(mut self, c: CovarStore) -> Self {
        self.covars = Some(c);
        self
    }

    /// Assembles the configured builder into an unfitted [`Hmm`].
    ///
    /// Infers `n_features` from preset means, falling back to the prior-mean array
    /// when means are not supplied.
    ///
    /// # Returns
    /// An [`Hmm`] carrying the emission model, EM inference, and convergence monitor.
    pub(crate) fn build(self) -> Hmm<GaussianEm> {
        let n_features = self
            .means
            .as_ref()
            .map(|m| m.ncols())
            .or_else(|| self.means_prior_arr.as_ref().map(|m| m.ncols()));
        let inference = Em::new(
            self.n_components,
            self.startprob_prior,
            self.transmat_prior,
            self.start_prob,
            self.trans_mat,
        );
        let emission = GaussianEm {
            n_components: self.n_components,
            n_features,
            covariance_type: self.covariance_type,
            means_preset: self.means.is_some(),
            covars_preset: self.covars.is_some(),
            means: self.means,
            covars: self.covars,
            min_covar: self.min_covar,
            means_prior: self.means_prior,
            means_prior_arr: self.means_prior_arr,
            means_weight: self.means_weight,
            covars_prior: self.covars_prior,
            covars_weight: self.covars_weight,
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
    /// * `x` — observations, shape `(n_samples, n_features)`.
    /// * `lengths` — optional per-sequence lengths partitioning the rows of `x` into
    ///   independent sequences; `None` treats all rows as a single sequence.
    ///
    /// # Returns
    /// The [`Fitted`] model after EM.
    ///
    /// # Errors
    /// Forwards errors from initialization, validation, and the EM loop, e.g.
    /// [`HmmError::DimensionMismatch`] or [`HmmError::ScalingUnderflow`].
    pub fn fit(self, x: ArrayView2<f64>, lengths: Option<&[usize]>) -> Result<Fitted<GaussianEm>> {
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
    pub fn into_fitted(self) -> Result<Fitted<GaussianEm>> {
        self.build().into_fitted()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::{normalize1, normalize_axis};
    use ndarray::array;

    const IMPLS: [Implementation; 2] = [Implementation::Log, Implementation::Scaling];
    const TYPES: [CovarianceType; 4] = [
        CovarianceType::Spherical,
        CovarianceType::Diag,
        CovarianceType::Tied,
        CovarianceType::Full,
    ];

    struct Fixture {
        startprob: Array1<f64>,
        transmat: Array2<f64>,
        means: Array2<f64>,
        covars: CovarStore,
    }

    /// A symmetric positive-definite matrix from a random factor.
    fn spd(nf: usize, rng: &mut NumpyRandomState) -> Array2<f64> {
        let a = Array2::from_shape_vec((nf, nf), rng.random_sample_n(nf * nf).to_vec()).unwrap();
        let mut m = a.dot(&a.t());
        for d in 0..nf {
            m[[d, d]] += nf as f64; // ensure well-conditioned positive-definiteness
        }
        m
    }

    fn make_covar_matrix(
        ct: CovarianceType,
        nc: usize,
        nf: usize,
        rng: &mut NumpyRandomState,
    ) -> CovarStore {
        let mincv = 0.1;
        match ct {
            CovarianceType::Spherical => {
                let s = rng.random_sample_n(nc).mapv(|u| {
                    let v = mincv + mincv * u;
                    v * v
                });
                CovarStore::Spherical(s)
            }
            CovarianceType::Diag => {
                let d = rng.random_sample_n(nc * nf).mapv(|u| {
                    let v = mincv + mincv * u;
                    v * v
                });
                CovarStore::Diag(Array2::from_shape_vec((nc, nf), d.to_vec()).unwrap())
            }
            CovarianceType::Tied => {
                let mut m = spd(nf, rng);
                for d in 0..nf {
                    m[[d, d]] += mincv;
                }
                CovarStore::Tied(m)
            }
            CovarianceType::Full => {
                let mut full = Array3::zeros((nc, nf, nf));
                for c in 0..nc {
                    let mut m = spd(nf, rng);
                    for d in 0..nf {
                        m[[d, d]] += mincv;
                    }
                    full.slice_mut(ndarray::s![c, .., ..]).assign(&m);
                }
                CovarStore::Full(full)
            }
        }
    }

    fn fixture(ct: CovarianceType) -> Fixture {
        let nc = 3;
        let nf = 3;
        let mut rng = NumpyRandomState::new(10);
        let mut startprob = rng.random_sample_n(nc);
        normalize1(&mut startprob);
        let mut transmat =
            Array2::from_shape_vec((nc, nc), rng.random_sample_n(nc * nc).to_vec()).unwrap();
        normalize_axis(&mut transmat, 1);
        let means_i = rng.randint(-20, 20, nc * nf);
        let means = Array2::from_shape_vec((nc, nf), means_i.mapv(|v| v as f64).to_vec()).unwrap();
        let covars = make_covar_matrix(ct, nc, nf, &mut rng);
        Fixture {
            startprob,
            transmat,
            means,
            covars,
        }
    }

    /// `per_state` samples around each state's mean, with unit Gaussian noise.
    /// Modest identity-based covariances per type (a well-conditioned start).
    fn identity_covars(ct: CovarianceType, nc: usize, nf: usize) -> CovarStore {
        match ct {
            CovarianceType::Spherical => CovarStore::Spherical(Array1::ones(nc)),
            CovarianceType::Diag => CovarStore::Diag(Array2::ones((nc, nf))),
            CovarianceType::Tied => CovarStore::Tied(Array2::eye(nf)),
            CovarianceType::Full => {
                let mut f = Array3::zeros((nc, nf, nf));
                for c in 0..nc {
                    for d in 0..nf {
                        f[[c, d, d]] = 1.0;
                    }
                }
                CovarStore::Full(f)
            }
        }
    }

    fn clustered_data(means: &Array2<f64>, per_state: usize, seed: u32) -> Array2<f64> {
        let (nc, nf) = means.dim();
        let mut rng = NumpyRandomState::new(seed);
        let mut x = Array2::zeros((nc * per_state, nf));
        let mut row = 0;
        for c in 0..nc {
            for _ in 0..per_state {
                for f in 0..nf {
                    x[[row, f]] = means[[c, f]] + rng.standard_normal();
                }
                row += 1;
            }
        }
        x
    }

    #[test]
    fn bad_covariance_type_rejected() {
        // hmmlearn raises for an unknown covariance_type string.
        assert!("badcovariance_type".parse::<CovarianceType>().is_err());
    }

    #[test]
    fn score_samples_and_decode_recovers_states() {
        for ct in TYPES {
            for imp in IMPLS {
                let fx = fixture(ct);
                let means20 = &fx.means * 20.0;
                let per_state = 5;
                let nc = 3;
                let x = clustered_data(&means20, per_state, 99);
                let gaussidx: Vec<usize> = (0..nc * per_state).map(|i| i / per_state).collect();
                let h = GaussianHmm::new(nc)
                    .covariance_type(ct)
                    .implementation(imp)
                    .start_prob(fx.startprob.clone())
                    .trans_mat(fx.transmat.clone())
                    .means(means20)
                    .covars(fx.covars.clone())
                    .into_fitted()
                    .unwrap();
                let (_ll, post) = h.score_samples(x.view(), None).unwrap();
                assert_eq!(post.dim(), (nc * per_state, nc));
                for r in post.rows() {
                    assert!((r.sum() - 1.0).abs() < 1e-9);
                }
                let (_vll, seq) = h.decode(x.view(), None, None).unwrap();
                assert_eq!(seq.to_vec(), gaussidx);
            }
        }
    }

    #[test]
    fn sample_has_correct_shape() {
        for ct in TYPES {
            let fx = fixture(ct);
            let covars = match &fx.covars {
                CovarStore::Spherical(s) => CovarStore::Spherical(s.mapv(|v| v.max(0.1))),
                other => other.clone(),
            };
            let h = GaussianHmm::new(3)
                .covariance_type(ct)
                .start_prob(fx.startprob.clone())
                .trans_mat(fx.transmat.clone())
                .means(&fx.means * 20.0)
                .covars(covars)
                .into_fitted()
                .unwrap();
            let (x, states) = h.sample(1000, Some(7), None);
            assert_eq!(x.dim(), (1000, 3));
            assert_eq!(states.len(), 1000);
        }
    }

    #[test]
    fn fit_runs() {
        for ct in TYPES {
            for imp in IMPLS {
                let fx = fixture(ct);
                let h = GaussianHmm::new(3)
                    .covariance_type(ct)
                    .implementation(imp)
                    .start_prob(fx.startprob.clone())
                    .trans_mat(fx.transmat.clone())
                    .means(&fx.means * 20.0)
                    .covars(fx.covars.clone())
                    .into_fitted()
                    .unwrap();
                let (x, _) = h.sample(100, Some(3), None);
                let fitted = GaussianHmm::new(3)
                    .covariance_type(ct)
                    .implementation(imp)
                    .random_state(2)
                    .fit(x.view(), Some(&[10; 10]));
                assert!(fitted.is_ok());
            }
        }
    }

    #[test]
    fn criterion_aic_bic_positive() {
        for ct in TYPES {
            let fx = fixture(ct);
            let gen = GaussianHmm::new(3)
                .covariance_type(ct)
                .start_prob(fx.startprob.clone())
                .trans_mat(fx.transmat.clone())
                .means(&fx.means * 10.0)
                .covars(fx.covars.clone())
                .into_fitted()
                .unwrap();
            let (x, _) = gen.sample(500, Some(42), None);
            for n in [2usize, 3, 4] {
                let h = GaussianHmm::new(n)
                    .covariance_type(ct)
                    .n_iter(20)
                    .random_state(42)
                    .fit(x.view(), None)
                    .unwrap();
                // hmmlearn asserts `np.all(aic) > 0`, i.e. the values are
                // computed and non-zero (Gaussian log-densities can exceed 0,
                // so AIC itself may be negative).
                assert!(h.aic(x.view(), None).unwrap().is_finite());
                assert!(h.bic(x.view(), None).unwrap().is_finite());
            }
        }
    }

    #[test]
    fn fit_with_priors_recovers_means() {
        for ct in TYPES {
            for imp in IMPLS {
                let fx = fixture(ct);
                let means20 = &fx.means * 20.0;
                let mut covars_weight = 2.0;
                if matches!(ct, CovarianceType::Full | CovarianceType::Tied) {
                    covars_weight += 3.0;
                }
                // Balanced, well-separated data so every state is represented.
                let x = clustered_data(&means20, 200, 1);
                let lengths = vec![600usize];

                // Init only the means (k-means); start from modest covariances.
                // hmmlearn initializes covariances from the full-data sample
                // covariance, which (with our differing k-means/RNG) can drive a
                // degenerate fixed point; a well-conditioned start tests the
                // M-step + prior recovery, which is the property under test.
                let fitted = GaussianHmm::new(3)
                    .covariance_type(ct)
                    .implementation(imp)
                    .init_params("stm")
                    .covars(identity_covars(ct, 3, 3))
                    .means_prior_array(fx.means.clone())
                    .means_weight(2.0)
                    .covars_weight(covars_weight)
                    .n_iter(20)
                    // Run all iterations (MAP-EM data-LL can dip with strong
                    // priors); mirrors assert_log_likelihood_increasing's loop.
                    .tol(f64::NEG_INFINITY)
                    .random_state(1)
                    .fit(x.view(), Some(&lengths))
                    .unwrap();
                // recovery: sorted means within a generous relative tolerance
                let mut got: Vec<f64> = fitted.emission().means().iter().copied().collect();
                let mut want: Vec<f64> = means20.iter().copied().collect();
                got.sort_by(|a, b| a.partial_cmp(b).unwrap());
                want.sort_by(|a, b| a.partial_cmp(b).unwrap());
                for (g, w) in got.iter().zip(&want) {
                    assert!(
                        (g - w).abs() <= 0.05 * w.abs().max(1.0) + 1.0,
                        "mean recovery {ct:?}/{imp:?}: {g} vs {w}"
                    );
                }
            }
        }
    }

    #[test]
    fn fit_sequences_of_different_length() {
        for ct in TYPES {
            let mut rng = NumpyRandomState::new(3);
            let x = Array2::from_shape_vec((12, 3), rng.random_sample_n(36).to_vec()).unwrap();
            assert!(GaussianHmm::new(3)
                .covariance_type(ct)
                .random_state(1)
                .fit(x.view(), Some(&[3, 4, 5]))
                .is_ok());
        }
    }

    #[test]
    fn fit_with_length_one_signal() {
        for ct in TYPES {
            let mut rng = NumpyRandomState::new(4);
            let x = Array2::from_shape_vec((19, 3), rng.random_sample_n(57).to_vec()).unwrap();
            assert!(GaussianHmm::new(3)
                .covariance_type(ct)
                .random_state(1)
                .fit(x.view(), Some(&[10, 8, 1]))
                .is_ok());
        }
    }

    #[test]
    fn fit_zero_variance() {
        // GitHub issue #2: a constant feature must not break the fit.
        let data = array2_issue2();
        for ct in TYPES {
            assert!(GaussianHmm::new(3)
                .covariance_type(ct)
                .random_state(1)
                .fit(data.view(), None)
                .is_ok());
        }
    }

    fn array2_issue2() -> Array2<f64> {
        Array2::from_shape_vec(
            (9, 4),
            vec![
                715.0,
                585.0,
                0.0,
                0.0,
                715.0,
                520.0,
                1.04705811,
                -60.3696289,
                715.0,
                455.0,
                0.72088623,
                -52.7055664,
                715.0,
                390.0,
                -0.45794678,
                -78.0605469,
                715.0,
                325.0,
                -6.43127441,
                -55.9954834,
                715.0,
                260.0,
                -2.90063477,
                -78.0220947,
                715.0,
                195.0,
                8.45532227,
                -70.3294373,
                715.0,
                130.0,
                4.09387207,
                -58.3621216,
                715.0,
                65.0,
                -1.2166748,
                -44.8131409,
            ],
        )
        .unwrap()
    }

    #[test]
    fn issue_385_spherical_sample() {
        // A spherical model must sample without a shape error.
        let h = GaussianHmm::new(2)
            .covariance_type(CovarianceType::Spherical)
            .start_prob(array![0.6, 0.4])
            .trans_mat(array![[0.4, 0.6], [0.9, 0.1]])
            .means(array![[3.0], [5.0]])
            .covars(CovarStore::Spherical(array![4.0, 3.0]))
            .into_fitted()
            .unwrap();
        let (x, _) = h.sample(1000, Some(0), None);
        assert_eq!(x.dim(), (1000, 1));
    }

    #[test]
    fn underflow_from_scaling_errors() {
        // An outlier makes the scaling forward pass underflow; log is fine.
        let mut rng = NumpyRandomState::new(1234);
        let mut data: Vec<f64> = Vec::new();
        for (mean, n) in [(0.0, 100), (5.0, 100), (0.0, 100), (5.0, 100)] {
            for _ in 0..n {
                data.push(mean + rng.standard_normal());
            }
        }
        data[40] = 10000.0;
        let n = data.len();
        let x = Array2::from_shape_vec((n, 1), data).unwrap();

        let build = |imp| {
            GaussianHmm::new(2)
                .covariance_type(CovarianceType::Spherical)
                .implementation(imp)
                .init_params("")
                .n_iter(100)
                .start_prob(array![0.0, 1.0])
                .trans_mat(array![[0.4, 0.6], [0.6, 0.4]])
                .means(array![[0.0], [5.0]])
                .covars(CovarStore::Spherical(array![1.0, 1.0]))
        };
        assert!(matches!(
            build(Implementation::Scaling).fit(x.view(), Some(&[n])),
            Err(HmmError::ScalingUnderflow)
        ));
        assert!(build(Implementation::Log).fit(x.view(), Some(&[n])).is_ok());
    }

    #[test]
    fn fit_left_right_preserves_zeros() {
        let nc = 3;
        let mut transmat = Array2::zeros((nc, nc));
        for i in 0..nc {
            if i == nc - 1 {
                transmat[[i, i]] = 1.0;
            } else {
                transmat[[i, i]] = 0.5;
                transmat[[i, i + 1]] = 0.5;
            }
        }
        let mut startprob = Array1::zeros(nc);
        startprob[0] = 1.0;
        let mut rng = NumpyRandomState::new(5);
        let x = Array2::from_shape_vec((19, 3), rng.random_sample_n(57).to_vec()).unwrap();

        for imp in IMPLS {
            let fitted = GaussianHmm::new(nc)
                .covariance_type(CovarianceType::Diag)
                .implementation(imp)
                .params("mct")
                .init_params("cm")
                .random_state(1)
                .start_prob(startprob.clone())
                .trans_mat(transmat.clone())
                .fit(x.view(), Some(&[10, 8, 1]))
                .unwrap();
            // Structural zeros are preserved ('s' is not in params, so startprob
            // is untouched; 't' is fit but its zeros must stay zero).
            for i in 0..nc {
                if startprob[i] == 0.0 {
                    assert_eq!(fitted.start_prob()[i], 0.0);
                }
                for j in 0..nc {
                    if transmat[[i, j]] == 0.0 {
                        assert_eq!(fitted.trans_mat()[[i, j]], 0.0);
                    }
                }
            }
            let post = fitted.predict_proba(x.view(), Some(&[10, 8, 1])).unwrap();
            assert!(post.iter().all(|p| p.is_finite()));
            let (score, _seq) = fitted.decode(x.view(), Some(&[10, 8, 1]), None).unwrap();
            assert!(score.is_finite());
        }
    }
}
