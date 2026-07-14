//! Variational-Bayes Gaussian HMM — port of
//! `hmmlearn.vhmm.VariationalGaussianHMM`.
//!
//! The emission parameters are given conjugate priors: a Normal over each
//! state's mean (with precision scale `beta`) and a Wishart over each state's
//! precision (degrees of freedom `dof`, inverse scale matrix `scale`). Fitting
//! updates the posteriors of these distributions; the E-step scores frames with
//! the *sub-normalized* density (expectations under the posteriors), while
//! `score`/`decode` use the ordinary Gaussian density at the posterior point
//! estimate `means_posterior_` / `_covars_`.
//!
//! References: Gruhl & Sick (arXiv:1605.08618), McGrory & Titterington.

use crate::cluster::{cluster_counts, kmeans};
use crate::core::emission::EmissionModel;
use crate::core::hmm::Hmm;
use crate::core::inference::Variational;
use crate::core::params::{DecoderAlgorithm, Implementation, Param, ParamSet};
use crate::core::{ConvergenceMonitor, Fitted};
use crate::covariance::{distribute_covar, CovarStore, CovarianceType};
use crate::error::{HmmError, Result};
use crate::kl::{kl_multivariate_normal, kl_wishart};
use crate::linalg::{inv, logdet};
use crate::rng::NumpyRandomState;
use crate::special::digamma;
use crate::stats::{log_multivariate_normal_density, sample_covariance};
use ndarray::{Array1, Array2, Array3, ArrayView2, Axis};

/// Wishart degrees of freedom: one value per state (full/diag/spherical) or a
/// single shared value (tied).
#[derive(Debug, Clone, PartialEq)]
pub enum Dof {
    /// One degrees-of-freedom value per state (full/diag/spherical),
    /// length `n_components`.
    PerComponent(Array1<f64>),
    /// A single shared degrees-of-freedom scalar (tied covariance).
    Tied(f64),
}

impl Dof {
    /// The degrees of freedom associated with state `c`.
    fn get(&self, c: usize) -> f64 {
        match self {
            Dof::PerComponent(a) => a[c],
            Dof::Tied(v) => *v,
        }
    }

    /// The per-state degrees-of-freedom array (full/diag/spherical models).
    ///
    /// # Panics
    /// Panics (`unreachable!`) if called on a [`Dof::Tied`] value.
    fn per_component(&self) -> &Array1<f64> {
        match self {
            Dof::PerComponent(a) => a,
            Dof::Tied(_) => unreachable!("per-component dof for a tied model"),
        }
    }

    /// The single shared degrees-of-freedom scalar (tied model).
    ///
    /// # Panics
    /// Panics (`unreachable!`) if called on a [`Dof::PerComponent`] value.
    fn scalar(&self) -> f64 {
        match self {
            Dof::Tied(v) => *v,
            Dof::PerComponent(_) => unreachable!("scalar dof for a non-tied model"),
        }
    }
}

/// Emission sufficient statistics (identical shape to the EM Gaussian model).
pub struct VGaussianStats {
    /// Per-state posterior mass `Σₜ γ(t,c)`, shape `(n_components,)`.
    post: Array1<f64>,
    /// Posterior-weighted observation sums `Σₜ γ(t,c)·xₜ`, shape
    /// `(n_components, n_features)`.
    obs: Array2<f64>,
    /// Posterior-weighted squared-observation sums (spherical/diag), shape
    /// `(n_components, n_features)`.
    obs2: Array2<f64>,
    /// Posterior-weighted outer products `Σₜ γ(t,c)·xₜxₜᵀ` (tied/full), shape
    /// `(n_components, n_features, n_features)`; `None` for spherical/diag.
    obs_obs_t: Option<Array3<f64>>,
}

/// Variational Gaussian emission model.
#[derive(Clone)]
pub struct VariationalGaussianEm {
    n_components: usize,
    n_features: Option<usize>,
    covariance_type: CovarianceType,
    // Normal prior/posterior over the means.
    means_prior: Option<Array2<f64>>,
    means_posterior: Option<Array2<f64>>,
    beta_prior: Option<Array1<f64>>,
    beta_posterior: Option<Array1<f64>>,
    // Wishart prior/posterior over the precisions.
    dof_prior: Option<Dof>,
    dof_posterior: Option<Dof>,
    scale_prior: Option<CovarStore>,
    scale_posterior: Option<CovarStore>,
    // Point-estimate covariance `_covars_ = scale_posterior / dof_posterior`.
    covars: Option<CovarStore>,
    // Prior overrides supplied before fitting (constructor arguments).
    means_prior_arg: Option<Array2<f64>>,
    beta_prior_arg: Option<Array1<f64>>,
    dof_prior_arg: Option<Dof>,
    scale_prior_arg: Option<CovarStore>,
    means_preset: bool,
    covars_preset: bool,
}

impl VariationalGaussianEm {
    /// The feature dimensionality.
    ///
    /// # Panics
    /// Panics if `n_features` has not been set (no data seen / not initialized).
    fn nf(&self) -> usize {
        self.n_features.expect("n_features not set")
    }

    /// The posterior mean estimates, shape `(n_components, n_features)`.
    ///
    /// # Panics
    /// Panics if the means have not been initialized.
    pub fn means_posterior(&self) -> &Array2<f64> {
        self.means_posterior
            .as_ref()
            .expect("means not initialized")
    }
    /// The Normal prior means `μ₀`, shape `(n_components, n_features)`.
    ///
    /// # Panics
    /// Panics if the means prior has not been initialized.
    fn means_prior(&self) -> &Array2<f64> {
        self.means_prior
            .as_ref()
            .expect("means_prior not initialized")
    }
    /// The Normal prior mean-precision scalings `β₀`, shape `(n_components,)`.
    ///
    /// # Panics
    /// Panics if the beta prior has not been initialized.
    fn beta_prior(&self) -> &Array1<f64> {
        self.beta_prior
            .as_ref()
            .expect("beta_prior not initialized")
    }
    /// The Normal posterior mean-precision scalings `β`, shape `(n_components,)`.
    ///
    /// # Panics
    /// Panics if the beta posterior has not been initialized.
    fn beta_posterior(&self) -> &Array1<f64> {
        self.beta_posterior
            .as_ref()
            .expect("beta_posterior not initialized")
    }
    /// The Wishart inverse-scale prior over the precisions.
    ///
    /// # Panics
    /// Panics if the scale prior has not been initialized.
    fn scale_prior(&self) -> &CovarStore {
        self.scale_prior
            .as_ref()
            .expect("scale_prior not initialized")
    }
    /// The Wishart inverse-scale posterior over the precisions.
    ///
    /// # Panics
    /// Panics if the scale posterior has not been initialized.
    fn scale_posterior(&self) -> &CovarStore {
        self.scale_posterior
            .as_ref()
            .expect("scale_posterior not initialized")
    }
    /// The Wishart degrees-of-freedom prior over the precisions.
    ///
    /// # Panics
    /// Panics if the dof prior has not been initialized.
    fn dof_prior(&self) -> &Dof {
        self.dof_prior.as_ref().expect("dof_prior not initialized")
    }
    /// The Wishart degrees-of-freedom posterior over the precisions.
    ///
    /// # Panics
    /// Panics if the dof posterior has not been initialized.
    fn dof_posterior(&self) -> &Dof {
        self.dof_posterior
            .as_ref()
            .expect("dof_posterior not initialized")
    }
    /// The compressed point-estimate covariances (`_covars_`).
    ///
    /// # Panics
    /// Panics if the covariances have not been initialized.
    pub fn covars(&self) -> &CovarStore {
        self.covars.as_ref().expect("covars not initialized")
    }

    /// `scale_posterior / dof_posterior`, the covariance point estimate.
    ///
    /// # Panics
    /// Panics if `scale_posterior`/`dof_posterior` are uninitialized, or
    /// (`unreachable!`) if their shapes are inconsistent with `covariance_type`
    /// (`Tied` scale paired with `Tied` dof, otherwise per-component).
    fn point_covariance(&self) -> CovarStore {
        let nc = self.n_components;
        match (self.scale_posterior(), self.dof_posterior()) {
            (CovarStore::Full(s), Dof::PerComponent(d)) => {
                CovarStore::Full(Array3::from_shape_fn(s.dim(), |(c, k, l)| {
                    s[[c, k, l]] / d[c]
                }))
            }
            (CovarStore::Diag(s), Dof::PerComponent(d)) => {
                CovarStore::Diag(Array2::from_shape_fn(s.dim(), |(c, j)| s[[c, j]] / d[c]))
            }
            (CovarStore::Spherical(s), Dof::PerComponent(d)) => {
                CovarStore::Spherical(Array1::from_shape_fn(nc, |c| s[c] / d[c]))
            }
            (CovarStore::Tied(s), Dof::Tied(d)) => CovarStore::Tied(s.mapv(|v| v / d)),
            _ => unreachable!("scale/dof shapes are consistent with covariance_type"),
        }
    }
}

impl EmissionModel for VariationalGaussianEm {
    type Inference = Variational;
    type Stats = VGaussianStats;

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
        let means_needs = init.contains(Param::Means) || !self.means_preset;
        let covars_needs = init.contains(Param::Covars) || !self.covars_preset;
        if !means_needs && !covars_needs {
            return Ok(());
        }

        // k-means seeds both the means and the covariance scale; the cluster
        // sizes seed the beta/dof posteriors.
        let centers = kmeans(x, nc, 10, 300, seed);
        let counts = cluster_counts(x, centers.view());

        if means_needs {
            let x_mean = x.mean_axis(Axis(0)).unwrap();
            self.means_prior = Some(match &self.means_prior_arg {
                Some(m) => m.clone(),
                None => Array2::from_shape_fn((nc, nf), |(_, j)| x_mean[j]),
            });
            self.means_posterior = Some(centers.clone());
            self.beta_prior = Some(
                self.beta_prior_arg
                    .clone()
                    .unwrap_or_else(|| Array1::ones(nc)),
            );
            self.beta_posterior = Some(counts.clone());
        }

        if covars_needs {
            self.dof_prior =
                Some(
                    self.dof_prior_arg
                        .clone()
                        .unwrap_or_else(|| match self.covariance_type {
                            CovarianceType::Tied => Dof::Tied(nf as f64),
                            _ => Dof::PerComponent(Array1::from_elem(nc, nf as f64)),
                        }),
                );
            self.dof_posterior = Some(match self.covariance_type {
                CovarianceType::Tied => Dof::Tied(counts.sum()),
                _ => Dof::PerComponent(counts.clone()),
            });

            // Covariance estimate from the data, distributed to the type's shape.
            let mut cv = sample_covariance(x);
            for d in 0..nf {
                cv[[d, d]] += 1e-3;
            }
            let covars = distribute_covar(cv.view(), self.covariance_type, nc);
            self.scale_prior = Some(
                self.scale_prior_arg
                    .clone()
                    .unwrap_or_else(|| default_scale_prior(self.covariance_type, nc, nf)),
            );
            self.scale_posterior =
                Some(scale_from_covariance(&covars, self.dof_posterior(), nc, nf));
            self.covars = Some(covars);
        }

        self.means_preset = true;
        self.covars_preset = true;
        Ok(())
    }

    fn check(&self, n_components: usize) -> Result<()> {
        let nf = self.nf();
        let means_shape = (n_components, nf);
        if self.means_prior().dim() != means_shape || self.means_posterior().dim() != means_shape {
            return Err(HmmError::DimensionMismatch(
                "means prior/posterior must have shape (n_components, n_features)".into(),
            ));
        }
        if self.beta_prior().len() != n_components || self.beta_posterior().len() != n_components {
            return Err(HmmError::DimensionMismatch(
                "beta prior/posterior must have length n_components".into(),
            ));
        }
        check_dof_shape(self.dof_prior(), self.covariance_type, n_components)?;
        check_dof_shape(self.dof_posterior(), self.covariance_type, n_components)?;
        check_scale_shape(self.scale_prior(), self.covariance_type)?;
        check_scale_shape(self.scale_posterior(), self.covariance_type)?;
        Ok(())
    }

    fn n_fit_scalars(&self, n_components: usize, params: ParamSet) -> usize {
        let nf = self.nf();
        let mut n = 0;
        if params.contains(Param::Means) {
            n += n_components * nf + n_components;
        }
        if params.contains(Param::Covars) {
            n += match self.covariance_type {
                CovarianceType::Full => n_components + n_components * nf * (nf + 1) / 2,
                CovarianceType::Tied => 1 + nf * (nf + 1) / 2,
                CovarianceType::Diag => n_components + n_components * nf,
                CovarianceType::Spherical => n_components + n_components,
            };
        }
        n
    }

    /// Scoring: the ordinary Gaussian density at the posterior point estimate.
    fn log_likelihood(&self, x: ArrayView2<f64>) -> Array2<f64> {
        log_multivariate_normal_density(x, self.means_posterior().view(), self.covars())
    }

    /// Fitting: the sub-normalized log-density (Gruhl & Sick), the expectation
    /// of `log N(x | μ, Σ)` under the Normal/Wishart posteriors.
    ///
    /// # Panics
    /// Panics if the posteriors are uninitialized, or if a per-state
    /// `scale_posterior` block is not invertible.
    fn fit_log_likelihood(&self, x: ArrayView2<f64>) -> Array2<f64> {
        let nc = self.n_components;
        let nf = self.nf();
        let ns = x.nrows();
        let means = self.means_posterior();
        let beta = self.beta_posterior();
        let ln2 = std::f64::consts::LN_2;

        // Per-state precision W_k = scale_posterior⁻¹ and the constant term1.
        let mut w_k: Vec<Array2<f64>> = Vec::with_capacity(nc);
        let mut dof: Vec<f64> = Vec::with_capacity(nc);
        let mut term1: Vec<f64> = Vec::with_capacity(nc);
        for c in 0..nc {
            let scale_c = self.scale_posterior().covariance_of(c, nf);
            let wk = inv(scale_c.view()).expect("scale_posterior invertible");
            let dof_c = self.dof_posterior().get(c);
            let digamma_sum: f64 = (0..nf).map(|d| digamma(0.5 * (dof_c - d as f64))).sum();
            term1.push(0.5 * (digamma_sum + nf as f64 * ln2 + logdet(wk.view())));
            w_k.push(wk);
            dof.push(dof_c);
        }

        Array2::from_shape_fn((ns, nc), |(t, c)| {
            let delta = Array1::from_shape_fn(nf, |j| x[[t, j]] - means[[c, j]]);
            let quad = delta.dot(&w_k[c].dot(&delta));
            term1[c] - 0.5 * (dof[c] * quad + nf as f64 / beta[c])
        })
    }

    /// Scaling-path fitting likelihood: the exponentiated sub-normalized density.
    fn fit_likelihood(&self, x: ArrayView2<f64>) -> Array2<f64> {
        self.fit_log_likelihood(x).mapv(f64::exp)
    }

    fn init_stats(&self) -> VGaussianStats {
        let nc = self.n_components;
        let nf = self.nf();
        let obs_obs_t = matches!(
            self.covariance_type,
            CovarianceType::Tied | CovarianceType::Full
        )
        .then(|| Array3::zeros((nc, nf, nf)));
        VGaussianStats {
            post: Array1::zeros(nc),
            obs: Array2::zeros((nc, nf)),
            obs2: Array2::zeros((nc, nf)),
            obs_obs_t,
        }
    }

    fn accumulate(
        &self,
        stats: &mut VGaussianStats,
        x: ArrayView2<f64>,
        posteriors: ArrayView2<f64>,
        params: ParamSet,
    ) {
        // The covariance update reads `post`/`obs`, so the mean statistics are
        // needed whenever means *or* covariances are being fit.
        let needs_mean = params.contains(Param::Means) || params.contains(Param::Covars);
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
                    let (nc, nf) = (self.n_components, self.nf());
                    for t in 0..x.nrows() {
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

    fn mstep(&mut self, stats: &VGaussianStats, params: ParamSet) -> Result<()> {
        // Means are updated first; the covariance update reads the new means.
        if params.contains(Param::Means) {
            self.update_means(stats);
        }
        if params.contains(Param::Covars) {
            self.update_covars(stats);
            self.covars = Some(self.point_covariance());
        }
        Ok(())
    }

    fn lower_bound_contribution(&self) -> f64 {
        let nc = self.n_components;
        let nf = self.nf();
        let (means_q, means_p) = (self.means_posterior(), self.means_prior());
        let (beta_q, beta_p) = (self.beta_posterior(), self.beta_prior());
        let mut lb = 0.0;
        for i in 0..nc {
            let scale_q = self.scale_posterior().covariance_of(i, nf);
            let scale_p = self.scale_prior().covariance_of(i, nf);
            let dof_q = self.dof_posterior().get(i);
            let precision = inv(scale_q.view())
                .expect("scale_posterior invertible")
                .mapv(|v| v * dof_q);
            // KL between the Normal posterior and prior over the mean.
            let covar_q = inv(precision.mapv(|v| v * beta_q[i]).view()).unwrap();
            let covar_p = inv(precision.mapv(|v| v * beta_p[i]).view()).unwrap();
            lb -= kl_multivariate_normal(
                means_q.row(i),
                covar_q.view(),
                means_p.row(i),
                covar_p.view(),
            );
            // KL between the Wishart posterior and prior over the precision.
            let klw = match self.covariance_type {
                CovarianceType::Tied if i > 0 => 0.0,
                CovarianceType::Tied => kl_wishart(
                    self.dof_posterior().scalar(),
                    scale_q.view(),
                    self.dof_prior().scalar(),
                    scale_p.view(),
                ),
                _ => kl_wishart(
                    dof_q,
                    scale_q.view(),
                    self.dof_prior().get(i),
                    scale_p.view(),
                ),
            };
            lb -= klw;
        }
        lb
    }

    fn sample_state(&self, state: usize, rng: &mut NumpyRandomState) -> Array1<f64> {
        let cov = self.covars().covariance_of(state, self.nf());
        rng.multivariate_normal(self.means_posterior().row(state), cov.view())
    }
}

impl VariationalGaussianEm {
    /// Normal-Wishart mean update:
    /// `μ = (β₀ μ₀ + Σ post·x) / (β₀ + Σ post)`, `β = β₀ + Σ post`.
    fn update_means(&mut self, stats: &VGaussianStats) {
        let (nc, nf) = (self.n_components, self.nf());
        let beta_prior = self.beta_prior().clone();
        let means_prior = self.means_prior().clone();
        let beta_post = &beta_prior + &stats.post;
        let means_post = Array2::from_shape_fn((nc, nf), |(c, j)| {
            (beta_prior[c] * means_prior[[c, j]] + stats.obs[[c, j]]) / beta_post[c]
        });
        self.beta_posterior = Some(beta_post);
        self.means_posterior = Some(means_post);
    }

    /// Wishart scale/dof update. The scale posterior gathers the prior scale,
    /// the observed scatter, and the prior/posterior mean-outer-product terms.
    ///
    /// # Panics
    /// Panics if the priors/posteriors are uninitialized; if `stats.obs_obs_t`
    /// is absent for tied/full covariance; or (`unreachable!`) if the `Dof`
    /// variant or scale store does not match `covariance_type`.
    fn update_covars(&mut self, stats: &VGaussianStats) {
        let (nc, nf) = (self.n_components, self.nf());
        let bp = self.beta_prior().clone();
        let bq = self.beta_posterior().clone();
        let mp = self.means_prior().clone();
        let mq = self.means_posterior().clone();
        let post = &stats.post;

        match self.covariance_type {
            CovarianceType::Full => {
                let cross = stats.obs_obs_t.as_ref().unwrap();
                let sp = self.scale_prior().full(nc, nf);
                let scale = Array3::from_shape_fn((nc, nf, nf), |(c, k, l)| {
                    sp[[c, k, l]] + cross[[c, k, l]] + bp[c] * mp[[c, k]] * mp[[c, l]]
                        - bq[c] * mq[[c, k]] * mq[[c, l]]
                });
                self.dof_posterior =
                    Some(Dof::PerComponent(self.dof_prior().per_component() + post));
                self.scale_posterior = Some(CovarStore::Full(scale));
            }
            CovarianceType::Tied => {
                let cross = stats.obs_obs_t.as_ref().unwrap();
                let sp = self.scale_prior().covariance_of(0, nf);
                let scale = Array2::from_shape_fn((nf, nf), |(k, l)| {
                    let scatter: f64 = (0..nc)
                        .map(|c| {
                            cross[[c, k, l]] + bp[c] * mp[[c, k]] * mp[[c, l]]
                                - bq[c] * mq[[c, k]] * mq[[c, l]]
                        })
                        .sum();
                    sp[[k, l]] + scatter
                });
                self.dof_posterior = Some(Dof::Tied(self.dof_prior().scalar() + post.sum()));
                self.scale_posterior = Some(CovarStore::Tied(scale));
            }
            CovarianceType::Diag => {
                let sp = diag_arr(self.scale_prior());
                let scale = Array2::from_shape_fn((nc, nf), |(c, j)| {
                    sp[[c, j]] + stats.obs2[[c, j]] + bp[c] * mp[[c, j]] * mp[[c, j]]
                        - bq[c] * mq[[c, j]] * mq[[c, j]]
                });
                self.dof_posterior =
                    Some(Dof::PerComponent(self.dof_prior().per_component() + post));
                self.scale_posterior = Some(CovarStore::Diag(scale));
            }
            CovarianceType::Spherical => {
                let sp = spherical_arr(self.scale_prior());
                let scale = Array1::from_shape_fn(nc, |c| {
                    let term: f64 = (0..nf)
                        .map(|j| {
                            stats.obs2[[c, j]] + bp[c] * mp[[c, j]] * mp[[c, j]]
                                - bq[c] * mq[[c, j]] * mq[[c, j]]
                        })
                        .sum::<f64>()
                        / nf as f64;
                    sp[c] + term
                });
                self.dof_posterior =
                    Some(Dof::PerComponent(self.dof_prior().per_component() + post));
                self.scale_posterior = Some(CovarStore::Spherical(scale));
            }
        }
    }
}

/// The default Wishart inverse-scale prior (`1e-3` times the identity).
fn default_scale_prior(ct: CovarianceType, nc: usize, nf: usize) -> CovarStore {
    let eps = 1e-3;
    match ct {
        CovarianceType::Full => {
            let mut f = Array3::zeros((nc, nf, nf));
            for c in 0..nc {
                for d in 0..nf {
                    f[[c, d, d]] = eps;
                }
            }
            CovarStore::Full(f)
        }
        CovarianceType::Tied => CovarStore::Tied(Array2::from_shape_fn((nf, nf), |(k, l)| {
            if k == l {
                eps
            } else {
                0.0
            }
        })),
        CovarianceType::Diag => CovarStore::Diag(Array2::from_elem((nc, nf), eps)),
        CovarianceType::Spherical => CovarStore::Spherical(Array1::from_elem(nc, eps)),
    }
}

/// The initial scale posterior, `covariance · dof` in the type's native shape.
///
/// # Arguments
/// * `covars` — the point-estimate covariances to scale.
/// * `dof` — the degrees-of-freedom posterior to multiply by.
/// * `nc` — number of states.
/// * `nf` — feature dimension.
///
/// # Returns
/// A `CovarStore` of the same parameterization as `covars`, each entry scaled
/// by the matching degrees of freedom.
///
/// # Panics
/// Panics (`unreachable!`) if the `dof` variant does not match `covars`
/// (per-component dof for the non-tied stores, a scalar dof for `Tied`).
fn scale_from_covariance(covars: &CovarStore, dof: &Dof, nc: usize, nf: usize) -> CovarStore {
    match covars {
        CovarStore::Full(c) => {
            let d = dof.per_component();
            CovarStore::Full(Array3::from_shape_fn((nc, nf, nf), |(i, k, l)| {
                c[[i, k, l]] * d[i]
            }))
        }
        CovarStore::Diag(c) => {
            let d = dof.per_component();
            CovarStore::Diag(Array2::from_shape_fn((nc, nf), |(i, j)| c[[i, j]] * d[i]))
        }
        CovarStore::Spherical(c) => {
            let d = dof.per_component();
            CovarStore::Spherical(Array1::from_shape_fn(nc, |i| c[i] * d[i]))
        }
        CovarStore::Tied(c) => CovarStore::Tied(c.mapv(|v| v * dof.scalar())),
    }
}

/// The per-feature variance array backing a `CovarStore::Diag`.
///
/// # Panics
/// Panics (`unreachable!`) unless `cs` is [`CovarStore::Diag`].
fn diag_arr(cs: &CovarStore) -> &Array2<f64> {
    match cs {
        CovarStore::Diag(a) => a,
        _ => unreachable!("covariance_type is diag"),
    }
}

/// The scalar-variance array backing a `CovarStore::Spherical`.
///
/// # Panics
/// Panics (`unreachable!`) unless `cs` is [`CovarStore::Spherical`].
fn spherical_arr(cs: &CovarStore) -> &Array1<f64> {
    match cs {
        CovarStore::Spherical(a) => a,
        _ => unreachable!("covariance_type is spherical"),
    }
}

/// Validate that a `Dof` has the shape required by `covariance_type`.
///
/// # Arguments
/// * `dof` — the degrees-of-freedom value to check.
/// * `ct` — the covariance parameterization it must match.
/// * `nc` — expected number of states for the per-component case.
///
/// # Errors
/// [`HmmError::DimensionMismatch`] if `ct` is `Tied` but `dof` is not
/// [`Dof::Tied`] (or `dof` is `Tied` for a non-tied `ct`), or if a
/// per-component array's length is not `nc`.
fn check_dof_shape(dof: &Dof, ct: CovarianceType, nc: usize) -> Result<()> {
    let ok = match (dof, ct) {
        (Dof::Tied(_), CovarianceType::Tied) => true,
        (Dof::PerComponent(a), ct) if ct != CovarianceType::Tied => a.len() == nc,
        _ => false,
    };
    ok.then_some(()).ok_or_else(|| {
        HmmError::DimensionMismatch("dof has the wrong shape for covariance_type".into())
    })
}

/// Validate that a scale `CovarStore` matches `covariance_type`.
///
/// # Arguments
/// * `scale` — the inverse-scale store to check.
/// * `ct` — the covariance parameterization it must match.
///
/// # Errors
/// [`HmmError::DimensionMismatch`] if the store's covariance type is not `ct`.
fn check_scale_shape(scale: &CovarStore, ct: CovarianceType) -> Result<()> {
    (scale.covariance_type() == ct)
        .then_some(())
        .ok_or_else(|| {
            HmmError::DimensionMismatch("scale has the wrong shape for covariance_type".into())
        })
}

/// Builder for [`VariationalGaussianEm`] HMMs (`hmmlearn.vhmm.VariationalGaussianHMM`).
///
/// Unlike the EM builders there is no `into_fitted`: a variational model is
/// always obtained by fitting.
#[derive(Clone)]
pub struct VariationalGaussianHmm {
    n_components: usize,
    covariance_type: CovarianceType,
    algorithm: DecoderAlgorithm,
    implementation: Implementation,
    params: ParamSet,
    init_params: ParamSet,
    n_iter: usize,
    tol: f64,
    verbose: bool,
    random_state: Option<u32>,
    startprob_prior: Option<Array1<f64>>,
    startprob_posterior: Option<Array1<f64>>,
    transmat_prior: Option<Array2<f64>>,
    transmat_posterior: Option<Array2<f64>>,
    means_prior: Option<Array2<f64>>,
    means_posterior: Option<Array2<f64>>,
    beta_prior: Option<Array1<f64>>,
    beta_posterior: Option<Array1<f64>>,
    dof_prior: Option<Dof>,
    scale_prior: Option<CovarStore>,
}

impl VariationalGaussianHmm {
    /// A model with `n_components` states, full covariance, hmmlearn defaults.
    ///
    /// Matches `VariationalGaussianHMM`'s defaults: `covariance_type = full`,
    /// `n_iter = 100`, `tol = 1e-6`, log-space forward-backward, all parameters
    /// (`stmc`) both initialized and updated, and no random seed. The
    /// Normal-Wishart priors are left unset and derived from the data at `init`.
    ///
    /// # Arguments
    /// * `n_components` — number of hidden states.
    ///
    /// # Returns
    /// An unfitted builder.
    pub fn new(n_components: usize) -> Self {
        VariationalGaussianHmm {
            n_components,
            covariance_type: CovarianceType::Full,
            algorithm: DecoderAlgorithm::Viterbi,
            implementation: Implementation::Log,
            params: ParamSet::from_codes("stmc"),
            init_params: ParamSet::from_codes("stmc"),
            n_iter: 100,
            tol: 1e-6,
            verbose: false,
            random_state: None,
            startprob_prior: None,
            startprob_posterior: None,
            transmat_prior: None,
            transmat_posterior: None,
            means_prior: None,
            means_posterior: None,
            beta_prior: None,
            beta_posterior: None,
            dof_prior: None,
            scale_prior: None,
        }
    }

    /// Sets the covariance parameterization (spherical, diag, full, or tied).
    pub fn covariance_type(mut self, ct: CovarianceType) -> Self {
        self.covariance_type = ct;
        self
    }
    /// Sets the forward-backward implementation (log-space or scaling).
    pub fn implementation(mut self, i: Implementation) -> Self {
        self.implementation = i;
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
    /// Sets which parameters are updated each iteration (`s`/`t`/`m`/`c`).
    pub fn params(mut self, codes: &str) -> Self {
        self.params = ParamSet::from_codes(codes);
        self
    }
    /// Sets which parameters are initialized before fitting (`s`/`t`/`m`/`c`).
    pub fn init_params(mut self, codes: &str) -> Self {
        self.init_params = ParamSet::from_codes(codes);
        self
    }
    /// Sets the random seed used for initialization.
    pub fn random_state(mut self, seed: u32) -> Self {
        self.random_state = Some(seed);
        self
    }
    /// Enables per-iteration convergence reporting.
    pub fn verbose(mut self, v: bool) -> Self {
        self.verbose = v;
        self
    }
    /// Sets the Dirichlet concentration prior on the initial-state distribution.
    pub fn startprob_prior(mut self, p: Array1<f64>) -> Self {
        self.startprob_prior = Some(p);
        self
    }
    /// Presets the Dirichlet posterior over the initial-state distribution.
    pub fn startprob_posterior(mut self, p: Array1<f64>) -> Self {
        self.startprob_posterior = Some(p);
        self
    }
    /// Sets the Dirichlet concentration prior on the transition-matrix rows.
    pub fn transmat_prior(mut self, p: Array2<f64>) -> Self {
        self.transmat_prior = Some(p);
        self
    }
    /// Presets the Dirichlet posterior over the transition-matrix rows.
    pub fn transmat_posterior(mut self, p: Array2<f64>) -> Self {
        self.transmat_posterior = Some(p);
        self
    }
    /// Sets the mean `μ₀` of the Normal prior over each state's mean.
    pub fn means_prior(mut self, m: Array2<f64>) -> Self {
        self.means_prior = Some(m);
        self
    }
    /// Presets the Normal posterior means (the point estimate used for scoring).
    pub fn means_posterior(mut self, m: Array2<f64>) -> Self {
        self.means_posterior = Some(m);
        self
    }
    /// Sets the mean-precision scaling `β₀` of the Normal prior over the means.
    pub fn beta_prior(mut self, b: Array1<f64>) -> Self {
        self.beta_prior = Some(b);
        self
    }
    /// Presets the mean-precision scaling `β` of the Normal posterior.
    pub fn beta_posterior(mut self, b: Array1<f64>) -> Self {
        self.beta_posterior = Some(b);
        self
    }
    /// Sets the Wishart degrees-of-freedom prior over the precisions.
    pub fn dof_prior(mut self, d: Dof) -> Self {
        self.dof_prior = Some(d);
        self
    }
    /// Sets the Wishart inverse-scale prior over the precisions.
    pub fn scale_prior(mut self, s: CovarStore) -> Self {
        self.scale_prior = Some(s);
        self
    }

    /// Assemble the configured builder into an unfitted [`Hmm`].
    ///
    /// Derives `n_features` from any preset posterior/prior means, records which
    /// emission parameters are preset (so `init` can skip them), and packages the
    /// variational inference core with the Normal-Wishart emission model.
    ///
    /// # Returns
    /// An unfitted [`Hmm`] wrapping a [`VariationalGaussianEm`].
    pub(crate) fn build(self) -> Hmm<VariationalGaussianEm> {
        let n_features = self
            .means_posterior
            .as_ref()
            .or(self.means_prior.as_ref())
            .map(|m| m.ncols());
        let means_preset = self.means_posterior.is_some();
        let covars_preset = self.scale_prior.is_some() && self.means_posterior.is_some();
        let inference = Variational::new(
            self.n_components,
            None,
            None,
            self.startprob_prior,
            self.startprob_posterior,
            self.transmat_prior,
            self.transmat_posterior,
        );
        let emission = VariationalGaussianEm {
            n_components: self.n_components,
            n_features,
            covariance_type: self.covariance_type,
            means_prior: self.means_prior.clone(),
            means_posterior: self.means_posterior,
            beta_prior: self.beta_prior.clone(),
            beta_posterior: self.beta_posterior,
            dof_prior: self.dof_prior.clone(),
            dof_posterior: None,
            scale_prior: self.scale_prior.clone(),
            scale_posterior: None,
            covars: None,
            means_prior_arg: self.means_prior,
            beta_prior_arg: self.beta_prior,
            dof_prior_arg: self.dof_prior,
            scale_prior_arg: self.scale_prior,
            means_preset,
            covars_preset,
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
    /// * `x` — observation matrix, `(n_samples, n_features)`.
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
    ) -> Result<Fitted<VariationalGaussianEm>> {
        self.build().fit(x, lengths)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::GaussianHmm;
    use crate::testutil::assert_ll_increasing;
    use ndarray::array;

    const IMPLS: [Implementation; 2] = [Implementation::Log, Implementation::Scaling];
    const TYPES: [CovarianceType; 4] = [
        CovarianceType::Spherical,
        CovarianceType::Diag,
        CovarianceType::Tied,
        CovarianceType::Full,
    ];

    /// Three well-separated 3-D states; samples generated by an EM Gaussian model.
    fn sample_data(ct: CovarianceType, n: usize, seed: u32) -> Array2<f64> {
        let means = array![[0.0, 0.0, 0.0], [12.0, 12.0, 12.0], [24.0, 0.0, 12.0]];
        let covars = match ct {
            CovarianceType::Spherical => CovarStore::Spherical(array![1.0, 1.5, 0.8]),
            CovarianceType::Diag => {
                CovarStore::Diag(array![[1.0, 1.2, 0.9], [0.8, 1.1, 1.3], [1.4, 0.7, 1.0]])
            }
            CovarianceType::Tied => {
                CovarStore::Tied(array![[1.2, 0.2, 0.0], [0.2, 1.0, 0.1], [0.0, 0.1, 0.9]])
            }
            CovarianceType::Full => {
                let m = array![[1.2, 0.2, 0.0], [0.2, 1.0, 0.1], [0.0, 0.1, 0.9]];
                CovarStore::Full(ndarray::stack![Axis(0), m, m, m])
            }
        };
        let gen = GaussianHmm::new(3)
            .covariance_type(ct)
            .start_prob(array![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
            .trans_mat(array![[0.7, 0.2, 0.1], [0.1, 0.7, 0.2], [0.2, 0.1, 0.7]])
            .means(means)
            .covars(covars)
            .into_fitted()
            .unwrap();
        gen.sample(n, Some(seed), None).0
    }

    #[test]
    fn random_fit_increases_log_likelihood() {
        for ct in TYPES {
            for imp in IMPLS {
                let x = sample_data(ct, 500, 7);
                let lengths = [100usize; 5];
                let model = VariationalGaussianHmm::new(3)
                    .covariance_type(ct)
                    .implementation(imp)
                    .random_state(1)
                    .n_iter(50)
                    .tol(1e-9)
                    .build();
                assert_ll_increasing(model, x.view(), Some(&lengths), 10);
            }
        }
    }

    #[test]
    fn log_and_scaling_agree() {
        for ct in TYPES {
            let x = sample_data(ct, 400, 13);
            let lengths = [80usize; 5];
            let score = |imp| {
                VariationalGaussianHmm::new(3)
                    .covariance_type(ct)
                    .implementation(imp)
                    .random_state(5)
                    .n_iter(40)
                    .tol(1e-9)
                    .fit(x.view(), Some(&lengths))
                    .unwrap()
                    .score(x.view(), Some(&lengths))
                    .unwrap()
            };
            let log = score(Implementation::Log);
            let scaling = score(Implementation::Scaling);
            assert!(
                (log - scaling).abs() < 1e-6,
                "{ct:?}: log {log} vs scaling {scaling}"
            );
        }
    }

    /// Port of `compare_variational_and_em_models`: an EM model whose parameters
    /// are the variational posterior point estimates must score, decode, and
    /// sample identically to the variational model.
    #[test]
    fn matches_em_model_at_posterior() {
        use crate::core::params::DecoderAlgorithm::{Map, Viterbi};
        for ct in TYPES {
            for imp in IMPLS {
                let x = sample_data(ct, 300, 21);
                let lengths = [60usize; 5];
                let vi = VariationalGaussianHmm::new(3)
                    .covariance_type(ct)
                    .implementation(imp)
                    .random_state(3)
                    .n_iter(30)
                    .tol(1e-9)
                    .fit(x.view(), Some(&lengths))
                    .unwrap();

                let em = GaussianHmm::new(3)
                    .covariance_type(ct)
                    .implementation(imp)
                    .start_prob(vi.start_prob().clone())
                    .trans_mat(vi.trans_mat().clone())
                    .means(vi.emission().means_posterior().clone())
                    .covars(vi.emission().covars().clone())
                    .into_fitted()
                    .unwrap();

                let vi_score = vi.score(x.view(), Some(&lengths)).unwrap();
                let em_score = em.score(x.view(), Some(&lengths)).unwrap();
                assert!(
                    (vi_score - em_score).abs() < 1e-9,
                    "{ct:?}/{imp:?}: score {vi_score} vs {em_score}"
                );

                for algo in [Viterbi, Map] {
                    let (vi_lp, vi_path) = vi.decode(x.view(), Some(&lengths), Some(algo)).unwrap();
                    let (em_lp, em_path) = em.decode(x.view(), Some(&lengths), Some(algo)).unwrap();
                    assert!((vi_lp - em_lp).abs() < 1e-9, "{ct:?}/{imp:?}/{algo:?}");
                    assert_eq!(vi_path, em_path, "{ct:?}/{imp:?}/{algo:?}");
                }

                let (vi_obs, vi_states) = vi.sample(100, Some(42), None);
                let (em_obs, em_states) = em.sample(100, Some(42), None);
                assert_eq!(vi_states, em_states, "{ct:?}/{imp:?} sample states");
                crate::testutil::assert_close(&vi_obs, &em_obs, 0.0);
            }
        }
    }

    #[test]
    fn fit_sequences_of_different_length() {
        for ct in TYPES {
            let x = sample_data(ct, 60, 4);
            assert!(VariationalGaussianHmm::new(3)
                .covariance_type(ct)
                .random_state(1)
                .n_iter(5)
                .fit(x.view(), Some(&[15, 20, 25]))
                .is_ok());
        }
    }

    /// 500 samples from a 4-state 1-D model (McGrory & Titterington). Fitting
    /// five components with uniform variational priors lets automatic relevance
    /// determination prune the surplus component. The exact posterior mean is an
    /// init-dependent artifact (hmmlearn records three different values across
    /// its covariance subclasses), so we port the test's real check: the fitted
    /// posterior point estimate agrees with the equivalent EM model.
    #[test]
    fn mcgrory_titterington_matches_em() {
        use crate::core::params::DecoderAlgorithm::{Map, Viterbi};
        let gen = GaussianHmm::new(4)
            .covariance_type(CovarianceType::Diag)
            .start_prob(array![0.25, 0.25, 0.25, 0.25])
            .trans_mat(array![
                [0.2, 0.2, 0.3, 0.3],
                [0.3, 0.2, 0.2, 0.3],
                [0.2, 0.3, 0.3, 0.2],
                [0.3, 0.3, 0.2, 0.2]
            ])
            .means(array![[-1.5], [0.0], [1.5], [3.0]])
            .covars(CovarStore::Diag(array![[0.25], [0.25], [0.25], [0.25]]))
            .into_fitted()
            .unwrap();
        let x = gen.sample(500, Some(234234), None).0;
        let lengths = [500usize];

        // On 1-D data the four covariance types reduce to the same computation,
        // so the dense (full) and per-feature (diag) paths are representative.
        for ct in [CovarianceType::Full, CovarianceType::Diag] {
            for imp in IMPLS {
                let nc = 5;
                // vi_uniform_startprob_and_transmat: uniform Dirichlet priors,
                // posteriors scaled by the sequence count / total sample count.
                let uniform = 1.0 / nc as f64;
                let vi = VariationalGaussianHmm::new(nc)
                    .covariance_type(ct)
                    .implementation(imp)
                    .init_params("mc")
                    .random_state(234234)
                    .n_iter(60)
                    .tol(1e-9)
                    .startprob_prior(Array1::from_elem(nc, uniform))
                    .startprob_posterior(Array1::from_elem(nc, uniform * lengths.len() as f64))
                    .transmat_prior(Array2::from_elem((nc, nc), uniform))
                    .transmat_posterior(Array2::from_elem((nc, nc), uniform * 500.0))
                    .fit(x.view(), Some(&lengths))
                    .unwrap();

                let em = GaussianHmm::new(nc)
                    .covariance_type(ct)
                    .implementation(imp)
                    .start_prob(vi.start_prob().clone())
                    .trans_mat(vi.trans_mat().clone())
                    .means(vi.emission().means_posterior().clone())
                    .covars(vi.emission().covars().clone())
                    .into_fitted()
                    .unwrap();

                let vs = vi.score(x.view(), Some(&lengths)).unwrap();
                let es = em.score(x.view(), Some(&lengths)).unwrap();
                assert!((vs - es).abs() < 1e-9, "{ct:?}/{imp:?}: {vs} vs {es}");
                for algo in [Viterbi, Map] {
                    let (vlp, vpath) = vi.decode(x.view(), Some(&lengths), Some(algo)).unwrap();
                    let (elp, epath) = em.decode(x.view(), Some(&lengths), Some(algo)).unwrap();
                    assert!((vlp - elp).abs() < 1e-9, "{ct:?}/{imp:?}/{algo:?}");
                    assert_eq!(vpath, epath, "{ct:?}/{imp:?}/{algo:?}");
                }
            }
        }
    }
}
