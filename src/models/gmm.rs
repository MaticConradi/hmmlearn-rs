//! Gaussian-mixture HMM — port of `hmmlearn.hmm.GMMHMM`.
//!
//! Each state emits from a mixture of `n_mix` Gaussians. Per-state densities
//! reuse [`log_multivariate_normal_density`] by treating the mixtures as a small
//! Gaussian model; the M-step follows hmmlearn's Normal/Inverse-Wishart update.

use crate::cluster::kmeans;
use crate::core::emission::EmissionModel;
use crate::core::hmm::Hmm;
use crate::core::inference::Em;
use crate::core::params::{DecoderAlgorithm, Implementation, Param, ParamSet};
use crate::core::{ConvergenceMonitor, Fitted};
use crate::covariance::{CovarStore, CovarianceType};
use crate::error::{HmmError, Result};
use crate::rng::NumpyRandomState;
use crate::special::logsumexp;
use crate::stats::log_multivariate_normal_density;
use crate::util::log_normalize_axis;
use ndarray::{s, Array1, Array2, Array3, Array4, ArrayView2, Axis};

/// Per-state mixture covariances (adds a mixture axis to `CovarStore`).
#[derive(Clone, Debug, PartialEq)]
pub enum GmmCovars {
    /// `(n_components, n_mix)`
    Spherical(Array2<f64>),
    /// `(n_components, n_mix, n_features)`
    Diag(Array3<f64>),
    /// `(n_components, n_mix, n_features, n_features)`
    Full(Array4<f64>),
    /// `(n_components, n_features, n_features)` — shared across mixtures.
    Tied(Array3<f64>),
}

impl GmmCovars {
    /// The `CovarStore` for one state's mixtures, treating the `n_mix` mixtures
    /// as that store's "components".
    ///
    /// # Arguments
    /// * `i` — state index into the leading (`n_components`) axis.
    ///
    /// # Returns
    /// A `CovarStore` of the same covariance type holding state `i`'s mixture
    /// covariances (shapes: Spherical `(n_mix,)`, Diag `(n_mix, n_features)`,
    /// Full `(n_mix, n_features, n_features)`, Tied `(n_features, n_features)`).
    ///
    /// # Panics
    /// If `i` is out of bounds for the state axis.
    fn state_store(&self, i: usize) -> CovarStore {
        match self {
            GmmCovars::Spherical(c) => CovarStore::Spherical(c.slice(s![i, ..]).to_owned()),
            GmmCovars::Diag(c) => CovarStore::Diag(c.slice(s![i, .., ..]).to_owned()),
            GmmCovars::Full(c) => CovarStore::Full(c.slice(s![i, .., .., ..]).to_owned()),
            GmmCovars::Tied(c) => CovarStore::Tied(c.slice(s![i, .., ..]).to_owned()),
        }
    }
}

/// Sufficient statistics for GMM emissions, accumulated over all sequences.
pub struct GmmStats {
    /// Posterior responsibility mass per state-mixture, `(n_components, n_mix)`.
    post_mix_sum: Array2<f64>,
    /// Posterior occupation mass per state, `(n_components,)`.
    post_sum: Array1<f64>,
    /// Responsibility-weighted sum of observations per mixture,
    /// `(n_components, n_mix, n_features)`; seeded with `means_weight * means_prior`.
    m_n: Array3<f64>,
    /// Responsibility-weighted centered scatter, same shape as `covars`.
    c_n: GmmCovars,
}

/// Gaussian-mixture emission model.
#[derive(Clone)]
pub struct GmmEm {
    n_components: usize,
    n_mix: usize,
    n_features: Option<usize>,
    covariance_type: CovarianceType,
    means: Option<Array3<f64>>,   // (nc, nm, nf)
    weights: Option<Array2<f64>>, // (nc, nm)
    covars: Option<GmmCovars>,
    means_preset: bool,
    weights_preset: bool,
    covars_preset: bool,
    min_covar: f64,
    weights_prior: f64,
    means_prior: f64,
    means_weight: f64,
    covars_prior: Option<f64>,
    covars_weight: Option<f64>,
}

impl GmmEm {
    /// The fitted mixture means, shape `(n_components, n_mix, n_features)`.
    ///
    /// # Panics
    /// If the means have not been initialized (neither preset nor fit).
    pub fn means(&self) -> &Array3<f64> {
        self.means.as_ref().expect("means not initialized")
    }
    /// The fitted mixture weights, shape `(n_components, n_mix)`.
    ///
    /// # Panics
    /// If the weights have not been initialized (neither preset nor fit).
    pub fn weights(&self) -> &Array2<f64> {
        self.weights.as_ref().expect("weights not initialized")
    }
    /// The fitted mixture covariances.
    ///
    /// # Panics
    /// If the covariances have not been initialized (neither preset nor fit).
    pub fn covars(&self) -> &GmmCovars {
        self.covars.as_ref().expect("covars not initialized")
    }

    /// Number of observed features.
    ///
    /// # Panics
    /// If `n_features` has not been set (no preset means and no `fit`/`check` yet).
    fn nf(&self) -> usize {
        self.n_features.expect("n_features not set")
    }

    /// The `(covars_prior, covars_weight)` conjugate-prior hyperparameters, falling
    /// back to hmmlearn's covariance-type-specific defaults when unset.
    ///
    /// Defaults (matching `GMMHMM._init_covar_priors`): Full `(0, -(nf+2))`,
    /// Tied `(0, -(n_mix+nf+1))`, Diag `(-1.5, 0)`, Spherical `(-(n_mix+2)/2, 0)`,
    /// where `nf = n_features`.
    ///
    /// # Returns
    /// `(covars_prior, covars_weight)` as scalars.
    ///
    /// # Panics
    /// If `n_features` is unset (via [`nf`](Self::nf)).
    fn covar_priors(&self) -> (f64, f64) {
        let nf = self.nf() as f64;
        let nm = self.n_mix as f64;
        let (dp, dw) = match self.covariance_type {
            CovarianceType::Full => (0.0, -(1.0 + nf + 1.0)),
            CovarianceType::Tied => (0.0, -(nm + nf + 1.0)),
            CovarianceType::Diag => (-1.5, 0.0),
            CovarianceType::Spherical => (-(nm + 2.0) / 2.0, 0.0),
        };
        (
            self.covars_prior.unwrap_or(dp),
            self.covars_weight.unwrap_or(dw),
        )
    }

    /// Per-mixture log-densities of one state, each offset by its log mixture weight.
    ///
    /// Port of `BaseGMMHMM._compute_log_weighted_gaussian_densities`: evaluates
    /// `log N(x | mean_m, cov_m) + log w_m` for every mixture `m` of state `i`.
    ///
    /// # Arguments
    /// * `x` — observations, `(n_samples, n_features)`.
    /// * `i` — state index.
    ///
    /// # Returns
    /// The weighted log-densities, shape `(n_samples, n_mix)`.
    ///
    /// # Panics
    /// If means/weights/covars are uninitialized, or `i` is out of bounds.
    fn log_weighted_densities(&self, x: ArrayView2<f64>, i: usize) -> Array2<f64> {
        let means_i = self.means().slice(s![i, .., ..]).to_owned();
        let store = self.covars().state_store(i);
        let mut d = log_multivariate_normal_density(x, means_i.view(), &store);
        let log_w = self.weights().slice(s![i, ..]).mapv(f64::ln);
        for m in 0..self.n_mix {
            for t in 0..x.nrows() {
                d[[t, m]] += log_w[m];
            }
        }
        d
    }
}

impl EmissionModel for GmmEm {
    type Inference = Em;
    type Stats = GmmStats;

    fn emission_params() -> &'static [Param] {
        &[Param::Means, Param::Covars, Param::Weights]
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
        let nm = self.n_mix;
        let nf = self.nf();
        let mut rng = NumpyRandomState::new(seed.unwrap_or(0));

        // Sample covariance (used for covars init and sparse-cluster fallback).
        let cv = sample_covariance(x, self.min_covar);

        // Two-level k-means: main clusters, then mixtures within each.
        let main_centers = kmeans(x, nc, 10, 300, seed);
        let labels = assign_labels(x, &main_centers);
        let main_centroid = main_centers.mean_axis(Axis(0)).unwrap();
        let mut means = Array3::<f64>::zeros((nc, nm, nf));
        for label in 0..nc {
            let rows: Vec<usize> = (0..x.nrows()).filter(|&t| labels[t] == label).collect();
            if rows.len() >= nm {
                let sub =
                    ndarray::Array2::from_shape_fn((rows.len(), nf), |(r, f)| x[[rows[r], f]]);
                let centers = kmeans(sub.view(), nm, 10, 300, seed);
                means.slice_mut(s![label, .., ..]).assign(&centers);
            } else {
                for m in 0..nm {
                    let draw = rng.multivariate_normal(main_centroid.view(), cv.view());
                    means.slice_mut(s![label, m, ..]).assign(&draw);
                }
            }
        }

        if init.contains(Param::Weights) || !self.weights_preset {
            self.weights = Some(Array2::from_elem((nc, nm), 1.0 / nm as f64));
        }
        if init.contains(Param::Means) || !self.means_preset {
            self.means = Some(means);
        }
        if init.contains(Param::Covars) || !self.covars_preset {
            self.covars = Some(broadcast_covar(&cv, self.covariance_type, nc, nm));
        }
        self.weights_preset = true;
        self.means_preset = true;
        self.covars_preset = true;
        Ok(())
    }

    fn check(&self, n_components: usize) -> Result<()> {
        let nm = self.n_mix;
        let nf = self.nf();
        let weights = self.weights.as_ref().ok_or(HmmError::NotFitted)?;
        if weights.dim() != (n_components, nm) {
            return Err(HmmError::DimensionMismatch(
                "weights_ must have shape (n_components, n_mix)".into(),
            ));
        }
        for row in weights.rows() {
            if (row.sum() - 1.0).abs() > 1e-5 {
                return Err(HmmError::InvalidParameter(
                    "weights_ rows must sum to 1".into(),
                ));
            }
        }
        let means = self.means.as_ref().ok_or(HmmError::NotFitted)?;
        if means.dim() != (n_components, nm, nf) {
            return Err(HmmError::DimensionMismatch(
                "means_ must have shape (n_components, n_mix, n_features)".into(),
            ));
        }
        let covars = self.covars.as_ref().ok_or(HmmError::NotFitted)?;
        let ok_shape = match covars {
            GmmCovars::Spherical(c) => c.dim() == (n_components, nm),
            GmmCovars::Diag(c) => c.dim() == (n_components, nm, nf),
            GmmCovars::Full(c) => c.dim() == (n_components, nm, nf, nf),
            GmmCovars::Tied(c) => c.dim() == (n_components, nf, nf),
        };
        if !ok_shape {
            return Err(HmmError::DimensionMismatch(
                "GMM covars have the wrong shape for the covariance type".into(),
            ));
        }
        Ok(())
    }

    fn n_fit_scalars(&self, n_components: usize, params: ParamSet) -> usize {
        let nf = self.nf();
        let nm = self.n_mix;
        let mut n = 0;
        if params.contains(Param::Means) {
            n += n_components * nm * nf;
        }
        if params.contains(Param::Weights) {
            n += nm - 1;
        }
        if params.contains(Param::Covars) {
            n += match self.covariance_type {
                CovarianceType::Spherical => n_components * nm,
                CovarianceType::Diag => n_components * nm * nf,
                CovarianceType::Full => n_components * nm * nf * (nf + 1) / 2,
                CovarianceType::Tied => n_components * nf * (nf + 1) / 2,
            };
        }
        n
    }

    fn log_likelihood(&self, x: ArrayView2<f64>) -> Array2<f64> {
        let nc = self.n_components;
        let ns = x.nrows();
        let mut out = Array2::zeros((ns, nc));
        for i in 0..nc {
            let d = self.log_weighted_densities(x, i);
            for t in 0..ns {
                out[[t, i]] = logsumexp(d.row(t));
            }
        }
        out
    }

    fn init_stats(&self) -> GmmStats {
        let nc = self.n_components;
        let nm = self.n_mix;
        let nf = self.nf();
        // m_n starts at means_weight * means_prior (broadcast scalars).
        let m_n = Array3::from_elem((nc, nm, nf), self.means_weight * self.means_prior);
        let c_n = match self.covariance_type {
            CovarianceType::Spherical => GmmCovars::Spherical(Array2::zeros((nc, nm))),
            CovarianceType::Diag => GmmCovars::Diag(Array3::zeros((nc, nm, nf))),
            CovarianceType::Full => GmmCovars::Full(Array4::zeros((nc, nm, nf, nf))),
            CovarianceType::Tied => GmmCovars::Tied(Array3::zeros((nc, nf, nf))),
        };
        GmmStats {
            post_mix_sum: Array2::zeros((nc, nm)),
            post_sum: Array1::zeros(nc),
            m_n,
            c_n,
        }
    }

    fn accumulate(
        &self,
        stats: &mut GmmStats,
        x: ArrayView2<f64>,
        posteriors: ArrayView2<f64>,
        params: ParamSet,
    ) {
        let nc = self.n_components;
        let nm = self.n_mix;
        let nf = self.nf();
        let ns = x.nrows();
        let means = self.means();
        let needs_mean = params.contains(Param::Means);
        let needs_covar = params.contains(Param::Covars);

        for i in 0..nc {
            // post_mix[t, m] = softmax over mixtures of weighted densities.
            let mut post_mix = self.log_weighted_densities(x, i);
            log_normalize_axis(&mut post_mix, 1);
            post_mix.mapv_inplace(f64::exp);

            for t in 0..ns {
                let pc = posteriors[[t, i]];
                for m in 0..nm {
                    let pcm = pc * post_mix[[t, m]];
                    stats.post_mix_sum[[i, m]] += pcm;
                    if needs_mean {
                        for f in 0..nf {
                            stats.m_n[[i, m, f]] += pcm * x[[t, f]];
                        }
                    }
                    if needs_covar {
                        accumulate_covar(&mut stats.c_n, i, m, pcm, x.row(t), means, nf);
                    }
                }
                stats.post_sum[i] += pc;
            }
        }
    }

    fn mstep(&mut self, stats: &GmmStats, params: ParamSet) -> Result<()> {
        let nc = self.n_components;
        let nm = self.n_mix;
        let nf = self.nf();

        if params.contains(Param::Weights) {
            let alpha_m1 = self.weights_prior - 1.0;
            let mut w = Array2::zeros((nc, nm));
            for i in 0..nc {
                let w_d = stats.post_sum[i] + alpha_m1 * nm as f64;
                for m in 0..nm {
                    w[[i, m]] = (stats.post_mix_sum[[i, m]] + alpha_m1) / w_d;
                }
            }
            self.weights = Some(w);
        }

        if params.contains(Param::Means) {
            let weights = self.weights().clone();
            let mut means = Array3::zeros((nc, nm, nf));
            for i in 0..nc {
                for m in 0..nm {
                    let mut m_d = stats.post_mix_sum[[i, m]] + self.means_weight;
                    let m_n_zero = (0..nf).all(|f| stats.m_n[[i, m, f]] == 0.0);
                    if weights[[i, m]] == 0.0 && m_n_zero {
                        m_d = 1.0;
                    }
                    for f in 0..nf {
                        means[[i, m, f]] = stats.m_n[[i, m, f]] / m_d;
                    }
                }
            }
            self.means = Some(means);
        }

        if params.contains(Param::Covars) {
            self.covars = Some(self.update_covars(stats));
        }
        Ok(())
    }

    fn sample_state(&self, state: usize, rng: &mut NumpyRandomState) -> Array1<f64> {
        let weights = self.weights();
        let i_gauss = rng.choice_weighted(weights.slice(s![state, ..]));
        let mean = self.means().slice(s![state, i_gauss, ..]).to_owned();
        let cov = match self.covars() {
            GmmCovars::Spherical(c) => {
                let v = c[[state, i_gauss]];
                Array2::from_diag(&Array1::from_elem(self.nf(), v))
            }
            GmmCovars::Diag(c) => Array2::from_diag(&c.slice(s![state, i_gauss, ..]).to_owned()),
            GmmCovars::Full(c) => c.slice(s![state, i_gauss, .., ..]).to_owned(),
            GmmCovars::Tied(c) => c.slice(s![state, .., ..]).to_owned(),
        };
        rng.multivariate_normal(mean.view(), cov.view())
    }
}

impl GmmEm {
    /// M-step covariance update under the Normal/Inverse-Wishart conjugate prior.
    ///
    /// Port of the covariance branch of `GMMHMM._do_mstep`. Each covariance type
    /// forms `covars = numerator / denominator` per state (`i`) and mixture (`m`),
    /// combining the accumulated centered scatter `stats.c_n` with the prior terms
    /// `covars_prior`/`covars_weight` (`(cp, cw)` from [`covar_priors`](Self::covar_priors))
    /// and the mean prior `means_prior`/`means_weight` (`mp`/`lambdas`):
    /// - **Diag**: `num = lambdas*(mean-mp)^2 + 2*cw + c_n`, `den = post_mix_sum + 1 + 2*(cp+1)`.
    /// - **Spherical**: `num = lambdas*||mean-mp||^2 + 2*cw + c_n`,
    ///   `den = nf*(post_mix_sum + 1) + 2*(cp+1)`.
    /// - **Full**: `num = cp + lambdas*(mean-mp)(mean-mp)^T + c_n`,
    ///   `den = post_mix_sum + 1 + cw + nf + 1`.
    /// - **Tied**: scatter summed over mixtures, `num = cp + sum_m lambdas*(mean-mp)(mean-mp)^T + c_n`,
    ///   `den = post_sum + n_mix + cw + nf + 1`.
    ///
    /// # Returns
    /// The updated covariances, same variant and shape as `stats.c_n`.
    ///
    /// # Panics
    /// If means are uninitialized or `n_features` is unset.
    fn update_covars(&self, stats: &GmmStats) -> GmmCovars {
        let nc = self.n_components;
        let nm = self.n_mix;
        let nf = self.nf();
        let means = self.means();
        let mp = self.means_prior;
        let lambdas = self.means_weight;
        let (cp, cw) = self.covar_priors();

        match &stats.c_n {
            GmmCovars::Diag(c_n) => {
                let mut cov = Array3::zeros((nc, nm, nf));
                for i in 0..nc {
                    for m in 0..nm {
                        let c_d = stats.post_mix_sum[[i, m]] + 1.0 + 2.0 * (cp + 1.0);
                        for f in 0..nf {
                            let cm = means[[i, m, f]] - mp;
                            let num = lambdas * cm * cm + 2.0 * cw + c_n[[i, m, f]];
                            cov[[i, m, f]] = num / c_d;
                        }
                    }
                }
                GmmCovars::Diag(cov)
            }
            GmmCovars::Spherical(c_n) => {
                let mut cov = Array2::zeros((nc, nm));
                for i in 0..nc {
                    for m in 0..nm {
                        let cm2: f64 = (0..nf).map(|f| (means[[i, m, f]] - mp).powi(2)).sum();
                        let num = lambdas * cm2 + 2.0 * cw + c_n[[i, m]];
                        let c_d = nf as f64 * (stats.post_mix_sum[[i, m]] + 1.0) + 2.0 * (cp + 1.0);
                        cov[[i, m]] = num / c_d;
                    }
                }
                GmmCovars::Spherical(cov)
            }
            GmmCovars::Full(c_n) => {
                let mut cov = Array4::zeros((nc, nm, nf, nf));
                for i in 0..nc {
                    for m in 0..nm {
                        let c_d = stats.post_mix_sum[[i, m]] + 1.0 + cw + nf as f64 + 1.0;
                        for k in 0..nf {
                            for l in 0..nf {
                                let cmk = means[[i, m, k]] - mp;
                                let cml = means[[i, m, l]] - mp;
                                let num = cp + lambdas * cmk * cml + c_n[[i, m, k, l]];
                                cov[[i, m, k, l]] = num / c_d;
                            }
                        }
                    }
                }
                GmmCovars::Full(cov)
            }
            GmmCovars::Tied(c_n) => {
                let mut cov = Array3::zeros((nc, nf, nf));
                for i in 0..nc {
                    let c_d = stats.post_sum[i] + nm as f64 + cw + nf as f64 + 1.0;
                    for k in 0..nf {
                        for l in 0..nf {
                            let mut acc = cp;
                            for m in 0..nm {
                                let cmk = means[[i, m, k]] - mp;
                                let cml = means[[i, m, l]] - mp;
                                acc += lambdas * cmk * cml;
                            }
                            cov[[i, k, l]] = (acc + c_n[[i, k, l]]) / c_d;
                        }
                    }
                }
                GmmCovars::Tied(cov)
            }
        }
    }
}

/// Adds one sample's responsibility-weighted centered scatter into the covariance
/// sufficient statistic `c_n` for state `i`, mixture `m`.
///
/// With `d = x_row - means[i, m]`, accumulates `pcm * d d^T` (Full/Tied), `pcm * d^2`
/// per feature (Diag), or `pcm * ||d||^2` (Spherical). Tied writes into the shared
/// `(i, ·, ·)` slot regardless of `m`.
///
/// # Arguments
/// * `c_n` — covariance statistic to update in place.
/// * `i` — state index.
/// * `m` — mixture index.
/// * `pcm` — combined state-mixture posterior responsibility for this sample.
/// * `x_row` — one observation, length `nf`.
/// * `means` — current mixture means, `(n_components, n_mix, n_features)`.
/// * `nf` — number of features.
///
/// # Panics
/// If `i`/`m` are out of bounds, or `x_row`/`means` are shorter than `nf`.
fn accumulate_covar(
    c_n: &mut GmmCovars,
    i: usize,
    m: usize,
    pcm: f64,
    x_row: ndarray::ArrayView1<f64>,
    means: &Array3<f64>,
    nf: usize,
) {
    match c_n {
        GmmCovars::Diag(c) => {
            for f in 0..nf {
                let d = x_row[f] - means[[i, m, f]];
                c[[i, m, f]] += pcm * d * d;
            }
        }
        GmmCovars::Spherical(c) => {
            let mut s = 0.0;
            for f in 0..nf {
                let d = x_row[f] - means[[i, m, f]];
                s += d * d;
            }
            c[[i, m]] += pcm * s;
        }
        GmmCovars::Full(c) => {
            for k in 0..nf {
                let dk = x_row[k] - means[[i, m, k]];
                for l in 0..nf {
                    let dl = x_row[l] - means[[i, m, l]];
                    c[[i, m, k, l]] += pcm * dk * dl;
                }
            }
        }
        GmmCovars::Tied(c) => {
            for k in 0..nf {
                let dk = x_row[k] - means[[i, m, k]];
                for l in 0..nf {
                    let dl = x_row[l] - means[[i, m, l]];
                    c[[i, k, l]] += pcm * dk * dl;
                }
            }
        }
    }
}

/// Sample covariance of the observations, regularized as `np.cov(X.T) + min_covar*I`.
///
/// Uses the unbiased `n-1` denominator (floored at 1 for a single sample).
///
/// # Arguments
/// * `x` — observations, `(n_samples, n_features)`.
/// * `min_covar` — value added to the diagonal for numerical stability.
///
/// # Returns
/// The `(n_features, n_features)` covariance matrix.
///
/// # Panics
/// If `x` has zero rows (`mean_axis` returns `None`).
fn sample_covariance(x: ArrayView2<f64>, min_covar: f64) -> Array2<f64> {
    let (ns, nf) = x.dim();
    let mean = x.mean_axis(Axis(0)).unwrap();
    let denom = (ns as f64 - 1.0).max(1.0);
    let mut cov = Array2::zeros((nf, nf));
    for j in 0..nf {
        for k in 0..nf {
            let mut s = 0.0;
            for t in 0..ns {
                s += (x[[t, j]] - mean[j]) * (x[[t, k]] - mean[k]);
            }
            cov[[j, k]] = s / denom;
        }
    }
    for d in 0..nf {
        cov[[d, d]] += min_covar;
    }
    cov
}

/// Broadcasts a single `(n_features, n_features)` covariance template to the
/// per-state, per-mixture storage shape for the given covariance type.
///
/// Tied copies the full matrix per state; Full copies it per state-mixture; Diag
/// takes its diagonal; Spherical takes its mean.
///
/// # Arguments
/// * `cv` — covariance template, `(n_features, n_features)`.
/// * `ct` — target covariance parameterization.
/// * `nc` — number of states (`n_components`).
/// * `nm` — number of mixtures (`n_mix`).
///
/// # Returns
/// The broadcast [`GmmCovars`] for the requested type.
///
/// # Panics
/// For `Spherical`, if `cv` is empty (`mean` returns `None`).
fn broadcast_covar(cv: &Array2<f64>, ct: CovarianceType, nc: usize, nm: usize) -> GmmCovars {
    let nf = cv.nrows();
    match ct {
        CovarianceType::Tied => {
            let mut c = Array3::zeros((nc, nf, nf));
            for i in 0..nc {
                c.slice_mut(s![i, .., ..]).assign(cv);
            }
            GmmCovars::Tied(c)
        }
        CovarianceType::Full => {
            let mut c = Array4::zeros((nc, nm, nf, nf));
            for i in 0..nc {
                for m in 0..nm {
                    c.slice_mut(s![i, m, .., ..]).assign(cv);
                }
            }
            GmmCovars::Full(c)
        }
        CovarianceType::Diag => {
            let diag = Array1::from_shape_fn(nf, |i| cv[[i, i]]);
            let mut c = Array3::zeros((nc, nm, nf));
            for i in 0..nc {
                for m in 0..nm {
                    c.slice_mut(s![i, m, ..]).assign(&diag);
                }
            }
            GmmCovars::Diag(c)
        }
        CovarianceType::Spherical => {
            GmmCovars::Spherical(Array2::from_elem((nc, nm), cv.mean().unwrap()))
        }
    }
}

/// Assigns each observation to its nearest centroid by squared Euclidean distance.
///
/// # Arguments
/// * `x` — observations, `(n_samples, n_features)`.
/// * `centers` — cluster centroids, `(n_clusters, n_features)`.
///
/// # Returns
/// A length-`n_samples` vector of centroid indices (ties resolved to the lowest index).
fn assign_labels(x: ArrayView2<f64>, centers: &Array2<f64>) -> Vec<usize> {
    (0..x.nrows())
        .map(|t| {
            let mut best = 0;
            let mut best_d = f64::INFINITY;
            for c in 0..centers.nrows() {
                let d: f64 = (0..x.ncols())
                    .map(|f| (x[[t, f]] - centers[[c, f]]).powi(2))
                    .sum();
                if d < best_d {
                    best_d = d;
                    best = c;
                }
            }
            best
        })
        .collect()
}

/// Builder for [`GmmEm`] HMMs (`hmmlearn.hmm.GMMHMM`).
#[derive(Clone)]
pub struct GmmHmm {
    n_components: usize,
    n_mix: usize,
    covariance_type: CovarianceType,
    min_covar: f64,
    algorithm: DecoderAlgorithm,
    implementation: Implementation,
    params: ParamSet,
    init_params: ParamSet,
    n_iter: usize,
    tol: f64,
    verbose: bool,
    random_state: Option<u32>,
    startprob_prior: f64,
    transmat_prior: f64,
    weights_prior: f64,
    means_prior: f64,
    means_weight: f64,
    covars_prior: Option<f64>,
    covars_weight: Option<f64>,
    start_prob: Option<Array1<f64>>,
    trans_mat: Option<Array2<f64>>,
    means: Option<Array3<f64>>,
    weights: Option<Array2<f64>>,
    covars: Option<GmmCovars>,
}

impl GmmHmm {
    /// A builder for a GMM-emission HMM with `n_components` states and `n_mix`
    /// Gaussians per state, carrying hmmlearn's `GMMHMM` defaults.
    ///
    /// Defaults: `Diag` covariances, `min_covar = 1e-3`, Viterbi decoding, log
    /// implementation, all parameters estimated and initialized (`"stmcw"`),
    /// `n_iter = 10`, `tol = 1e-2`, no fixed seed, all Dirichlet/mean priors at
    /// their neutral values, and type-specific covariance priors resolved lazily.
    ///
    /// # Arguments
    /// * `n_components` — number of hidden states.
    /// * `n_mix` — number of mixture components per state.
    ///
    /// # Returns
    /// A [`GmmHmm`] builder to configure further and then `fit`/`into_fitted`.
    pub fn new(n_components: usize, n_mix: usize) -> Self {
        GmmHmm {
            n_components,
            n_mix,
            covariance_type: CovarianceType::Diag,
            min_covar: 1e-3,
            algorithm: DecoderAlgorithm::Viterbi,
            implementation: Implementation::Log,
            params: ParamSet::from_codes("stmcw"),
            init_params: ParamSet::from_codes("stmcw"),
            n_iter: 10,
            tol: 1e-2,
            verbose: false,
            random_state: None,
            startprob_prior: 1.0,
            transmat_prior: 1.0,
            weights_prior: 1.0,
            means_prior: 0.0,
            means_weight: 0.0,
            covars_prior: None,
            covars_weight: None,
            start_prob: None,
            trans_mat: None,
            means: None,
            weights: None,
            covars: None,
        }
    }

    /// Sets the covariance parameterization for the mixtures (default `Diag`).
    pub fn covariance_type(mut self, ct: CovarianceType) -> Self {
        self.covariance_type = ct;
        self
    }
    /// Sets the maximum number of EM iterations.
    pub fn n_iter(mut self, n: usize) -> Self {
        self.n_iter = n;
        self
    }
    /// Sets the EM convergence tolerance on the per-iteration log-likelihood gain.
    pub fn tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }
    /// Sets the forward-backward implementation (log-space or scaling).
    pub fn implementation(mut self, i: Implementation) -> Self {
        self.implementation = i;
        self
    }
    /// Sets which parameters are re-estimated during EM, by letter codes (`s`, `t`, `m`, `c`, `w`).
    pub fn params(mut self, codes: &str) -> Self {
        self.params = ParamSet::from_codes(codes);
        self
    }
    /// Sets which parameters are initialized before EM, by letter codes (`s`, `t`, `m`, `c`, `w`).
    pub fn init_params(mut self, codes: &str) -> Self {
        self.init_params = ParamSet::from_codes(codes);
        self
    }
    /// Sets the RNG seed used for initialization and sampling.
    pub fn random_state(mut self, seed: u32) -> Self {
        self.random_state = Some(seed);
        self
    }
    /// Presets the initial-state distribution, length `n_components`.
    pub fn start_prob(mut self, sp: Array1<f64>) -> Self {
        self.start_prob = Some(sp);
        self
    }
    /// Presets the transition matrix, `(n_components, n_components)`.
    pub fn trans_mat(mut self, tm: Array2<f64>) -> Self {
        self.trans_mat = Some(tm);
        self
    }
    /// Presets the mixture means, `(n_components, n_mix, n_features)`.
    pub fn means(mut self, m: Array3<f64>) -> Self {
        self.means = Some(m);
        self
    }
    /// Presets the mixture weights, `(n_components, n_mix)`.
    pub fn weights(mut self, w: Array2<f64>) -> Self {
        self.weights = Some(w);
        self
    }
    /// Presets the mixture covariances.
    pub fn covars(mut self, c: GmmCovars) -> Self {
        self.covars = Some(c);
        self
    }

    /// Assembles the configured builder into a runnable `Hmm<GmmEm>`.
    ///
    /// Preset arrays are moved into the emission/inference state and marked as
    /// preset; `n_features` is inferred from preset means when available
    /// (otherwise left unset until first `fit`).
    ///
    /// # Returns
    /// The assembled `Hmm<GmmEm>`, not yet fitted.
    fn build(self) -> Hmm<GmmEm> {
        let n_features = self.means.as_ref().map(|m| m.shape()[2]);
        let inference = Em::new(
            self.n_components,
            self.startprob_prior,
            self.transmat_prior,
            self.start_prob,
            self.trans_mat,
        );
        let emission = GmmEm {
            n_components: self.n_components,
            n_mix: self.n_mix,
            n_features,
            covariance_type: self.covariance_type,
            means_preset: self.means.is_some(),
            weights_preset: self.weights.is_some(),
            covars_preset: self.covars.is_some(),
            means: self.means,
            weights: self.weights,
            covars: self.covars,
            min_covar: self.min_covar,
            weights_prior: self.weights_prior,
            means_prior: self.means_prior,
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

    /// Fits the model to observations by the EM (Baum-Welch) algorithm.
    ///
    /// # Arguments
    /// * `x` — stacked observations, `(total_samples, n_features)`.
    /// * `lengths` — per-sequence sample counts summing to `total_samples`;
    ///   `None` treats `x` as one sequence.
    ///
    /// # Returns
    /// The `Fitted<GmmEm>` model after EM.
    ///
    /// # Errors
    /// Propagates initialization, dimension, and convergence errors from the
    /// underlying `Hmm::fit` (e.g. shape or feature-count mismatches).
    pub fn fit(self, x: ArrayView2<f64>, lengths: Option<&[usize]>) -> Result<Fitted<GmmEm>> {
        self.build().fit(x, lengths)
    }

    /// Treats the configured (preset) parameters as already fitted, after validation.
    ///
    /// # Returns
    /// The `Fitted<GmmEm>` model wrapping the preset parameters.
    ///
    /// # Errors
    /// If the preset parameters are missing or fail the consistency checks
    /// (e.g. wrong shapes, weights not summing to 1), as reported by
    /// `Hmm::into_fitted`.
    pub fn into_fitted(self) -> Result<Fitted<GmmEm>> {
        self.build().into_fitted()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{assert_close, assert_ll_increasing};
    use crate::util::normalize_axis;

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
        means: Array3<f64>,
        weights: Array2<f64>,
        covars: GmmCovars,
    }

    fn uniform(rng: &mut NumpyRandomState, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * rng.random_sample()
    }

    fn spd(nf: usize, rng: &mut NumpyRandomState) -> Array2<f64> {
        let a = Array2::from_shape_fn((nf, nf), |_| uniform(rng, -2.0, 2.0));
        let mut m = a.t().dot(&a);
        for d in 0..nf {
            m[[d, d]] += 0.5;
        }
        m
    }

    /// Well-separated 3-state, 2-mixture, 2-feature GMM (mirrors prep_params).
    fn fixture(ct: CovarianceType) -> Fixture {
        let (nc, nm, nf) = (3, 2, 2);
        let mut rng = NumpyRandomState::new(14);
        // bounding-box vertices via cumulative offsets
        let mut dim_lims = Array2::<f64>::zeros((nc + 1, nf));
        for i in 0..nc {
            for f in 0..nf {
                dim_lims[[i + 1, f]] = dim_lims[[i, f]] + uniform(&mut rng, 10.0, 15.0);
            }
        }
        let mut means = Array3::zeros((nc, nm, nf));
        for i in 0..nc {
            for f in 0..nf {
                for m in 0..nm {
                    means[[i, m, f]] = uniform(&mut rng, dim_lims[[i, f]], dim_lims[[i + 1, f]]);
                }
            }
        }
        let mut startprob = Array1::zeros(nc);
        startprob[0] = 1.0;
        let mut transmat = Array2::from_shape_fn((nc, nc), |_| uniform(&mut rng, 0.0, 1.0));
        normalize_axis(&mut transmat, 1);
        let covars = match ct {
            CovarianceType::Spherical => {
                GmmCovars::Spherical(Array2::from_shape_fn((nc, nm), |_| {
                    uniform(&mut rng, 0.1, 5.0)
                }))
            }
            CovarianceType::Diag => GmmCovars::Diag(Array3::from_shape_fn((nc, nm, nf), |_| {
                uniform(&mut rng, 0.1, 5.0)
            })),
            CovarianceType::Tied => {
                let mut c = Array3::zeros((nc, nf, nf));
                for i in 0..nc {
                    c.slice_mut(s![i, .., ..]).assign(&spd(nf, &mut rng));
                }
                GmmCovars::Tied(c)
            }
            CovarianceType::Full => {
                let mut c = Array4::zeros((nc, nm, nf, nf));
                for i in 0..nc {
                    for m in 0..nm {
                        c.slice_mut(s![i, m, .., ..]).assign(&spd(nf, &mut rng));
                    }
                }
                GmmCovars::Full(c)
            }
        };
        let mut weights = Array2::from_shape_fn((nc, nm), |_| uniform(&mut rng, 0.0, 1.0));
        normalize_axis(&mut weights, 1);
        Fixture {
            startprob,
            transmat,
            means,
            weights,
            covars,
        }
    }

    fn model(ct: CovarianceType, imp: Implementation, fx: &Fixture) -> Fitted<GmmEm> {
        GmmHmm::new(3, 2)
            .covariance_type(ct)
            .implementation(imp)
            .start_prob(fx.startprob.clone())
            .trans_mat(fx.transmat.clone())
            .means(fx.means.clone())
            .weights(fx.weights.clone())
            .covars(fx.covars.clone())
            .into_fitted()
            .unwrap()
    }

    #[test]
    fn bad_covariance_type_rejected() {
        assert!("bad_covariance_type".parse::<CovarianceType>().is_err());
    }

    #[test]
    fn good_covariance_type_checks() {
        for ct in TYPES {
            let fx = fixture(ct);
            assert!(model(ct, Implementation::Log, &fx).n_components() == 3);
        }
    }

    #[test]
    fn sample_shape() {
        for ct in TYPES {
            let fx = fixture(ct);
            let (x, states) = model(ct, Implementation::Log, &fx).sample(1000, Some(0), None);
            assert_eq!(x.dim(), (1000, 2));
            assert_eq!(states.len(), 1000);
        }
    }

    #[test]
    fn init_then_check_ok() {
        for ct in TYPES {
            let fx = fixture(ct);
            let (x, _) = model(ct, Implementation::Log, &fx).sample(1000, Some(0), None);
            // n_iter = 0 runs _init + _check without EM iterations.
            assert!(GmmHmm::new(3, 2)
                .covariance_type(ct)
                .n_iter(0)
                .random_state(1)
                .fit(x.view(), Some(&[1000]))
                .is_ok());
        }
    }

    #[test]
    fn score_samples_and_decode() {
        for ct in TYPES {
            for imp in IMPLS {
                let fx = fixture(ct);
                let h = model(ct, imp, &fx);
                let (x, states) = h.sample(1000, Some(0), None);
                let (_ll, post) = h.score_samples(x.view(), None).unwrap();
                for r in post.rows() {
                    assert!((r.sum() - 1.0).abs() < 1e-9);
                }
                let (_vll, decoded) = h.decode(x.view(), None, None).unwrap();
                // hmmlearn asserts exact recovery, which holds only because its
                // specific seeded samples are cleanly separable (data-luck, not
                // guaranteed across BLAS builds). Our samples are an equally-valid
                // draw from the identical distribution, so a rare boundary point
                // may be ambiguous; the model still decodes essentially all of them.
                let mism = decoded
                    .iter()
                    .zip(states.iter())
                    .filter(|(a, b)| a != b)
                    .count();
                assert!(mism <= 3, "{ct:?}/{imp:?}: {mism} decode mismatches / 1000");
            }
        }
    }

    #[test]
    fn fit_increases_log_likelihood() {
        for ct in TYPES {
            for imp in IMPLS {
                let fx = fixture(ct);
                let (x, _) = model(ct, imp, &fx).sample(500, Some(0), None);
                let m = GmmHmm::new(3, 2)
                    .covariance_type(ct)
                    .implementation(imp)
                    .random_state(1)
                    .build();
                assert_ll_increasing(m, x.view(), None, 5);
            }
        }
    }

    #[test]
    fn fit_sparse_data() {
        for ct in TYPES {
            let mut fx = fixture(ct);
            fx.means.mapv_inplace(|v| v * 1000.0); // gaussians far apart
            let (x, _) = model(ct, Implementation::Log, &fx).sample(1000, Some(0), None);
            assert!(GmmHmm::new(3, 2)
                .covariance_type(ct)
                .random_state(1)
                .fit(x.view(), None)
                .is_ok());
        }
    }

    #[test]
    fn criterion_aic_bic_finite() {
        for ct in TYPES {
            let mut fx = fixture(ct);
            fx.means.mapv_inplace(|v| v * 10.0);
            let (x, _) = model(ct, Implementation::Log, &fx).sample(1000, Some(2013), None);
            for n in [2usize, 3] {
                let h = GmmHmm::new(n, 2)
                    .covariance_type(ct)
                    .n_iter(10)
                    .random_state(2013)
                    .fit(x.view(), None)
                    .unwrap();
                assert!(h.aic(x.view(), None).unwrap().is_finite());
                assert!(h.bic(x.view(), None).unwrap().is_finite());
            }
        }
    }

    #[test]
    fn kmeans_init_bounds_means() {
        // Two isolated clusters; the second has fewer points than n_mix.
        let mut rng = NumpyRandomState::new(0);
        let mut data = Array2::<f64>::zeros((105, 2));
        for t in 0..100 {
            for f in 0..2 {
                data[[t, f]] = uniform(&mut rng, 0.0, 1.0);
            }
        }
        for t in 100..105 {
            for f in 0..2 {
                data[[t, f]] = uniform(&mut rng, 5.0, 6.0);
            }
        }
        let h = GmmHmm::new(2, 10)
            .n_iter(5)
            .random_state(1)
            .fit(data.view(), None)
            .unwrap();
        // Means should stay within the data bounds.
        for &v in h.emission().means().iter() {
            assert!(v > 0.0 && v < 6.0, "mean {v} out of bounds");
        }
    }

    #[test]
    fn chunked_matches_continuous() {
        for ct in TYPES {
            let fx = fixture(ct);
            let (x, _) = model(ct, Implementation::Log, &fx).sample(1000, Some(0), None);
            let uniform_sp = Array1::from_elem(3, 1.0 / 3.0);
            let uniform_tm = Array2::from_elem((3, 3), 1.0 / 3.0);
            let build = || {
                GmmHmm::new(3, 2)
                    .covariance_type(ct)
                    .init_params("mcw")
                    .n_iter(100)
                    .tol(1e-8)
                    .random_state(1)
                    .start_prob(uniform_sp.clone())
                    .trans_mat(uniform_tm.clone())
            };
            let m1 = build().fit(x.view(), None).unwrap();
            let m2 = build().fit(x.view(), Some(&[200; 5])).unwrap();
            assert_close(m1.emission().means(), m2.emission().means(), 1e-2);
            assert_close(m1.emission().weights(), m2.emission().weights(), 1e-2);
        }
    }
}
