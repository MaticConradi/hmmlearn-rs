//! Multivariate-normal log-density for the four covariance types.
//!
//! Port of `hmmlearn.stats.log_multivariate_normal_density`. Spherical reduces
//! to diagonal (broadcasting the scalar variance), and tied reduces to full
//! (broadcasting the shared matrix). The full case uses a lower Cholesky factor,
//! retrying with a `min_covar` jitter if the matrix is momentarily not
//! positive-definite (a component starved of observations).

use crate::covariance::CovarStore;
use crate::linalg::{cholesky_lower, solve_lower_triangular};
use ndarray::{Array2, Array3, ArrayView2, ArrayView3};
use std::f64::consts::PI;

/// Diagonal jitter added to a full covariance matrix when its Cholesky
/// factorization fails, mirroring `min_covar = 1e-7` in hmmlearn.
const MIN_COVAR_FULL: f64 = 1e-7;

/// Log `P(x | mean_c, covar_c)` for every sample/component, shape `(ns, nc)`.
///
/// Dispatches on the covariance type: spherical reduces to diagonal by
/// broadcasting the scalar variance across features, and tied reduces to full by
/// broadcasting the shared matrix across components.
///
/// # Arguments
/// * `x` — `(n_samples, n_features)` observation matrix.
/// * `means` — `(n_components, n_features)` component means.
/// * `covars` — covariance parameters in their compact per-type form.
///
/// # Returns
/// The `(n_samples, n_components)` matrix of Gaussian log-densities.
pub fn log_multivariate_normal_density(
    x: ArrayView2<f64>,
    means: ArrayView2<f64>,
    covars: &CovarStore,
) -> Array2<f64> {
    let (nc, nf) = means.dim();
    match covars {
        CovarStore::Diag(c) => density_diag(x, means, c.view()),
        CovarStore::Spherical(c) => {
            let cov2 = Array2::from_shape_fn((nc, nf), |(i, _)| c[i]);
            density_diag(x, means, cov2.view())
        }
        CovarStore::Full(c) => density_full(x, means, c.view()),
        CovarStore::Tied(c) => {
            let cov3 = Array3::from_shape_fn((nc, nf, nf), |(_, i, j)| c[[i, j]]);
            density_full(x, means, cov3.view())
        }
    }
}

/// Gaussian log-density for a diagonal covariance model.
///
/// Evaluates `-0.5 * (nf·log(2π) + Σ log(covar) + Σ (x - μ)² / covar)` per
/// sample/component. Each variance is floored at `f64::MIN_POSITIVE` to avoid
/// `0·log 0` in degenerate cases.
///
/// # Arguments
/// * `x` — `(n_samples, n_features)` observation matrix.
/// * `means` — `(n_components, n_features)` component means.
/// * `covars` — `(n_components, n_features)` per-feature variances.
///
/// # Returns
/// The `(n_samples, n_components)` matrix of log-densities.
fn density_diag(
    x: ArrayView2<f64>,
    means: ArrayView2<f64>,
    covars: ArrayView2<f64>,
) -> Array2<f64> {
    let (nc, nf) = means.dim();
    let ns = x.nrows();
    let tiny = f64::MIN_POSITIVE;
    let log2pi = (2.0 * PI).ln();
    let mut out = Array2::zeros((ns, nc));
    for c in 0..nc {
        let mut sum_log_cov = 0.0;
        for f in 0..nf {
            sum_log_cov += covars[[c, f]].max(tiny).ln();
        }
        for t in 0..ns {
            let mut quad = 0.0;
            for f in 0..nf {
                let cov = covars[[c, f]].max(tiny);
                let d = x[[t, f]] - means[[c, f]];
                quad += d * d / cov;
            }
            out[[t, c]] = -0.5 * (nf as f64 * log2pi + sum_log_cov + quad);
        }
    }
    out
}

/// Gaussian log-density for full covariance matrices.
///
/// For each component, factors the covariance as `L Lᵀ` and evaluates
/// `-0.5 * (nf·log(2π) + ‖L⁻¹(x - μ)‖² + log|covar|)`, where `log|covar|` is read
/// off the Cholesky diagonal. If the factorization fails, it retries after adding
/// [`MIN_COVAR_FULL`] to the diagonal; if that still fails, every log-density for
/// that component is set to `-inf`.
///
/// # Arguments
/// * `x` — `(n_samples, n_features)` observation matrix.
/// * `means` — `(n_components, n_features)` component means.
/// * `covars` — `(n_components, n_features, n_features)` covariance matrices.
///
/// # Returns
/// The `(n_samples, n_components)` matrix of log-densities.
fn density_full(
    x: ArrayView2<f64>,
    means: ArrayView2<f64>,
    covars: ArrayView3<f64>,
) -> Array2<f64> {
    let (nc, nf) = means.dim();
    let ns = x.nrows();
    let log2pi = (2.0 * PI).ln();
    let mut out = Array2::zeros((ns, nc));
    for c in 0..nc {
        let cv = covars.index_axis(ndarray::Axis(0), c);
        let chol = cholesky_lower(cv).or_else(|| {
            let mut jittered = cv.to_owned();
            for d in 0..nf {
                jittered[[d, d]] += MIN_COVAR_FULL;
            }
            cholesky_lower(jittered.view())
        });
        let chol = match chol {
            Some(l) => l,
            None => {
                // Non-positive-definite even after jitter: component impossible.
                for t in 0..ns {
                    out[[t, c]] = f64::NEG_INFINITY;
                }
                continue;
            }
        };
        let cv_log_det: f64 = 2.0 * (0..nf).map(|d| chol[[d, d]].ln()).sum::<f64>();
        // residuals (X - mu)^T, shape (nf, ns)
        let mut resid = Array2::<f64>::zeros((nf, ns));
        for t in 0..ns {
            for f in 0..nf {
                resid[[f, t]] = x[[t, f]] - means[[c, f]];
            }
        }
        let sol = solve_lower_triangular(chol.view(), resid.view());
        for t in 0..ns {
            let mut sq = 0.0;
            for f in 0..nf {
                sq += sol[[f, t]] * sol[[f, t]];
            }
            out[[t, c]] = -0.5 * (nf as f64 * log2pi + sq + cv_log_det);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::assert_close;
    use ndarray::array;

    #[test]
    fn diag_matches_scalar_gaussian() {
        // 1-D standard normal: log N(0;0,1) = -0.5*ln(2π).
        let x = array![[0.0]];
        let means = array![[0.0]];
        let covars = CovarStore::Diag(array![[1.0]]);
        let got = log_multivariate_normal_density(x.view(), means.view(), &covars);
        assert_close(&got, &array![[-0.5 * (2.0 * PI).ln()]], 1e-12);
    }

    #[test]
    fn spherical_equals_diag_with_equal_variances() {
        let x = array![[0.5, -1.0], [2.0, 0.3]];
        let means = array![[0.0, 0.0], [1.0, 1.0]];
        let sph = CovarStore::Spherical(array![2.0, 0.5]);
        let diag = CovarStore::Diag(array![[2.0, 2.0], [0.5, 0.5]]);
        let a = log_multivariate_normal_density(x.view(), means.view(), &sph);
        let b = log_multivariate_normal_density(x.view(), means.view(), &diag);
        assert_close(&a, &b, 1e-12);
    }

    #[test]
    fn full_diagonal_matches_diag() {
        let x = array![[0.5, -1.0], [2.0, 0.3]];
        let means = array![[0.0, 0.0], [1.0, 1.0]];
        let diag = CovarStore::Diag(array![[2.0, 3.0], [0.5, 1.5]]);
        let full = CovarStore::Full(ndarray::stack![
            ndarray::Axis(0),
            array![[2.0, 0.0], [0.0, 3.0]],
            array![[0.5, 0.0], [0.0, 1.5]]
        ]);
        let a = log_multivariate_normal_density(x.view(), means.view(), &diag);
        let b = log_multivariate_normal_density(x.view(), means.view(), &full);
        assert_close(&a, &b, 1e-10);
    }

    #[test]
    fn tied_equals_full_with_shared_matrix() {
        let x = array![[0.5, -1.0]];
        let means = array![[0.0, 0.0], [1.0, 1.0]];
        let shared = array![[2.0, 0.3], [0.3, 1.0]];
        let tied = CovarStore::Tied(shared.clone());
        let full = CovarStore::Full(ndarray::stack![ndarray::Axis(0), shared, shared]);
        let a = log_multivariate_normal_density(x.view(), means.view(), &tied);
        let b = log_multivariate_normal_density(x.view(), means.view(), &full);
        assert_close(&a, &b, 1e-12);
    }

    #[test]
    fn full_correlated_matches_closed_form() {
        // 2-D Gaussian at the mean: log density = -ln(2π) - 0.5 ln|Σ|.
        let x = array![[1.0, 2.0]];
        let means = array![[1.0, 2.0]];
        let sigma = array![[2.0, 0.5], [0.5, 1.0]];
        let det: f64 = 2.0 * 1.0 - 0.5 * 0.5;
        let full = CovarStore::Full(ndarray::stack![ndarray::Axis(0), sigma]);
        let got = log_multivariate_normal_density(x.view(), means.view(), &full);
        let expected = -(2.0 * PI).ln() - 0.5 * det.ln();
        assert_close(&got, &array![[expected]], 1e-12);
    }
}
