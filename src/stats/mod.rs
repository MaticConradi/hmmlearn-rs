//! Probability densities and estimators used by the emission models.

pub mod mvn;

pub use mvn::log_multivariate_normal_density;

use ndarray::{Array2, ArrayView2, Axis};

/// Sample covariance of the features, `np.cov(X.T)` with `ddof = 1`.
///
/// Rows of `x` are observations, columns are variables. The divisor is
/// `max(n_samples - 1, 1)`, so a single observation yields an all-zero matrix
/// rather than a division by zero.
///
/// # Arguments
/// * `x` — `(n_samples, n_features)` observation matrix.
///
/// # Returns
/// The `(n_features, n_features)` unbiased covariance.
///
/// # Panics
/// If `x` has zero rows (the per-feature mean cannot be formed).
pub fn sample_covariance(x: ArrayView2<f64>) -> Array2<f64> {
    let (ns, nf) = x.dim();
    let mean = x.mean_axis(Axis(0)).unwrap();
    let denom = (ns as f64 - 1.0).max(1.0);
    Array2::from_shape_fn((nf, nf), |(i, j)| {
        let mut s = 0.0;
        for t in 0..ns {
            s += (x[[t, i]] - mean[i]) * (x[[t, j]] - mean[j]);
        }
        s / denom
    })
}
