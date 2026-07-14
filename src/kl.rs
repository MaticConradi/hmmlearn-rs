//! Kullback–Leibler divergences used by the variational models.
//!
//! Direct ports of `hmmlearn._kl_divergence`. Distributions are Dirichlet,
//! (multivariate) Normal, Gamma, and Wishart.

use crate::linalg::{inv, logdet};
use crate::special::{digamma, ln_gamma};
use ndarray::{Array2, ArrayView1, ArrayView2};
use std::f64::consts::PI;

/// Trace of a matrix — the sum of its diagonal entries `a[i][i]`.
///
/// # Panics
/// If `a` is not square (has fewer columns than rows), as diagonal indexing
/// then goes out of bounds.
fn trace(a: &Array2<f64>) -> f64 {
    (0..a.nrows()).map(|i| a[[i, i]]).sum()
}

/// KL(q ‖ p) between two Dirichlet distributions with concentration vectors.
///
/// # Arguments
/// * `q` — concentration vector of the first distribution.
/// * `p` — concentration vector of the second distribution (same length as `q`).
///
/// # Returns
/// The divergence `KL(q ‖ p)` in nats.
pub fn kl_dirichlet(q: ArrayView1<f64>, p: ArrayView1<f64>) -> f64 {
    let qsum = q.sum();
    let psum = p.sum();
    let dg_qsum = digamma(qsum);
    let term_gamma: f64 = q
        .iter()
        .zip(p.iter())
        .map(|(&qi, &pi)| ln_gamma(qi) - ln_gamma(pi))
        .sum();
    let term_e: f64 = q
        .iter()
        .zip(p.iter())
        .map(|(&qi, &pi)| (qi - pi) * (digamma(qi) - dg_qsum))
        .sum();
    ln_gamma(qsum) - ln_gamma(psum) - term_gamma + term_e
}

/// KL(q ‖ p) between two univariate Normal distributions.
///
/// # Arguments
/// * `mean_q`, `variance_q` — mean and variance of the first Normal.
/// * `mean_p`, `variance_p` — mean and variance of the second Normal.
///
/// # Returns
/// The divergence `KL(q ‖ p)` in nats.
pub fn kl_normal(mean_q: f64, variance_q: f64, mean_p: f64, variance_p: f64) -> f64 {
    (variance_p / variance_q).ln() / 2.0
        + ((mean_q - mean_p).powi(2) + variance_q) / (2.0 * variance_p)
        - 0.5
}

/// KL(q ‖ p) between two multivariate Normal distributions.
///
/// # Arguments
/// * `mean_q` — mean vector of the first Normal, length `d`.
/// * `covar_q` — `(d, d)` covariance of the first Normal.
/// * `mean_p` — mean vector of the second Normal, length `d`.
/// * `covar_p` — `(d, d)` covariance of the second Normal.
///
/// # Returns
/// The divergence `KL(q ‖ p)` in nats.
///
/// # Panics
/// If `covar_p` is singular (not invertible).
pub fn kl_multivariate_normal(
    mean_q: ArrayView1<f64>,
    covar_q: ArrayView2<f64>,
    mean_p: ArrayView1<f64>,
    covar_p: ArrayView2<f64>,
) -> f64 {
    let precision_p = inv(covar_p).expect("covar_p must be invertible");
    let mean_diff = &mean_q - &mean_p;
    let d = mean_q.len() as f64;
    0.5 * (logdet(covar_p) - logdet(covar_q)
        + trace(&precision_p.dot(&covar_q))
        + mean_diff.dot(&precision_p.dot(&mean_diff))
        - d)
}

/// KL(q ‖ p) between two Gamma distributions (shape `b`, scale `c`).
///
/// # Arguments
/// * `b_q`, `c_q` — shape and scale of the first Gamma.
/// * `b_p`, `c_p` — shape and scale of the second Gamma.
///
/// # Returns
/// The divergence `KL(q ‖ p)` in nats.
pub fn kl_gamma(b_q: f64, c_q: f64, b_p: f64, c_p: f64) -> f64 {
    (b_q - b_p) * digamma(b_q) - ln_gamma(b_q)
        + ln_gamma(b_p)
        + b_p * (c_q.ln() - c_p.ln())
        + b_q * (c_p - c_q) / c_q
}

/// KL(q ‖ p) between two Wishart distributions.
///
/// # Arguments
/// * `dof_q` — degrees of freedom of the first Wishart.
/// * `scale_q` — `(d, d)` scale matrix of the first Wishart.
/// * `dof_p` — degrees of freedom of the second Wishart.
/// * `scale_p` — `(d, d)` scale matrix of the second Wishart.
///
/// # Returns
/// The divergence `KL(q ‖ p)` in nats.
///
/// # Panics
/// If `scale_q` is singular (not invertible).
pub fn kl_wishart(
    dof_q: f64,
    scale_q: ArrayView2<f64>,
    dof_p: f64,
    scale_p: ArrayView2<f64>,
) -> f64 {
    let d = scale_p.nrows() as f64;
    (dof_q - dof_p) / 2.0 * wishart_e(dof_q, scale_q) - d * dof_q / 2.0
        + dof_q / 2.0 * trace(&scale_p.dot(&inv(scale_q).expect("scale_q invertible")))
        + wishart_logz(dof_p, scale_p)
        - wishart_logz(dof_q, scale_q)
}

/// `E[log|Γ|]` under a Wishart (the `_E` helper).
///
/// Computes `-logdet(scale / 2) + Σ_{i=0}^{d-1} digamma((dof - i) / 2)`.
///
/// # Arguments
/// * `dof` — degrees of freedom.
/// * `scale` — `(d, d)` scale matrix.
///
/// # Returns
/// The expected log-determinant `E[log|Γ|]`.
fn wishart_e(dof: f64, scale: ArrayView2<f64>) -> f64 {
    let d = scale.nrows();
    let half = scale.mapv(|x| x / 2.0);
    let dg: f64 = (0..d).map(|i| digamma((dof - i as f64) / 2.0)).sum();
    -logdet(half.view()) + dg
}

/// Log-partition function of a Wishart (the `_logZ` helper).
///
/// Computes `(d(d-1)/4)·ln π - (dof/2)·logdet(scale/2) + Σ_{i=0}^{d-1} lnΓ((dof - i)/2)`.
///
/// # Arguments
/// * `dof` — degrees of freedom.
/// * `scale` — `(d, d)` scale matrix.
///
/// # Returns
/// The log normalizing constant `log Z`.
fn wishart_logz(dof: f64, scale: ArrayView2<f64>) -> f64 {
    let d = scale.nrows();
    let half = scale.mapv(|x| x / 2.0);
    let lg: f64 = (0..d).map(|i| ln_gamma((dof - i as f64) / 2.0)).sum();
    (d as f64 * (d as f64 - 1.0) / 4.0) * PI.ln() - dof / 2.0 * logdet(half.view()) + lg
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn dirichlet_zero_and_positive() {
        let v1 = array![1.0, 2.0, 3.0, 4.0];
        let v2 = array![4.0, 3.0, 2.0, 1.0];
        assert_eq!(kl_dirichlet(v1.view(), v1.view()), 0.0);
        assert_eq!(kl_dirichlet(v2.view(), v2.view()), 0.0);
        assert!(kl_dirichlet(v1.view(), v2.view()) > 0.0);
        assert!(kl_dirichlet(v2.view(), v1.view()) > 0.0);
    }

    #[test]
    fn normal_zero_and_positive() {
        assert_eq!(kl_normal(0.0, 1.0, 0.0, 1.0), 0.0);
        assert!(kl_normal(0.0, 1.0, 1.0, 1.0) > 0.0);
    }

    #[test]
    fn multivariate_matches_univariate() {
        let mp = array![0.0];
        let vp = array![[1.0]];
        let eq = kl_multivariate_normal(mp.view(), vp.view(), mp.view(), vp.view());
        assert_eq!(eq, 0.0);
        assert_eq!(eq, kl_normal(0.0, 1.0, 0.0, 1.0));

        let mq = array![1.0];
        let vq = array![[1.0]];
        let ne = kl_multivariate_normal(mp.view(), vp.view(), mq.view(), vq.view());
        assert_eq!(ne, kl_normal(0.0, 1.0, 1.0, 1.0));
    }

    #[test]
    fn gamma_zero_and_positive() {
        assert_eq!(kl_gamma(1.0, 0.01, 1.0, 0.01), 0.0);
        assert!(kl_gamma(1.0, 0.01, 2.0, 0.01) > 0.0);
        assert!(kl_gamma(1.0, 0.01, 1.0, 0.02) > 0.0);
    }

    #[test]
    fn wishart_zero_and_positive() {
        let scale1 = array![[339.8474024737109]];
        let scale2 = array![[0.001]];
        let eq = kl_wishart(952.0, scale1.view(), 952.0, scale1.view());
        assert!(eq.abs() < 1e-9, "wishart KL(x,x) = {eq}");
        let ne = kl_wishart(952.0, scale1.view(), 1.0, scale2.view());
        assert!(ne > 0.0);
    }
}
