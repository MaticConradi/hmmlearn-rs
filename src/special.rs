//! Special functions: log-gamma, digamma, and numerically stable log-sum-exp.
//!
//! These mirror the pieces of `scipy.special` that hmmlearn relies on. `ln_gamma`
//! is delegated to the pure-Rust `libm` (a Cephes-derived implementation, the same
//! lineage scipy uses), so results match scipy to within a few ulp.

use ndarray::ArrayView1;

/// Natural log of the absolute value of the gamma function, `ln|Γ(x)|`.
///
/// Delegates to `libm::lgamma`.
///
/// # Arguments
/// * `x` — real argument to the gamma function.
///
/// # Returns
/// `ln|Γ(x)|`.
#[inline]
pub fn ln_gamma(x: f64) -> f64 {
    libm::lgamma(x)
}

/// The digamma function `ψ(x) = d/dx ln Γ(x)`, for `x > 0`.
///
/// Uses the recurrence `ψ(x) = ψ(x + 1) - 1/x` to push the argument past 6, then
/// the standard asymptotic (Bernoulli) expansion. Accurate to ~1e-12 for `x > 0`.
///
/// # Arguments
/// * `x` — strictly positive real argument.
///
/// # Returns
/// `ψ(x)`.
///
/// # Panics
/// In debug builds, if `x <= 0` (a `debug_assert`); the expansion is only valid
/// for positive arguments.
pub fn digamma(mut x: f64) -> f64 {
    debug_assert!(
        x > 0.0,
        "digamma is only implemented for positive arguments"
    );
    let mut result = 0.0;
    while x < 6.0 {
        result -= 1.0 / x;
        x += 1.0;
    }
    let r = 1.0 / x;
    let rr = r * r;
    result + x.ln()
        - 0.5 * r
        - rr * (1.0 / 12.0
            - rr * (1.0 / 120.0 - rr * (1.0 / 252.0 - rr * (1.0 / 240.0 - rr * (1.0 / 132.0)))))
}

/// `log(exp(a) + exp(b))`, computed stably, with `-inf` handled as `log(0)`.
///
/// Shifts by the larger operand before exponentiating so no term overflows.
/// Matches the `logaddexp` helper in hmmlearn's `_hmmc.cpp`.
///
/// # Arguments
/// * `a` — first log-domain value.
/// * `b` — second log-domain value.
///
/// # Returns
/// `log(exp(a) + exp(b))`; if either input is `-inf`, the other is returned
/// unchanged (so two `-inf` inputs give `-inf`).
#[inline]
pub fn logaddexp(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        return b;
    }
    if b == f64::NEG_INFINITY {
        return a;
    }
    let (max, min) = if a > b { (a, b) } else { (b, a) };
    max + (min - max).exp().ln_1p()
}

/// `log(sum(exp(v)))` over a 1-D view, using the max-shift trick.
///
/// Subtracts the maximum element before exponentiating for numerical stability.
///
/// # Arguments
/// * `v` — 1-D view of log-domain values.
///
/// # Returns
/// `log(Σ exp(v))`; `-inf` when every element is `-inf`, and `+inf` when the
/// maximum element is `+inf` (both matching `_hmmc.cpp`).
pub fn logsumexp(v: ArrayView1<f64>) -> f64 {
    let max = v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if max == f64::NEG_INFINITY {
        return f64::NEG_INFINITY;
    }
    if max == f64::INFINITY {
        return f64::INFINITY;
    }
    let sum: f64 = v.iter().map(|&x| (x - max).exp()).sum();
    max + sum.ln()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn digamma_known_values() {
        // ψ(1) = -γ (Euler–Mascheroni)
        assert_abs_diff_eq!(digamma(1.0), -0.5772156649015329, epsilon = 1e-10);
        // ψ(2) = 1 - γ
        assert_abs_diff_eq!(digamma(2.0), 0.42278433509846713, epsilon = 1e-10);
        // ψ(0.5) = -γ - 2 ln 2
        assert_abs_diff_eq!(digamma(0.5), -1.9635100260214235, epsilon = 1e-10);
        // ψ(10)
        assert_abs_diff_eq!(digamma(10.0), 2.251752589066721, epsilon = 1e-10);
    }

    #[test]
    fn ln_gamma_known_values() {
        assert_abs_diff_eq!(ln_gamma(1.0), 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(ln_gamma(2.0), 0.0, epsilon = 1e-12);
        // Γ(5) = 24
        assert_abs_diff_eq!(ln_gamma(5.0), 24.0_f64.ln(), epsilon = 1e-12);
        // Γ(0.5) = sqrt(pi)
        assert_abs_diff_eq!(
            ln_gamma(0.5),
            std::f64::consts::PI.sqrt().ln(),
            epsilon = 1e-12
        );
    }

    #[test]
    fn logaddexp_basics() {
        assert_abs_diff_eq!(logaddexp(0.0, 0.0), 2.0_f64.ln(), epsilon = 1e-15);
        assert_eq!(logaddexp(f64::NEG_INFINITY, 3.0), 3.0);
        assert_eq!(logaddexp(3.0, f64::NEG_INFINITY), 3.0);
        assert_eq!(
            logaddexp(f64::NEG_INFINITY, f64::NEG_INFINITY),
            f64::NEG_INFINITY
        );
        // symmetry and stability for large values
        assert_abs_diff_eq!(
            logaddexp(1000.0, 1000.0),
            1000.0 + 2.0_f64.ln(),
            epsilon = 1e-9
        );
    }

    #[test]
    fn logsumexp_basics() {
        let v = array![0.0, 0.0, 0.0];
        assert_abs_diff_eq!(logsumexp(v.view()), 3.0_f64.ln(), epsilon = 1e-15);
        let all_neg_inf = array![f64::NEG_INFINITY, f64::NEG_INFINITY];
        assert_eq!(logsumexp(all_neg_inf.view()), f64::NEG_INFINITY);
        // matches logaddexp for two elements
        let v2 = array![1.5, -2.3];
        assert_abs_diff_eq!(logsumexp(v2.view()), logaddexp(1.5, -2.3), epsilon = 1e-15);
    }
}
