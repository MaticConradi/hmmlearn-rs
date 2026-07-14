//! Gaussian draws: `standard_normal` (polar Box–Muller with cache) and
//! `multivariate_normal`.
//!
//! Transcendental math uses `std` f64 methods so it routes through the same
//! system libm NumPy uses, matching `standard_normal` to the last bit on this
//! platform. `multivariate_normal` cannot be byte-matched in pure Rust (NumPy
//! transforms with a LAPACK SVD), so it draws the same number of normals — to
//! keep the RNG stream aligned — but transforms via Cholesky.

use super::NumpyRandomState;
use crate::linalg::cholesky_lower;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

impl NumpyRandomState {
    /// A standard normal draw, matching NumPy's `legacy_gauss`.
    ///
    /// Polar (Marsaglia) Box–Muller: rejection-samples a point in the unit disk
    /// and returns one of the two produced normals, caching the other in
    /// `self.gauss` for the next call. A cached value, if present, is returned
    /// first without touching the RNG stream.
    ///
    /// # Returns
    /// A draw from the standard normal distribution `N(0, 1)`.
    pub fn standard_normal(&mut self) -> f64 {
        if let Some(g) = self.gauss.take() {
            return g;
        }
        loop {
            let x1 = 2.0 * self.random_sample() - 1.0;
            let x2 = 2.0 * self.random_sample() - 1.0;
            let r2 = x1 * x1 + x2 * x2;
            if r2 < 1.0 && r2 != 0.0 {
                let f = (-2.0 * r2.ln() / r2).sqrt();
                self.gauss = Some(f * x1);
                return f * x2;
            }
        }
    }

    /// `n` standard normal draws (C order).
    ///
    /// # Arguments
    /// * `n` — number of samples to draw.
    ///
    /// # Returns
    /// A length-`n` array of `N(0, 1)` draws.
    pub fn standard_normal_n(&mut self, n: usize) -> Array1<f64> {
        Array1::from_iter((0..n).map(|_| self.standard_normal()))
    }

    /// `loc + scale * standard_normal()`.
    ///
    /// # Arguments
    /// * `loc` — the mean.
    /// * `scale` — the standard deviation.
    ///
    /// # Returns
    /// A draw from `N(loc, scale²)`.
    pub fn normal(&mut self, loc: f64, scale: f64) -> f64 {
        loc + scale * self.standard_normal()
    }

    /// A single multivariate-normal draw.
    ///
    /// Draws `mean.len()` standard normals (same as NumPy, so the stream stays
    /// aligned) and applies a Cholesky transform. Intentional deviation: the
    /// produced vector is a valid sample but is NOT byte-identical to NumPy's
    /// SVD-based transform. If `cov` is not positive-definite (no Cholesky
    /// factor), the identity is used in place of the factor.
    ///
    /// # Arguments
    /// * `mean` — the mean vector, length `d`.
    /// * `cov` — the `(d, d)` covariance matrix.
    ///
    /// # Returns
    /// A length-`d` sample from `N(mean, cov)`.
    pub fn multivariate_normal(
        &mut self,
        mean: ArrayView1<f64>,
        cov: ArrayView2<f64>,
    ) -> Array1<f64> {
        let n = mean.len();
        let z = self.standard_normal_n(n);
        let l = cholesky_lower(cov).unwrap_or_else(|| Array2::eye(n));
        let transformed = l.dot(&z);
        Array1::from_shape_fn(n, |i| mean[i] + transformed[i])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::golden;
    use approx::assert_abs_diff_eq;

    #[test]
    fn standard_normal_matches_numpy() {
        let mut rng = NumpyRandomState::new(42);
        for &expected in golden::NORMAL_S42.iter() {
            // System libm: matches to the last bit in practice, tolerance guards ulps.
            assert_abs_diff_eq!(rng.standard_normal(), expected, epsilon = 1e-15);
        }
    }

    #[test]
    fn multivariate_normal_keeps_stream_aligned() {
        let mut rng = NumpyRandomState::new(42);
        let mean = ndarray::array![1.0, 2.0];
        let cov = ndarray::array![[2.0, 0.5], [0.5, 1.0]];
        let _ = rng.multivariate_normal(mean.view(), cov.view());
        // After a 2-D draw, the next standard_normal must line up with NumPy.
        assert_abs_diff_eq!(
            rng.standard_normal(),
            golden::NORMAL_AFTER_MVN2_S42,
            epsilon = 1e-15
        );
    }
}
