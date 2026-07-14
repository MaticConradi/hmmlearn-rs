//! Exponential, gamma, and Dirichlet draws.
//!
//! Ports of NumPy's `legacy_standard_exponential`, `legacy_standard_gamma`
//! (Marsaglia–Tsang), `legacy_gamma`, and `dirichlet`. Transcendental math uses
//! `std` f64 methods (system libm), matching NumPy to within a rare last ulp.

use super::NumpyRandomState;
use ndarray::{Array1, ArrayView1};

impl NumpyRandomState {
    /// `standard_exponential`: `-ln(1 - U)`.
    ///
    /// Inversion sampling of the unit-rate exponential from one uniform draw.
    ///
    /// # Returns
    /// A draw from `Exp(1)`.
    pub fn standard_exponential(&mut self) -> f64 {
        -(1.0 - self.random_sample()).ln()
    }

    /// `standard_gamma(shape)` via Marsaglia–Tsang (with the `shape < 1` branch).
    ///
    /// For `shape >= 1` uses the Marsaglia–Tsang squeeze on a normal/uniform
    /// pair; for `shape < 1` uses Ahrens–Dieter's exponential-boosted rejection.
    /// Short-circuits `shape == 1` to an exponential and `shape == 0` to `0`.
    ///
    /// # Arguments
    /// * `shape` — the gamma shape parameter `k`.
    ///
    /// # Returns
    /// A draw from `Gamma(shape, 1)`.
    pub fn standard_gamma(&mut self, shape: f64) -> f64 {
        if shape == 1.0 {
            return self.standard_exponential();
        }
        if shape == 0.0 {
            return 0.0;
        }
        if shape < 1.0 {
            loop {
                let u = self.random_sample();
                let v = self.standard_exponential();
                if u <= 1.0 - shape {
                    let x = u.powf(1.0 / shape);
                    if x <= v {
                        return x;
                    }
                } else {
                    let y = -((1.0 - u) / shape).ln();
                    let x = (1.0 - shape + shape * y).powf(1.0 / shape);
                    if x <= v + y {
                        return x;
                    }
                }
            }
        }
        let b = shape - 1.0 / 3.0;
        let c = 1.0 / (9.0 * b).sqrt();
        loop {
            let (x, mut v);
            loop {
                let xx = self.standard_normal();
                let vv = 1.0 + c * xx;
                if vv > 0.0 {
                    x = xx;
                    v = vv;
                    break;
                }
            }
            v = v * v * v;
            let u = self.random_sample();
            if u < 1.0 - 0.0331 * (x * x) * (x * x) {
                return b * v;
            }
            if u.ln() < 0.5 * x * x + b * (1.0 - v + v.ln()) {
                return b * v;
            }
        }
    }

    /// `gamma(shape, scale)` = `scale * standard_gamma(shape)`.
    ///
    /// # Arguments
    /// * `shape` — the gamma shape parameter `k`.
    /// * `scale` — the scale parameter `θ`.
    ///
    /// # Returns
    /// A draw from `Gamma(shape, scale)`.
    pub fn gamma(&mut self, shape: f64, scale: f64) -> f64 {
        scale * self.standard_gamma(shape)
    }

    /// A single Dirichlet draw: per-component gammas normalized by `1/sum`.
    ///
    /// # Arguments
    /// * `alpha` — the concentration vector, length `d`.
    ///
    /// # Returns
    /// A length-`d` probability vector summing to 1, drawn from `Dir(alpha)`.
    pub fn dirichlet(&mut self, alpha: ArrayView1<f64>) -> Array1<f64> {
        let mut val = Array1::from_iter(alpha.iter().map(|&a| self.standard_gamma(a)));
        let acc: f64 = val.sum();
        let inv = 1.0 / acc;
        val.mapv_inplace(|v| v * inv);
        val
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::golden;

    fn assert_seq_close(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (a, e) in actual.iter().zip(expected) {
            // Relative tolerance: transcendental draws can differ by a last ulp.
            assert!(
                (a - e).abs() <= 1e-12 * e.abs().max(1e-300) + 1e-300,
                "{a} vs {e}"
            );
        }
    }

    #[test]
    fn exponential_matches_numpy() {
        let mut rng = NumpyRandomState::new(7);
        let got: Vec<f64> = (0..8).map(|_| rng.standard_exponential()).collect();
        assert_seq_close(&got, &golden::EXPONENTIAL_S7);
    }

    #[test]
    fn gamma_both_branches_match_numpy() {
        let mut rng = NumpyRandomState::new(3);
        let big: Vec<f64> = (0..8).map(|_| rng.standard_gamma(2.5)).collect();
        assert_seq_close(&big, &golden::GAMMA_SHAPE2P5_S3);

        let mut rng = NumpyRandomState::new(3);
        let small: Vec<f64> = (0..8).map(|_| rng.standard_gamma(0.4)).collect();
        assert_seq_close(&small, &golden::GAMMA_SHAPE0P4_S3);
    }

    #[test]
    fn gamma_with_scale_matches_numpy() {
        let mut rng = NumpyRandomState::new(9);
        let got: Vec<f64> = (0..6).map(|_| rng.gamma(3.0, 2.0)).collect();
        assert_seq_close(&got, &golden::GAMMA_3_SCALE2_S9);
    }

    #[test]
    fn dirichlet_matches_numpy() {
        let mut rng = NumpyRandomState::new(5);
        let got = rng.dirichlet(ndarray::array![2.0, 3.0, 4.0, 1.0].view());
        assert_seq_close(got.as_slice().unwrap(), &golden::DIRICHLET_2341_S5);

        let mut rng = NumpyRandomState::new(11);
        let third = 1.0 / 3.0;
        let got = rng.dirichlet(ndarray::array![third, third, third].view());
        assert_seq_close(got.as_slice().unwrap(), &golden::DIRICHLET_THIRD_S11);
    }
}
