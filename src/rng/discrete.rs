//! Discrete draws: `poisson`, `binomial`, and `multinomial`.
//!
//! Ports of NumPy's `rk_poisson` (multiplication + PTRS transformed rejection),
//! `rk_binomial` (inversion), and `rk_multinomial` (sequential conditional
//! binomials). These return integers and reproduce NumPy exactly.

use super::NumpyRandomState;
use crate::special::ln_gamma;
use ndarray::{Array1, ArrayView1};

impl NumpyRandomState {
    /// Poisson with rate `lam`, matching `rk_poisson`.
    ///
    /// Dispatches on `lam`: the PTRS transformed-rejection method for
    /// `lam >= 10`, the multiplication method for `0 < lam < 10`, and `0` for
    /// `lam == 0`.
    ///
    /// # Arguments
    /// * `lam` — the Poisson rate `λ`.
    ///
    /// # Returns
    /// A draw from `Poisson(lam)`.
    pub fn poisson(&mut self, lam: f64) -> i64 {
        if lam >= 10.0 {
            self.poisson_ptrs(lam)
        } else if lam == 0.0 {
            0
        } else {
            self.poisson_mult(lam)
        }
    }

    /// Small-`lam` multiplication method (`rk_poisson_mult`).
    ///
    /// Multiplies uniform draws until the running product drops below `e^(-lam)`,
    /// returning the count of factors taken.
    ///
    /// # Arguments
    /// * `lam` — the Poisson rate `λ`; assumed `> 0`.
    ///
    /// # Returns
    /// A draw from `Poisson(lam)`.
    fn poisson_mult(&mut self, lam: f64) -> i64 {
        let enlam = (-lam).exp();
        let mut x = 0i64;
        let mut prod = 1.0;
        loop {
            prod *= self.random_sample();
            if prod > enlam {
                x += 1;
            } else {
                return x;
            }
        }
    }

    /// Large-`lam` transformed-rejection method (`rk_poisson_ptrs`).
    ///
    /// Hörmann's PTRS algorithm: propose a candidate via the transformed
    /// generator and accept it through a squeeze plus a log-factorial acceptance
    /// test (using [`ln_gamma`]).
    ///
    /// # Arguments
    /// * `lam` — the Poisson rate `λ`; assumed `>= 10`.
    ///
    /// # Returns
    /// A draw from `Poisson(lam)`.
    fn poisson_ptrs(&mut self, lam: f64) -> i64 {
        let slam = lam.sqrt();
        let loglam = lam.ln();
        let b = 0.931 + 2.53 * slam;
        let a = -0.059 + 0.02483 * b;
        let invalpha = 1.1239 + 1.1328 / (b - 3.4);
        let vr = 0.9277 - 3.6224 / (b - 2.0);
        loop {
            let u = self.random_sample() - 0.5;
            let v = self.random_sample();
            let us = 0.5 - u.abs();
            let k = ((2.0 * a / us + b) * u + lam + 0.43).floor();
            if us >= 0.07 && v <= vr {
                return k as i64;
            }
            if k < 0.0 || (us < 0.013 && v > us) {
                continue;
            }
            if v.ln() + invalpha.ln() - (a / (us * us) + b).ln()
                <= -lam + k * loglam - ln_gamma(k + 1.0)
            {
                return k as i64;
            }
        }
    }

    /// Binomial via inversion, matching `rk_binomial` for `n*p <= 30`.
    ///
    /// Uses the `p > 0.5` reflection (sample with `1-p` and mirror). Intentional
    /// deviation: the BTPE branch (`n*min(p,1-p) > 30`) is not reached by
    /// hmmlearn's small-count sampling and is omitted, so this always falls back
    /// to inversion. Returns `0` for `n == 0` or `p == 0`.
    ///
    /// # Arguments
    /// * `n` — number of trials.
    /// * `p` — success probability per trial.
    ///
    /// # Returns
    /// A draw from `Binomial(n, p)`.
    fn binomial(&mut self, n: i64, p: f64) -> i64 {
        if n == 0 || p == 0.0 {
            return 0;
        }
        if p <= 0.5 {
            self.binomial_inversion(n, p)
        } else {
            n - self.binomial_inversion(n, 1.0 - p)
        }
    }

    /// Binomial inversion (CDF) sampler, matching `rk_binomial_inversion`.
    ///
    /// Accumulates the PMF from `x = 0` upward against one uniform draw, retrying
    /// from scratch if the search runs past a safety bound `> n*p`.
    ///
    /// # Arguments
    /// * `n` — number of trials.
    /// * `p` — success probability per trial; assumed `<= 0.5`.
    ///
    /// # Returns
    /// A draw from `Binomial(n, p)`.
    fn binomial_inversion(&mut self, n: i64, p: f64) -> i64 {
        let q = 1.0 - p;
        let qn = (n as f64 * q.ln()).exp();
        let np = n as f64 * p;
        let bound = (n as f64).min(np + 10.0 * (np * q + 1.0).sqrt());
        let mut x = 0i64;
        let mut px = qn;
        let mut u = self.random_sample();
        while u > px {
            x += 1;
            if x as f64 > bound {
                x = 0;
                px = qn;
                u = self.random_sample();
            } else {
                u -= px;
                px = ((n - x + 1) as f64 * p * px) / (x as f64 * q);
            }
        }
        x
    }

    /// Multinomial counts for `n` trials over `pix`, matching `rk_multinomial`.
    ///
    /// Draws each category from a conditional binomial on the remaining trials
    /// and remaining probability mass, stopping early once trials are exhausted
    /// and assigning any leftover to the final category.
    ///
    /// # Arguments
    /// * `n` — total number of trials.
    /// * `pix` — category probabilities, length `d`.
    ///
    /// # Returns
    /// A length-`d` array of counts summing to `n`.
    pub fn multinomial(&mut self, n: i64, pix: ArrayView1<f64>) -> Array1<i64> {
        let d = pix.len();
        let mut mnix = Array1::<i64>::zeros(d);
        let mut remaining = 1.0;
        let mut dn = n;
        for j in 0..d - 1 {
            mnix[j] = self.binomial(dn, pix[j] / remaining);
            dn -= mnix[j];
            if dn <= 0 {
                break;
            }
            remaining -= pix[j];
        }
        if dn > 0 {
            mnix[d - 1] = dn;
        }
        mnix
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::golden;

    #[test]
    fn poisson_matches_numpy() {
        let mut rng = NumpyRandomState::new(1);
        let got: Vec<i64> = (0..15).map(|_| rng.poisson(3.0)).collect();
        assert_eq!(got, golden::POISSON_3_S1);

        let mut rng = NumpyRandomState::new(2);
        let got: Vec<i64> = (0..15).map(|_| rng.poisson(15.0)).collect();
        assert_eq!(got, golden::POISSON_15_S2);
    }

    #[test]
    fn multinomial_matches_numpy() {
        let mut rng = NumpyRandomState::new(3);
        assert_eq!(
            rng.multinomial(10, ndarray::array![0.2, 0.3, 0.5].view())
                .to_vec(),
            golden::MULTINOMIAL_10_S3
        );
        let mut rng = NumpyRandomState::new(8);
        assert_eq!(
            rng.multinomial(1, ndarray::array![0.1, 0.6, 0.3].view())
                .to_vec(),
            golden::MULTINOMIAL_1_S8
        );
        let mut rng = NumpyRandomState::new(9);
        assert_eq!(
            rng.multinomial(20, ndarray::array![0.1, 0.2, 0.3, 0.4].view())
                .to_vec(),
            golden::MULTINOMIAL_20_S9
        );
    }
}
