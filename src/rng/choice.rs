//! Weighted `choice` (with replacement).
//!
//! Reproduces NumPy's probability-vector path: normalize the cumulative sum and
//! `searchsorted(cdf, U, side="right")`.

use super::NumpyRandomState;
use ndarray::ArrayView1;

impl NumpyRandomState {
    /// Draw one index in `0..p.len()` with probabilities proportional to `p`.
    ///
    /// Matches `choice(len(p), p=p)` for a scalar draw: builds the cumulative
    /// sum, then `searchsorted(cdf/total, U, side="right")` against one uniform
    /// draw, clamped to the last valid index.
    ///
    /// # Arguments
    /// * `p` — non-negative weights, length `>= 1`; need not be normalized.
    ///
    /// # Returns
    /// The selected index in `0..p.len()`.
    ///
    /// # Panics
    /// Panics if `p` is empty.
    pub fn choice_weighted(&mut self, p: ArrayView1<f64>) -> usize {
        let n = p.len();
        let mut cdf = Vec::with_capacity(n);
        let mut acc = 0.0;
        for &pi in p.iter() {
            acc += pi;
            cdf.push(acc);
        }
        let total = *cdf.last().expect("choice requires a non-empty p");
        let u = self.random_sample();
        // searchsorted(cdf/total, u, side="right"): count of normalized cdf <= u.
        let mut idx = 0usize;
        for &c in &cdf {
            if c / total <= u {
                idx += 1;
            } else {
                break;
            }
        }
        idx.min(n - 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::golden;

    #[test]
    fn choice_weighted_matches_numpy() {
        let mut rng = NumpyRandomState::new(6);
        let p = ndarray::array![0.2, 0.5, 0.3];
        let got: Vec<i64> = (0..12)
            .map(|_| rng.choice_weighted(p.view()) as i64)
            .collect();
        assert_eq!(got, golden::CHOICE_3_S6);
    }
}
