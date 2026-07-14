//! Uniform draws: `random_sample` (53-bit doubles) and `randint`.

use super::NumpyRandomState;
use ndarray::Array1;

/// Smallest `2^k - 1` that is `>= rng` (NumPy's `gen_mask`).
///
/// Smears the highest set bit down through all lower bits.
///
/// # Arguments
/// * `rng` — the inclusive span to cover.
///
/// # Returns
/// The all-ones mask with just enough bits to hold `rng`.
fn gen_mask(mut rng: u64) -> u64 {
    rng |= rng >> 1;
    rng |= rng >> 2;
    rng |= rng >> 4;
    rng |= rng >> 8;
    rng |= rng >> 16;
    rng |= rng >> 32;
    rng
}

impl NumpyRandomState {
    /// A single uniform double in `[0, 1)`, matching `random_sample`.
    ///
    /// Uses the 53-bit construction `((a >> 5) * 2²⁶ + (b >> 6)) / 2⁵³` from two
    /// consecutive `u32` draws.
    ///
    /// # Returns
    /// A uniform double in `[0, 1)`.
    pub fn random_sample(&mut self) -> f64 {
        let a = (self.mt.next_u32() >> 5) as f64;
        let b = (self.mt.next_u32() >> 6) as f64;
        (a * 67108864.0 + b) / 9007199254740992.0
    }

    /// `n` uniform doubles in `[0, 1)`, filled in C order like `random_sample(n)`.
    ///
    /// # Arguments
    /// * `n` — number of samples to draw.
    ///
    /// # Returns
    /// A length-`n` array of uniform doubles in `[0, 1)`.
    pub fn random_sample_n(&mut self, n: usize) -> Array1<f64> {
        Array1::from_iter((0..n).map(|_| self.random_sample()))
    }

    /// Integers in `[low, high)` (default int64 dtype), filled in C order.
    ///
    /// Reproduces NumPy's masked-rejection scheme: draw a masked value in the
    /// span and reject any that overshoot, using one `u32` draw per value when
    /// the span fits in 32 bits, otherwise a `u64` draw.
    ///
    /// # Arguments
    /// * `low` — inclusive lower bound.
    /// * `high` — exclusive upper bound.
    /// * `n` — number of integers to draw.
    ///
    /// # Returns
    /// A length-`n` array of integers in `[low, high)`.
    ///
    /// # Panics
    /// Panics unless `high > low`.
    pub fn randint(&mut self, low: i64, high: i64, n: usize) -> Array1<i64> {
        assert!(high > low, "randint requires low < high");
        let rng = (high - low - 1) as u64;
        let mut out = Array1::<i64>::zeros(n);
        if rng == 0 {
            out.fill(low);
            return out;
        }
        let mask = gen_mask(rng);
        let use_32 = rng <= 0xFFFF_FFFF;
        for slot in out.iter_mut() {
            loop {
                let v = if use_32 {
                    (self.mt.next_u32() as u64) & mask
                } else {
                    self.mt.next_u64() & mask
                };
                if v <= rng {
                    *slot = low + v as i64;
                    break;
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::golden;

    #[test]
    fn random_sample_matches_numpy() {
        let mut rng = NumpyRandomState::new(42);
        for &expected in golden::SAMPLE_SEED_42.iter() {
            assert_eq!(rng.random_sample(), expected);
        }
    }

    #[test]
    fn random_sample_far_stream_matches() {
        // Draw 1001 samples; index 1000 crosses the twist boundary repeatedly.
        let mut rng = NumpyRandomState::new(42);
        let s = rng.random_sample_n(1001);
        assert_eq!(s[1000], golden::SAMPLE_SEED_42_AT_1000);
    }

    #[test]
    fn randint_matches_numpy() {
        assert_eq!(
            NumpyRandomState::new(42).randint(-20, 20, 12).to_vec(),
            golden::RANDINT_M20_20_S42
        );
        assert_eq!(
            NumpyRandomState::new(0).randint(0, 3, 16).to_vec(),
            golden::RANDINT_0_3_S0
        );
        assert_eq!(
            NumpyRandomState::new(42).randint(0, 256, 12).to_vec(),
            golden::RANDINT_0_256_S42
        );
        assert_eq!(
            NumpyRandomState::new(42).randint(0, 1 << 40, 5).to_vec(),
            golden::RANDINT_0_2P40_S42
        );
    }
}
