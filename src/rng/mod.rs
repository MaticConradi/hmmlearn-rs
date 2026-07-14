//! Bit-exact reproduction of NumPy's legacy `RandomState`.
//!
//! [`NumpyRandomState`] wraps an [`mt19937::Mt19937`] generator and the cached
//! Gaussian used by `standard_normal`. Each distribution is implemented in its
//! own submodule as an `impl` block, mirroring the routines in NumPy's
//! `distributions.c` / `legacy-distributions.c`.

pub mod choice;
pub mod discrete;
pub mod gamma;
pub mod gauss;
pub mod mt19937;
pub mod uniform;

#[cfg(test)]
pub(crate) mod golden;

use mt19937::Mt19937;

/// A NumPy-compatible `RandomState`.
#[derive(Clone)]
pub struct NumpyRandomState {
    /// The underlying MT19937 bit generator driving every draw.
    pub(crate) mt: Mt19937,
    /// Cached second value from the polar Box–Muller pair (`has_gauss`/`gauss`).
    pub(crate) gauss: Option<f64>,
}

impl NumpyRandomState {
    /// Construct from a scalar seed, like `np.random.RandomState(seed)`.
    ///
    /// # Arguments
    /// * `seed` — a 32-bit seed fed to Knuth's `init_genrand`.
    ///
    /// # Returns
    /// A generator with an empty Gaussian cache and its MT19937 state seeded.
    pub fn new(seed: u32) -> Self {
        NumpyRandomState {
            mt: Mt19937::new_seed(seed),
            gauss: None,
        }
    }

    /// Construct from an array seed, like `np.random.RandomState([..])`.
    ///
    /// # Arguments
    /// * `key` — the seed words fed to MT19937's `init_by_array`.
    ///
    /// # Returns
    /// A generator with an empty Gaussian cache and its MT19937 state seeded.
    ///
    /// # Panics
    /// Panics if `key` is empty (see [`Mt19937::new_by_array`]).
    pub fn from_array_seed(key: &[u32]) -> Self {
        NumpyRandomState {
            mt: Mt19937::new_by_array(key),
            gauss: None,
        }
    }
}
