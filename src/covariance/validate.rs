//! Covariance validation — port of `hmmlearn._utils._validate_covars`.

use super::store::CovarStore;
use crate::error::{HmmError, Result};
use crate::linalg::{eigvalsh, is_symmetric};
use ndarray::s;

const SYM_TOL: f64 = 1e-8;

/// Validate covariance shapes and positive-definiteness.
///
/// Port of `hmmlearn._utils._validate_covars`.
///
/// # Arguments
/// * `covars` — the covariance parameters to check, in compact form.
/// * `n_components` — expected number of states (enforced only for `Spherical`).
///
/// # Errors
/// * [`HmmError::DimensionMismatch`] — `Spherical` length ≠ `n_components`, or a
///   `Tied`/`Full` matrix is not square.
/// * [`HmmError::InvalidParameter`] — a `Spherical` or `Diag` variance is ≤ 0.
/// * [`HmmError::NotPositiveDefinite`] — a `Tied` matrix, or any component of a
///   `Full` array, is not symmetric (within `SYM_TOL`) or not positive-definite.
pub fn validate_covars(covars: &CovarStore, n_components: usize) -> Result<()> {
    match covars {
        // Length must equal the state count; every scalar variance must be positive.
        CovarStore::Spherical(c) => {
            if c.len() != n_components {
                return Err(HmmError::DimensionMismatch(
                    "'spherical' covars have length n_components".into(),
                ));
            }
            if c.iter().any(|&v| v <= 0.0) {
                return Err(HmmError::InvalidParameter(
                    "'spherical' covars must be positive".into(),
                ));
            }
        }
        // Every per-feature variance must be positive.
        CovarStore::Diag(c) => {
            if c.iter().any(|&v| v <= 0.0) {
                return Err(HmmError::InvalidParameter(
                    "'diag' covars must be positive".into(),
                ));
            }
        }
        // Single shared matrix: must be square, symmetric, and positive-definite.
        CovarStore::Tied(c) => {
            if c.nrows() != c.ncols() {
                return Err(HmmError::DimensionMismatch(
                    "'tied' covars must have shape (n_dim, n_dim)".into(),
                ));
            }
            if !is_symmetric(c.view(), SYM_TOL) || eigvalsh(c.view()).iter().any(|&v| v <= 0.0) {
                return Err(HmmError::NotPositiveDefinite(
                    "'tied' covars must be symmetric, positive-definite".into(),
                ));
            }
        }
        // Each per-state matrix must be square, symmetric, and positive-definite.
        CovarStore::Full(c) => {
            let (nc, nf1, nf2) = c.dim();
            if nf1 != nf2 {
                return Err(HmmError::DimensionMismatch(
                    "'full' covars must have shape (n_components, n_dim, n_dim)".into(),
                ));
            }
            for n in 0..nc {
                let cv = c.slice(s![n, .., ..]);
                if !is_symmetric(cv, SYM_TOL) || eigvalsh(cv).iter().any(|&v| v <= 0.0) {
                    return Err(HmmError::NotPositiveDefinite(format!(
                        "component {n} of 'full' covars must be symmetric, positive-definite"
                    )));
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array3};

    #[test]
    fn spherical_ok_and_errors() {
        assert!(validate_covars(&CovarStore::Spherical(array![1.0, 2.0, 3.0]), 3).is_ok());
        assert!(validate_covars(&CovarStore::Spherical(array![1.0, 2.0]), 3).is_err());
        assert!(validate_covars(&CovarStore::Spherical(array![1.0, -2.0, 3.0]), 3).is_err());
    }

    #[test]
    fn diag_ok_and_errors() {
        assert!(validate_covars(&CovarStore::Diag(array![[1.0, 2.0], [3.0, 4.0]]), 2).is_ok());
        assert!(validate_covars(&CovarStore::Diag(array![[1.0, 0.0], [3.0, 4.0]]), 2).is_err());
    }

    #[test]
    fn tied_pd_check() {
        assert!(validate_covars(&CovarStore::Tied(array![[2.0, 1.0], [1.0, 2.0]]), 3).is_ok());
        // symmetric but indefinite
        assert!(validate_covars(&CovarStore::Tied(array![[1.0, 2.0], [2.0, 1.0]]), 3).is_err());
    }

    #[test]
    fn full_pd_per_component() {
        let good = Array3::from_shape_vec((2, 2, 2), vec![2.0, 0.0, 0.0, 3.0, 1.0, 0.0, 0.0, 1.0])
            .unwrap();
        assert!(validate_covars(&CovarStore::Full(good), 2).is_ok());

        let bad = Array3::from_shape_vec(
            (2, 2, 2),
            vec![2.0, 0.0, 0.0, 3.0, 1.0, 2.0, 2.0, 1.0], // 2nd component indefinite
        )
        .unwrap();
        assert!(validate_covars(&CovarStore::Full(bad), 2).is_err());
    }
}
