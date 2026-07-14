//! Compressed covariance storage and expansion to full matrices.
//!
//! `CovarStore` is the Rust counterpart of hmmlearn's private `_covars_`: the
//! minimal per-type representation. `full` is the `covars_` property — the
//! expansion to dense `(n_components, n_features, n_features)` matrices,
//! mirroring `hmmlearn.utils.fill_covars`.

use super::types::CovarianceType;
use ndarray::{s, Array1, Array2, Array3, ArrayView2, Axis};

/// Covariance parameters in their compact, per-type form.
#[derive(Debug, Clone, PartialEq)]
pub enum CovarStore {
    /// `(n_components,)` scalar variances.
    Spherical(Array1<f64>),
    /// `(n_components, n_features)` per-feature variances.
    Diag(Array2<f64>),
    /// `(n_components, n_features, n_features)` full matrices.
    Full(Array3<f64>),
    /// `(n_features, n_features)` single shared matrix.
    Tied(Array2<f64>),
}

impl CovarStore {
    /// The covariance type tag.
    pub fn covariance_type(&self) -> CovarianceType {
        match self {
            CovarStore::Spherical(_) => CovarianceType::Spherical,
            CovarStore::Diag(_) => CovarianceType::Diag,
            CovarStore::Full(_) => CovarianceType::Full,
            CovarStore::Tied(_) => CovarianceType::Tied,
        }
    }

    /// Expand to dense `(n_components, n_features, n_features)` matrices.
    ///
    /// Port of `fill_covars(self, covariance_type, n_components, n_features)`.
    ///
    /// # Arguments
    /// * `n_components` — number of states; the leading axis of the output.
    /// * `n_features` — feature dimension; the trailing two axes.
    ///
    /// # Returns
    /// A `(n_components, n_features, n_features)` array. `Spherical`/`Diag`
    /// entries populate the diagonal, `Tied` tiles its single matrix across all
    /// states, and `Full` is copied as-is.
    pub fn full(&self, n_components: usize, n_features: usize) -> Array3<f64> {
        let mut out = Array3::<f64>::zeros((n_components, n_features, n_features));
        match self {
            CovarStore::Full(c) => out.assign(c),
            CovarStore::Diag(c) => {
                for i in 0..n_components {
                    for d in 0..n_features {
                        out[[i, d, d]] = c[[i, d]];
                    }
                }
            }
            CovarStore::Tied(c) => {
                for i in 0..n_components {
                    out.slice_mut(s![i, .., ..]).assign(c);
                }
            }
            CovarStore::Spherical(c) => {
                for i in 0..n_components {
                    for d in 0..n_features {
                        out[[i, d, d]] = c[i];
                    }
                }
            }
        }
        out
    }

    /// The dense `(n_features, n_features)` covariance for a single state.
    ///
    /// # Arguments
    /// * `state` — index of the state to expand.
    /// * `n_features` — feature dimension, used to size the diagonal for the
    ///   `Spherical` case.
    ///
    /// # Returns
    /// The dense `(n_features, n_features)` covariance for `state`. `Tied`
    /// ignores `state` and returns its single shared matrix.
    ///
    /// # Panics
    /// If `state` is out of bounds for the stored per-state parameters
    /// (`Spherical`, `Diag`, or `Full`).
    pub fn covariance_of(&self, state: usize, n_features: usize) -> Array2<f64> {
        match self {
            CovarStore::Spherical(c) => Array2::from_diag(&Array1::from_elem(n_features, c[state])),
            CovarStore::Diag(c) => Array2::from_diag(&c.row(state).to_owned()),
            CovarStore::Tied(c) => c.clone(),
            CovarStore::Full(c) => c.index_axis(Axis(0), state).to_owned(),
        }
    }
}

/// Broadcast a single `(n_features, n_features)` covariance template to the
/// storage shape of `covariance_type`. Port of
/// `distribute_covar_matrix_to_match_covariance_type`.
///
/// # Arguments
/// * `tied_cv` — the `(n_features, n_features)` covariance template.
/// * `covariance_type` — target storage parameterization.
/// * `n_components` — number of states to replicate across.
///
/// # Returns
/// A `CovarStore` in the requested parameterization: `Spherical` uses the mean
/// of `tied_cv` (0.0 if empty), `Diag` takes its diagonal, `Tied` keeps it
/// as-is, and `Full` tiles it across states.
pub fn distribute_covar(
    tied_cv: ArrayView2<f64>,
    covariance_type: CovarianceType,
    n_components: usize,
) -> CovarStore {
    let nf = tied_cv.ncols();
    match covariance_type {
        CovarianceType::Spherical => {
            let mean = tied_cv.mean().unwrap_or(0.0);
            CovarStore::Spherical(Array1::from_elem(n_components, mean))
        }
        CovarianceType::Diag => {
            let diag = Array1::from_shape_fn(nf, |i| tied_cv[[i, i]]);
            let mut d = Array2::zeros((n_components, nf));
            for c in 0..n_components {
                d.row_mut(c).assign(&diag);
            }
            CovarStore::Diag(d)
        }
        CovarianceType::Tied => CovarStore::Tied(tied_cv.to_owned()),
        CovarianceType::Full => {
            let mut f = Array3::zeros((n_components, nf, nf));
            for c in 0..n_components {
                f.slice_mut(s![c, .., ..]).assign(&tied_cv);
            }
            CovarStore::Full(f)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    // Mirrors hmmlearn/tests/test_utils.py::test_fill_covars.

    #[test]
    fn fill_full_is_identity() {
        let full = Array3::from_shape_vec((3, 2, 2), (1..=12).map(|x| x as f64).collect()).unwrap();
        let store = CovarStore::Full(full.clone());
        assert_eq!(store.full(3, 2), full);
    }

    #[test]
    fn fill_diag_builds_diagonals() {
        let diag = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let expected = ndarray::stack![
            ndarray::Axis(0),
            array![[1.0, 0.0], [0.0, 2.0]],
            array![[3.0, 0.0], [0.0, 4.0]],
            array![[5.0, 0.0], [0.0, 6.0]]
        ];
        assert_eq!(CovarStore::Diag(diag).full(3, 2), expected);
    }

    #[test]
    fn fill_tied_tiles() {
        let tied = array![[1.0, 2.0], [3.0, 4.0]];
        let one = array![[1.0, 2.0], [3.0, 4.0]];
        let expected = ndarray::stack![ndarray::Axis(0), one, one, one];
        assert_eq!(CovarStore::Tied(tied).full(3, 2), expected);
    }

    #[test]
    fn fill_spherical_scales_identity() {
        let sph = array![1.0, 2.0, 3.0];
        let expected = ndarray::stack![
            ndarray::Axis(0),
            array![[1.0, 0.0], [0.0, 1.0]],
            array![[2.0, 0.0], [0.0, 2.0]],
            array![[3.0, 0.0], [0.0, 3.0]]
        ];
        assert_eq!(CovarStore::Spherical(sph).full(3, 2), expected);
    }
}
