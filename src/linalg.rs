//! Small dense linear-algebra helpers backed by `nalgebra` (pure Rust, no LAPACK).
//!
//! These operate on the modest `n_features × n_features` covariance / scale
//! matrices that the models manipulate. Inputs and outputs are `ndarray` arrays;
//! conversion to `nalgebra` happens internally.

use nalgebra::{DMatrix, SymmetricEigen};
use ndarray::{Array1, Array2, ArrayView2};

/// Convert an `ndarray` 2-D view into a column-major `nalgebra` matrix.
///
/// # Arguments
/// * `a` — source 2-D view, of any memory layout.
///
/// # Returns
/// A `nalgebra` `DMatrix` with the same elements and shape.
fn to_na(a: ArrayView2<f64>) -> DMatrix<f64> {
    let (r, c) = a.dim();
    // `a.iter()` yields elements in logical row-major order; `from_row_iterator`
    // fills row by row, so this is correct for any memory layout.
    DMatrix::from_row_iterator(r, c, a.iter().copied())
}

/// Convert a `nalgebra` matrix back into a row-major `ndarray` array.
///
/// # Arguments
/// * `m` — source `nalgebra` matrix.
///
/// # Returns
/// An owned `(nrows, ncols)` `ndarray` array with the same elements.
fn from_na(m: &DMatrix<f64>) -> Array2<f64> {
    Array2::from_shape_fn((m.nrows(), m.ncols()), |(i, j)| m[(i, j)])
}

/// Lower-triangular Cholesky factor `L` with `a = L Lᵀ`.
///
/// # Arguments
/// * `a` — square, symmetric matrix to factor.
///
/// # Returns
/// `Some(L)`, the lower-triangular factor, or `None` if `a` is not symmetric
/// positive-definite.
pub fn cholesky_lower(a: ArrayView2<f64>) -> Option<Array2<f64>> {
    let chol = nalgebra::Cholesky::new(to_na(a))?;
    Some(from_na(&chol.l()))
}

/// Solve `L X = B` for a lower-triangular `L` (forward substitution).
///
/// # Arguments
/// * `l` — `(n, n)` lower-triangular matrix; only the lower triangle and diagonal
///   are read.
/// * `b` — `(n, m)` right-hand side.
///
/// # Returns
/// The `(n, m)` solution `X`.
///
/// # Panics
/// Indexes out of bounds unless `l` is `(n, n)` and `b` has at least `n` rows.
pub fn solve_lower_triangular(l: ArrayView2<f64>, b: ArrayView2<f64>) -> Array2<f64> {
    let n = l.nrows();
    let m = b.ncols();
    let mut x = Array2::<f64>::zeros((n, m));
    for col in 0..m {
        for i in 0..n {
            let mut acc = b[[i, col]];
            for k in 0..i {
                acc -= l[[i, k]] * x[[k, col]];
            }
            x[[i, col]] = acc / l[[i, i]];
        }
    }
    x
}

/// Eigenvalues of a symmetric matrix, in ascending order (like `numpy.eigvalsh`).
///
/// # Arguments
/// * `a` — square, symmetric matrix.
///
/// # Returns
/// The `n` real eigenvalues sorted ascending.
///
/// # Panics
/// If an eigenvalue is `NaN`, so the ascending sort's `partial_cmp` returns
/// `None` and is unwrapped.
pub fn eigvalsh(a: ArrayView2<f64>) -> Array1<f64> {
    let eig = SymmetricEigen::new(to_na(a));
    let mut vals: Vec<f64> = eig.eigenvalues.iter().copied().collect();
    vals.sort_by(|x, y| x.partial_cmp(y).unwrap());
    Array1::from(vals)
}

/// Sign and natural log of the absolute determinant, like `numpy.linalg.slogdet`.
///
/// # Arguments
/// * `a` — square matrix.
///
/// # Returns
/// `(sign, ln|det(a)|)`, where `sign` is the sign of the determinant.
pub fn slogdet(a: ArrayView2<f64>) -> (f64, f64) {
    let det = to_na(a).determinant();
    (det.signum(), det.abs().ln())
}

/// Log-determinant of a symmetric positive-definite matrix via its Cholesky
/// factor: `log|a| = 2 Σ log(diag(L))`.
///
/// # Arguments
/// * `a` — square, symmetric positive-definite matrix.
///
/// # Returns
/// `Some(log|a|)`, or `None` if `a` is not SPD (Cholesky fails).
pub fn logdet_spd(a: ArrayView2<f64>) -> Option<f64> {
    let l = cholesky_lower(a)?;
    Some(2.0 * (0..l.nrows()).map(|i| l[[i, i]].ln()).sum::<f64>())
}

/// Log of the absolute determinant, `NaN` if the determinant is negative.
///
/// Port of `hmmlearn._utils.logdet` (`slogdet` with a sign check).
///
/// # Arguments
/// * `a` — square matrix.
///
/// # Returns
/// `ln|det(a)|`, or `NaN` when `det(a) < 0`.
pub fn logdet(a: ArrayView2<f64>) -> f64 {
    let (sign, logabs) = slogdet(a);
    if sign < 0.0 {
        f64::NAN
    } else {
        logabs
    }
}

/// Matrix inverse, or `None` if singular.
///
/// # Arguments
/// * `a` — square matrix to invert.
///
/// # Returns
/// `Some(a⁻¹)`, or `None` if `a` is singular.
pub fn inv(a: ArrayView2<f64>) -> Option<Array2<f64>> {
    to_na(a).try_inverse().map(|m| from_na(&m))
}

/// Stationary distribution of a row-stochastic transition matrix `p`.
///
/// Solves `vᵀ P = vᵀ` with `Σ v = 1` (the left eigenvector for eigenvalue 1) by
/// replacing the last row of `(Pᵀ - I)` with the normalization constraint and
/// solving the resulting linear system. Port of
/// `BaseHMM.get_stationary_distribution`.
///
/// # Arguments
/// * `p` — `(n, n)` row-stochastic transition matrix.
///
/// # Returns
/// The length-`n` stationary distribution `v`.
///
/// # Panics
/// If the assembled linear system is singular (the LU solve `expect`s a
/// solution).
pub fn stationary_distribution(p: ArrayView2<f64>) -> Array1<f64> {
    let n = p.nrows();
    let mut a = DMatrix::<f64>::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            a[(i, j)] = p[[j, i]] - if i == j { 1.0 } else { 0.0 };
        }
    }
    // Replace the last equation with Σ v = 1.
    for j in 0..n {
        a[(n - 1, j)] = 1.0;
    }
    let mut b = nalgebra::DVector::<f64>::zeros(n);
    b[n - 1] = 1.0;
    let v = a.lu().solve(&b).expect("singular stationary system");
    Array1::from_iter(v.iter().copied())
}

/// Whether `a` equals its transpose within `tol` (absolute).
///
/// # Arguments
/// * `a` — matrix to test.
/// * `tol` — maximum allowed absolute difference between `a[i,j]` and `a[j,i]`.
///
/// # Returns
/// `true` if `a` is square and symmetric to within `tol`; `false` otherwise
/// (including non-square inputs).
pub fn is_symmetric(a: ArrayView2<f64>, tol: f64) -> bool {
    let n = a.nrows();
    if a.ncols() != n {
        return false;
    }
    for i in 0..n {
        for j in (i + 1)..n {
            if (a[[i, j]] - a[[j, i]]).abs() > tol {
                return false;
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::assert_close;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn cholesky_diagonal() {
        let a = array![[4.0, 0.0], [0.0, 9.0]];
        let l = cholesky_lower(a.view()).unwrap();
        assert_close(&l, &array![[2.0, 0.0], [0.0, 3.0]], 1e-12);
    }

    #[test]
    fn cholesky_reconstructs() {
        let a = array![[2.0, 1.0], [1.0, 2.0]];
        let l = cholesky_lower(a.view()).unwrap();
        let recon = l.dot(&l.t());
        assert_close(&recon, &a, 1e-12);
    }

    #[test]
    fn cholesky_rejects_non_pd() {
        let a = array![[1.0, 2.0], [2.0, 1.0]]; // indefinite
        assert!(cholesky_lower(a.view()).is_none());
    }

    #[test]
    fn solve_lower_triangular_works() {
        let l = array![[2.0, 0.0], [1.0, 3.0]];
        let b = array![[2.0], [7.0]];
        let x = solve_lower_triangular(l.view(), b.view());
        // 2*x0 = 2 -> x0 = 1; 1*1 + 3*x1 = 7 -> x1 = 2
        assert_close(&x, &array![[1.0], [2.0]], 1e-12);
    }

    #[test]
    fn eigvalsh_sorted_ascending() {
        let a = array![[3.0, 0.0], [0.0, 2.0]];
        let vals = eigvalsh(a.view());
        assert_close(&vals, &array![2.0, 3.0], 1e-12);
    }

    #[test]
    fn slogdet_and_logdet_spd_agree() {
        let a = array![[2.0, 0.0], [0.0, 3.0]];
        let (sign, logabs) = slogdet(a.view());
        assert_eq!(sign, 1.0);
        assert_abs_diff_eq!(logabs, 6.0_f64.ln(), epsilon = 1e-12);
        assert_abs_diff_eq!(logdet_spd(a.view()).unwrap(), 6.0_f64.ln(), epsilon = 1e-12);
    }

    #[test]
    fn inv_diagonal() {
        let a = array![[2.0, 0.0], [0.0, 4.0]];
        let ai = inv(a.view()).unwrap();
        assert_close(&ai, &array![[0.5, 0.0], [0.0, 0.25]], 1e-12);
    }

    #[test]
    fn symmetry_check() {
        assert!(is_symmetric(array![[1.0, 2.0], [2.0, 5.0]].view(), 1e-12));
        assert!(!is_symmetric(array![[1.0, 2.0], [3.0, 5.0]].view(), 1e-12));
    }

    #[test]
    fn stationary_distribution_is_fixed_point() {
        // test_base.py::test_stationary_distribution: π @ P == π and Σπ == 1.
        let p = array![[0.7, 0.2, 0.1], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]];
        let pi = stationary_distribution(p.view());
        assert_abs_diff_eq!(pi.sum(), 1.0, epsilon = 1e-12);
        let pushed = pi.dot(&p);
        assert_close(&pushed, &pi, 1e-12);
    }
}
