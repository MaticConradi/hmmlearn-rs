//! Array normalization helpers and sequence splitting.
//!
//! Ports of `hmmlearn.utils.{normalize, log_normalize}` and
//! `hmmlearn._utils.split_X_lengths`.

use crate::error::{HmmError, Result};
use crate::special::logsumexp;
use ndarray::{Array1, Array2, Axis};

/// Normalize a 1-D array in place so that it sums to 1.
///
/// Mirrors `normalize(a, axis=None)`. A zero-sum array is left unchanged rather
/// than producing `NaN`.
///
/// # Arguments
/// * `a` — array to normalize in place; divided by its own sum unless that sum
///   is exactly zero.
pub fn normalize1(a: &mut Array1<f64>) {
    let s = a.sum();
    if s != 0.0 {
        a.mapv_inplace(|x| x / s);
    }
}

/// Normalize a 2-D array in place so that it sums to 1 along `axis`.
///
/// Mirrors `normalize(a, axis=axis)`. Lanes that sum to zero are preserved as
/// zero (the divisor is forced to 1), which keeps structural zeros — e.g. the
/// forbidden transitions of a left-to-right model — intact.
///
/// # Arguments
/// * `a` — array to normalize in place.
/// * `axis` — axis whose lanes are each rescaled to sum to 1.
pub fn normalize_axis(a: &mut Array2<f64>, axis: usize) {
    let ax = Axis(axis);
    let sums = a.sum_axis(ax);
    // Each lane along `ax` should sum to 1; divide it by its (zero-guarded) sum.
    for (mut lane, &s) in a.lanes_mut(ax).into_iter().zip(sums.iter()) {
        let d = if s == 0.0 { 1.0 } else { s };
        lane.mapv_inplace(|x| x / d);
    }
}

/// Log-domain normalization in place so that `sum(exp(a)) == 1` along `axis`.
///
/// Subtracts each lane's log-sum-exp. Mirrors `log_normalize(a, axis=axis)`,
/// including the degenerate single-element case (axis length 1) where the lane
/// is set to all-zeros rather than normalizing a lone `-inf`.
///
/// # Arguments
/// * `a` — array to normalize in place.
/// * `axis` — axis whose lanes are each shifted to have `sum(exp(·)) == 1`.
pub fn log_normalize_axis(a: &mut Array2<f64>, axis: usize) {
    let ax = Axis(axis);
    if a.len_of(ax) == 1 {
        a.fill(0.0);
        return;
    }
    for mut lane in a.lanes_mut(ax) {
        let lse = logsumexp(lane.view());
        lane.mapv_inplace(|x| x - lse);
    }
}

/// Translate an optional `lengths` array into `[start, end)` index ranges over a
/// concatenated observation matrix of `n_samples` rows.
///
/// `None` yields a single sequence spanning all rows. Mirrors `split_X_lengths`.
///
/// # Arguments
/// * `n_samples` — total number of rows in the concatenated observation matrix.
/// * `lengths` — per-sequence row counts; `None` treats all rows as one sequence.
///
/// # Returns
/// A vector of `(start, end)` half-open row ranges, one per sequence.
///
/// # Errors
/// [`HmmError::LengthsMismatch`] if the supplied lengths do not sum to
/// `n_samples`.
pub fn split_lengths(n_samples: usize, lengths: Option<&[usize]>) -> Result<Vec<(usize, usize)>> {
    match lengths {
        None => Ok(vec![(0, n_samples)]),
        Some(ls) => {
            let total: usize = ls.iter().sum();
            if total != n_samples {
                return Err(HmmError::LengthsMismatch { total, n_samples });
            }
            let mut ranges = Vec::with_capacity(ls.len());
            let mut start = 0;
            for &l in ls {
                ranges.push((start, start + l));
                start += l;
            }
            Ok(ranges)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::{array, Array2};

    #[test]
    fn normalize1_sums_to_one() {
        // Mirrors test_normalize: some entries zeroed, the rest positive.
        let mut a = array![3.0, 0.0, 1.0, 0.0, 6.0];
        normalize1(&mut a);
        assert_abs_diff_eq!(a.sum(), 1.0, epsilon = 1e-12);
        assert_eq!(a[1], 0.0);
    }

    #[test]
    fn normalize_axis_both_axes() {
        // Mirrors test_normalize_along_axis on a 2-D array.
        let mut a: Array2<f64> =
            Array2::from_shape_vec((3, 4), (1..=12).map(|x| x as f64).collect()).unwrap();
        let mut by_cols = a.clone();
        normalize_axis(&mut by_cols, 0);
        for col in by_cols.columns() {
            assert_abs_diff_eq!(col.sum(), 1.0, epsilon = 1e-12);
        }
        normalize_axis(&mut a, 1);
        for row in a.rows() {
            assert_abs_diff_eq!(row.sum(), 1.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn normalize_axis_preserves_zero_rows() {
        let mut a = array![[0.0, 0.0, 0.0], [1.0, 1.0, 2.0]];
        normalize_axis(&mut a, 1);
        assert_eq!(a.row(0).to_vec(), vec![0.0, 0.0, 0.0]);
        assert_abs_diff_eq!(a.row(1).sum(), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn log_normalize_axis_matches_exp_sum_one() {
        let mut a = array![[0.1_f64, 0.5, -0.2], [2.0, -1.0, 0.3]];
        log_normalize_axis(&mut a, 1);
        for row in a.rows() {
            let s: f64 = row.iter().map(|x| x.exp()).sum();
            assert_abs_diff_eq!(s, 1.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn log_normalize_axis_single_column_collapses() {
        let mut a = array![[f64::NEG_INFINITY], [1.0]];
        log_normalize_axis(&mut a, 1);
        assert_eq!(a, array![[0.0], [0.0]]);
    }

    #[test]
    fn split_lengths_none_is_single_sequence() {
        assert_eq!(split_lengths(10, None).unwrap(), vec![(0, 10)]);
    }

    #[test]
    fn split_lengths_partitions() {
        assert_eq!(
            split_lengths(10, Some(&[3, 5, 2])).unwrap(),
            vec![(0, 3), (3, 8), (8, 10)]
        );
    }

    #[test]
    fn split_lengths_mismatch_errors() {
        assert_eq!(
            split_lengths(10, Some(&[3, 5])),
            Err(HmmError::LengthsMismatch {
                total: 8,
                n_samples: 10
            })
        );
    }
}
