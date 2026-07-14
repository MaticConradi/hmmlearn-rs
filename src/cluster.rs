//! A small k-means++ used to initialize Gaussian means.
//!
//! hmmlearn initializes Gaussian/GMM means with scikit-learn's `KMeans`. Its
//! exact RNG-call sequence is version-sensitive and cannot be reproduced in pure
//! Rust, so this is a clean k-means++ seeded by the NumPy-compatible RNG. Fits
//! that rely on it are validated by log-likelihood monotonicity and
//! parameter-recovery-within-tolerance, both robust to the initialization.

use crate::rng::NumpyRandomState;
use ndarray::{Array1, Array2, ArrayView2};

/// Squared Euclidean distance between row `i` of `a` and center vector `c`.
///
/// # Arguments
/// * `a` — the `(n_samples, n_features)` data matrix.
/// * `i` — row index into `a`.
/// * `c` — a center vector; its length sets how many features are compared.
///
/// # Returns
/// The sum of squared per-feature differences.
fn sq_dist(a: ArrayView2<f64>, i: usize, c: &Array1<f64>) -> f64 {
    let mut d = 0.0;
    for j in 0..c.len() {
        let diff = a[[i, j]] - c[j];
        d += diff * diff;
    }
    d
}

/// k-means++ seeding followed by Lloyd iterations, with `n_init` restarts.
///
/// # Arguments
/// * `x` — the `(n_samples, n_features)` data matrix.
/// * `k` — number of clusters.
/// * `n_init` — number of independent restarts (at least one is always run).
/// * `max_iter` — maximum Lloyd iterations per restart.
/// * `seed` — RNG seed for reproducible seeding (defaults to 0 when `None`).
///
/// # Returns
/// The `(k, n_features)` centers of the restart with the lowest total
/// within-cluster squared distance (inertia).
pub fn kmeans(
    x: ArrayView2<f64>,
    k: usize,
    n_init: usize,
    max_iter: usize,
    seed: Option<u32>,
) -> Array2<f64> {
    let n = x.nrows();
    let nf = x.ncols();
    let mut rng = NumpyRandomState::new(seed.unwrap_or(0));
    let mut best_centers = Array2::zeros((k, nf));
    let mut best_inertia = f64::INFINITY;

    for _ in 0..n_init.max(1) {
        let mut centers = kmeans_plusplus(x, k, &mut rng);
        let mut labels = vec![0usize; n];
        for _ in 0..max_iter {
            let changed = assign(x, &centers, &mut labels);
            recompute(x, &labels, k, &mut centers);
            if !changed {
                break;
            }
        }
        let inertia = (0..n)
            .map(|i| sq_dist(x, i, &centers.row(labels[i]).to_owned()))
            .sum::<f64>();
        if inertia < best_inertia {
            best_inertia = inertia;
            best_centers = centers;
        }
    }
    best_centers
}

/// Number of points nearest to each center, like `np.bincount(predict(X))`
/// padded to `k` bins.
///
/// # Arguments
/// * `x` — the `(n_samples, n_features)` data matrix.
/// * `centers` — the `(k, n_features)` cluster centers.
///
/// # Returns
/// A length-`k` array; entry `c` counts the rows of `x` whose nearest center is
/// `c` (ties broken toward the lowest index).
pub fn cluster_counts(x: ArrayView2<f64>, centers: ArrayView2<f64>) -> Array1<f64> {
    let k = centers.nrows();
    let mut counts = Array1::zeros(k);
    for i in 0..x.nrows() {
        let mut best = 0;
        let mut best_d = f64::INFINITY;
        for c in 0..k {
            let d = sq_dist(x, i, &centers.row(c).to_owned());
            if d < best_d {
                best_d = d;
                best = c;
            }
        }
        counts[best] += 1.0;
    }
    counts
}

/// k-means++ initial center selection.
///
/// Picks the first center uniformly at random, then each subsequent center by
/// sampling a point with probability proportional to its squared distance (D²)
/// to the nearest already-chosen center.
///
/// # Arguments
/// * `x` — the `(n_samples, n_features)` data matrix.
/// * `k` — number of centers to pick.
/// * `rng` — NumPy-compatible RNG driving the random draws.
///
/// # Returns
/// The `(k, n_features)` seeded centers.
fn kmeans_plusplus(x: ArrayView2<f64>, k: usize, rng: &mut NumpyRandomState) -> Array2<f64> {
    let n = x.nrows();
    let nf = x.ncols();
    let mut centers = Array2::zeros((k, nf));
    // First center: uniformly at random.
    let first = (rng.random_sample() * n as f64) as usize % n;
    centers.row_mut(0).assign(&x.row(first));
    let mut d2 = vec![f64::INFINITY; n];
    for c in 1..k {
        let prev = centers.row(c - 1).to_owned();
        let mut total = 0.0;
        for (i, d) in d2.iter_mut().enumerate() {
            let dist = sq_dist(x, i, &prev);
            if dist < *d {
                *d = dist;
            }
            total += *d;
        }
        // Sample proportionally to D².
        let target = rng.random_sample() * total;
        let mut acc = 0.0;
        let mut chosen = n - 1;
        for (i, &d) in d2.iter().enumerate() {
            acc += d;
            if acc >= target {
                chosen = i;
                break;
            }
        }
        centers.row_mut(c).assign(&x.row(chosen));
    }
    centers
}

/// Assign each point to its nearest center, updating `labels` in place.
///
/// # Arguments
/// * `x` — the `(n_samples, n_features)` data matrix.
/// * `centers` — current `(k, n_features)` centers.
/// * `labels` — per-point cluster labels, overwritten with the nearest center
///   index (ties broken toward the lowest index).
///
/// # Returns
/// `true` if any label changed, `false` if the assignment is already stable.
fn assign(x: ArrayView2<f64>, centers: &Array2<f64>, labels: &mut [usize]) -> bool {
    let mut changed = false;
    for (i, label) in labels.iter_mut().enumerate() {
        let mut best = 0;
        let mut best_d = f64::INFINITY;
        for c in 0..centers.nrows() {
            let d = sq_dist(x, i, &centers.row(c).to_owned());
            if d < best_d {
                best_d = d;
                best = c;
            }
        }
        if *label != best {
            *label = best;
            changed = true;
        }
    }
    changed
}

/// Recompute each center as the mean of its assigned points, in place.
///
/// Empty clusters keep their previous center.
///
/// # Arguments
/// * `x` — the `(n_samples, n_features)` data matrix.
/// * `labels` — per-point cluster labels.
/// * `k` — number of clusters.
/// * `centers` — `(k, n_features)` centers, overwritten with the cluster means.
fn recompute(x: ArrayView2<f64>, labels: &[usize], k: usize, centers: &mut Array2<f64>) {
    let nf = x.ncols();
    let mut counts = vec![0usize; k];
    let mut sums = Array2::<f64>::zeros((k, nf));
    for (i, &lab) in labels.iter().enumerate() {
        counts[lab] += 1;
        for j in 0..nf {
            sums[[lab, j]] += x[[i, j]];
        }
    }
    for c in 0..k {
        if counts[c] > 0 {
            for j in 0..nf {
                centers[[c, j]] = sums[[c, j]] / counts[c] as f64;
            }
        }
    }
}
