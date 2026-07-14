//! Forward/backward, Viterbi, transition-count, and posterior recursions.
//!
//! Direct ports of hmmlearn's `ext/_hmmc.cpp`. The log-domain routines take
//! `startprob`/`transmat` as *probabilities* and log them internally (matching
//! the C++); callers pass the raw parameter arrays. Arrays are `(n_samples,
//! n_components)` for lattices and `(n_components, n_components)` for transitions.

use crate::error::{HmmError, Result};
use crate::special::{logaddexp, logsumexp};
use crate::util::{log_normalize_axis, normalize_axis};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

/// Forward recursion in log space.
///
/// Fills the forward lattice with `alpha[t, j] = log P(o_1..o_t, state_t = j)`
/// using `logsumexp` over the log-transition terms.
///
/// # Arguments
/// * `startprob` — initial-state probabilities (not logs), length `n_components`;
///   logged internally.
/// * `transmat` — row-stochastic transition probabilities (not logs),
///   `(n_components, n_components)`; logged internally.
/// * `log_frameprob` — per-frame emission log-likelihoods,
///   `(n_samples, n_components)`.
///
/// # Returns
/// `(log_prob, fwdlattice)`: the sequence log-likelihood (`logsumexp` of the
/// final lattice row) and the `(n_samples, n_components)` log-forward lattice.
pub fn forward_log(
    startprob: ArrayView1<f64>,
    transmat: ArrayView2<f64>,
    log_frameprob: ArrayView2<f64>,
) -> (f64, Array2<f64>) {
    let (ns, nc) = log_frameprob.dim();
    let log_startprob = startprob.mapv(f64::ln);
    let log_transmat = transmat.mapv(f64::ln);
    let mut fwd = Array2::<f64>::zeros((ns, nc));
    for i in 0..nc {
        fwd[[0, i]] = log_startprob[i] + log_frameprob[[0, i]];
    }
    let mut buf = Array1::<f64>::zeros(nc);
    for t in 1..ns {
        for j in 0..nc {
            for i in 0..nc {
                buf[i] = fwd[[t - 1, i]] + log_transmat[[i, j]];
            }
            fwd[[t, j]] = logsumexp(buf.view()) + log_frameprob[[t, j]];
        }
    }
    let log_prob = logsumexp(fwd.row(ns - 1));
    (log_prob, fwd)
}

/// Forward recursion with per-step scaling.
///
/// Each lattice row is rescaled to sum to 1; the reciprocal scale factors
/// accumulate the log-likelihood as `-sum(ln(scaling[t]))`.
///
/// # Arguments
/// * `startprob` — initial-state probabilities, length `n_components`.
/// * `transmat` — row-stochastic transition matrix, `(n_components, n_components)`.
/// * `frameprob` — per-frame emission likelihoods, `(n_samples, n_components)`.
///
/// # Returns
/// `(log_prob, fwdlattice, scaling)`: the sequence log-likelihood, the scaled
/// `(n_samples, n_components)` forward lattice, and the per-step scaling factors
/// (length `n_samples`).
///
/// # Errors
/// [`HmmError::ScalingUnderflow`] if a lattice row sums below `1e-300`.
pub fn forward_scaling(
    startprob: ArrayView1<f64>,
    transmat: ArrayView2<f64>,
    frameprob: ArrayView2<f64>,
) -> Result<(f64, Array2<f64>, Array1<f64>)> {
    const MIN_SUM: f64 = 1e-300;
    let (ns, nc) = frameprob.dim();
    let mut fwd = Array2::<f64>::zeros((ns, nc));
    let mut scaling = Array1::<f64>::zeros(ns);
    let mut log_prob = 0.0;

    for i in 0..nc {
        fwd[[0, i]] = startprob[i] * frameprob[[0, i]];
    }
    let sum: f64 = fwd.row(0).sum();
    if sum < MIN_SUM {
        return Err(HmmError::ScalingUnderflow);
    }
    let scale = 1.0 / sum;
    scaling[0] = scale;
    log_prob -= scale.ln();
    for i in 0..nc {
        fwd[[0, i]] *= scale;
    }

    for t in 1..ns {
        for j in 0..nc {
            let mut acc = 0.0;
            for i in 0..nc {
                acc += fwd[[t - 1, i]] * transmat[[i, j]];
            }
            fwd[[t, j]] = acc * frameprob[[t, j]];
        }
        let sum: f64 = fwd.row(t).sum();
        if sum < MIN_SUM {
            return Err(HmmError::ScalingUnderflow);
        }
        let scale = 1.0 / sum;
        scaling[t] = scale;
        log_prob -= scale.ln();
        for j in 0..nc {
            fwd[[t, j]] *= scale;
        }
    }
    Ok((log_prob, fwd, scaling))
}

/// Backward recursion in log space.
///
/// Fills `beta[t, i] = log P(o_{t+1}..o_T | state_t = i)`, with the final row
/// left at zero (`log 1`).
///
/// # Arguments
/// * `_startprob` — unused; accepted for signature parity with [`forward_log`].
/// * `transmat` — row-stochastic transition probabilities (not logs),
///   `(n_components, n_components)`; logged internally.
/// * `log_frameprob` — per-frame emission log-likelihoods,
///   `(n_samples, n_components)`.
///
/// # Returns
/// The `(n_samples, n_components)` log-backward lattice.
pub fn backward_log(
    _startprob: ArrayView1<f64>,
    transmat: ArrayView2<f64>,
    log_frameprob: ArrayView2<f64>,
) -> Array2<f64> {
    let (ns, nc) = log_frameprob.dim();
    let log_transmat = transmat.mapv(f64::ln);
    let mut bwd = Array2::<f64>::zeros((ns, nc));
    let mut buf = Array1::<f64>::zeros(nc);
    for t in (0..ns - 1).rev() {
        for i in 0..nc {
            for j in 0..nc {
                buf[j] = log_transmat[[i, j]] + log_frameprob[[t + 1, j]] + bwd[[t + 1, j]];
            }
            bwd[[t, i]] = logsumexp(buf.view());
        }
    }
    bwd
}

/// Backward recursion with scaling.
///
/// Applies the same per-step `scaling` factors produced by [`forward_scaling`],
/// with the final row initialized to `scaling[n_samples - 1]`.
///
/// # Arguments
/// * `_startprob` — unused; accepted for signature parity with [`forward_scaling`].
/// * `transmat` — row-stochastic transition matrix, `(n_components, n_components)`.
/// * `frameprob` — per-frame emission likelihoods, `(n_samples, n_components)`.
/// * `scaling` — per-step scaling factors from the forward pass, length
///   `n_samples`.
///
/// # Returns
/// The `(n_samples, n_components)` scaled backward lattice.
pub fn backward_scaling(
    _startprob: ArrayView1<f64>,
    transmat: ArrayView2<f64>,
    frameprob: ArrayView2<f64>,
    scaling: ArrayView1<f64>,
) -> Array2<f64> {
    let (ns, nc) = frameprob.dim();
    let mut bwd = Array2::<f64>::zeros((ns, nc));
    for i in 0..nc {
        bwd[[ns - 1, i]] = scaling[ns - 1];
    }
    for t in (0..ns - 1).rev() {
        for i in 0..nc {
            let mut acc = 0.0;
            for j in 0..nc {
                acc += transmat[[i, j]] * frameprob[[t + 1, j]] * bwd[[t + 1, j]];
            }
            bwd[[t, i]] = acc * scaling[t];
        }
    }
    bwd
}

/// Accumulated transition counts in scaling form.
///
/// Sums `xi[i, j] = sum_t fwd[t, i] * transmat[i, j] * frameprob[t+1, j] *
/// bwd[t+1, j]` over all adjacent frame pairs. The scaling normalization makes
/// the sequence log-likelihood divide out, so no explicit division is needed.
///
/// # Arguments
/// * `fwd` — scaled forward lattice, `(n_samples, n_components)`.
/// * `transmat` — transition matrix used in the E-step,
///   `(n_components, n_components)`.
/// * `bwd` — scaled backward lattice, `(n_samples, n_components)`.
/// * `frameprob` — per-frame emission likelihoods, `(n_samples, n_components)`.
///
/// # Returns
/// The `(n_components, n_components)` matrix of expected transition counts.
pub fn compute_scaling_xi_sum(
    fwd: ArrayView2<f64>,
    transmat: ArrayView2<f64>,
    bwd: ArrayView2<f64>,
    frameprob: ArrayView2<f64>,
) -> Array2<f64> {
    let (ns, nc) = frameprob.dim();
    let mut xi_sum = Array2::<f64>::zeros((nc, nc));
    for t in 0..ns - 1 {
        for i in 0..nc {
            for j in 0..nc {
                xi_sum[[i, j]] +=
                    fwd[[t, i]] * transmat[[i, j]] * frameprob[[t + 1, j]] * bwd[[t + 1, j]];
            }
        }
    }
    xi_sum
}

/// Accumulated transition counts in log form.
///
/// Accumulates `log_xi[i, j] = logaddexp_t (fwd[t, i] + log_transmat[i, j] +
/// log_frameprob[t+1, j] + bwd[t+1, j] - log_prob)` over adjacent frame pairs,
/// where `log_prob` is the sequence log-likelihood (`logsumexp` of the final
/// forward row). Entries start at `-inf` (no contribution).
///
/// # Arguments
/// * `fwd` — log-forward lattice, `(n_samples, n_components)`.
/// * `transmat` — transition probabilities (not logs) used in the E-step,
///   `(n_components, n_components)`; logged internally.
/// * `bwd` — log-backward lattice, `(n_samples, n_components)`.
/// * `log_frameprob` — per-frame emission log-likelihoods,
///   `(n_samples, n_components)`.
///
/// # Returns
/// The `(n_components, n_components)` matrix of log expected transition counts
/// (the caller exponentiates to obtain counts).
pub fn compute_log_xi_sum(
    fwd: ArrayView2<f64>,
    transmat: ArrayView2<f64>,
    bwd: ArrayView2<f64>,
    log_frameprob: ArrayView2<f64>,
) -> Array2<f64> {
    let (ns, nc) = log_frameprob.dim();
    let log_transmat = transmat.mapv(f64::ln);
    let log_prob = logsumexp(fwd.row(ns - 1));
    let mut log_xi_sum = Array2::<f64>::from_elem((nc, nc), f64::NEG_INFINITY);
    for t in 0..ns - 1 {
        for i in 0..nc {
            for j in 0..nc {
                let log_xi = fwd[[t, i]]
                    + log_transmat[[i, j]]
                    + log_frameprob[[t + 1, j]]
                    + bwd[[t + 1, j]]
                    - log_prob;
                log_xi_sum[[i, j]] = logaddexp(log_xi_sum[[i, j]], log_xi);
            }
        }
    }
    log_xi_sum
}

/// Viterbi decoding: the single most likely state sequence.
///
/// Runs the max-product recursion in log space, then backtraces. The final
/// state is the first argmax of the last lattice row (first maximum wins, like
/// `std::max_element`); during backtrace ties favor the larger state index,
/// matching the pair comparison in the C++ source.
///
/// # Arguments
/// * `startprob` — initial-state probabilities (not logs), length `n_components`;
///   logged internally.
/// * `transmat` — row-stochastic transition probabilities (not logs),
///   `(n_components, n_components)`; logged internally.
/// * `log_frameprob` — per-frame emission log-likelihoods,
///   `(n_samples, n_components)`.
///
/// # Returns
/// `(log_prob, state_sequence)`: the log-probability of the winning path and the
/// decoded state indices, length `n_samples`.
pub fn viterbi(
    startprob: ArrayView1<f64>,
    transmat: ArrayView2<f64>,
    log_frameprob: ArrayView2<f64>,
) -> (f64, Array1<usize>) {
    let (ns, nc) = log_frameprob.dim();
    let log_startprob = startprob.mapv(f64::ln);
    let log_transmat = transmat.mapv(f64::ln);
    let mut lattice = Array2::<f64>::zeros((ns, nc));
    for i in 0..nc {
        lattice[[0, i]] = log_startprob[i] + log_frameprob[[0, i]];
    }
    for t in 1..ns {
        for i in 0..nc {
            let mut max = f64::NEG_INFINITY;
            for j in 0..nc {
                let v = lattice[[t - 1, j]] + log_transmat[[j, i]];
                if v > max {
                    max = v;
                }
            }
            lattice[[t, i]] = max + log_frameprob[[t, i]];
        }
    }
    let mut state_sequence = Array1::<usize>::zeros(ns);
    // Last state: argmax of the final row (first max wins, like std::max_element).
    let mut prev = 0usize;
    let mut best = f64::NEG_INFINITY;
    for i in 0..nc {
        if lattice[[ns - 1, i]] > best {
            best = lattice[[ns - 1, i]];
            prev = i;
        }
    }
    state_sequence[ns - 1] = prev;
    // Backtrace: ties favor the larger index (pair comparison in the C++).
    for t in (0..ns - 1).rev() {
        let mut best_val = f64::NEG_INFINITY;
        let mut best_idx = 0usize;
        for i in 0..nc {
            let v = lattice[[t, i]] + log_transmat[[i, prev]];
            if v > best_val || (v == best_val && i > best_idx) {
                best_val = v;
                best_idx = i;
            }
        }
        prev = best_idx;
        state_sequence[t] = prev;
    }
    (lattice[[ns - 1, state_sequence[ns - 1]]], state_sequence)
}

/// Posteriors from log-domain lattices: `exp(normalize_rows(fwd + bwd))`.
///
/// Each row of `fwd + bwd` is log-normalized then exponentiated, so every output
/// row sums to 1.
///
/// # Arguments
/// * `fwd` — log-forward lattice, `(n_samples, n_components)`.
/// * `bwd` — log-backward lattice, `(n_samples, n_components)`.
///
/// # Returns
/// The `(n_samples, n_components)` state posteriors (row-stochastic).
pub fn compute_posteriors_log(fwd: ArrayView2<f64>, bwd: ArrayView2<f64>) -> Array2<f64> {
    let mut log_gamma = &fwd + &bwd;
    log_normalize_axis(&mut log_gamma, 1);
    log_gamma.mapv_inplace(f64::exp);
    log_gamma
}

/// Posteriors from scaling-form lattices: `normalize_rows(fwd * bwd)`.
///
/// Each row of the elementwise product is normalized to sum to 1.
///
/// # Arguments
/// * `fwd` — scaled forward lattice, `(n_samples, n_components)`.
/// * `bwd` — scaled backward lattice, `(n_samples, n_components)`.
///
/// # Returns
/// The `(n_samples, n_components)` state posteriors (row-stochastic).
pub fn compute_posteriors_scaling(fwd: ArrayView2<f64>, bwd: ArrayView2<f64>) -> Array2<f64> {
    let mut post = &fwd * &bwd;
    normalize_axis(&mut post, 1);
    post
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::assert_close;
    use ndarray::array;

    // The Wikipedia forward-backward example (test_base.py::TestBaseAgainstWikipedia).
    fn wikipedia() -> (Array1<f64>, Array2<f64>, Array2<f64>) {
        let startprob = array![0.5, 0.5];
        let transmat = array![[0.7, 0.3], [0.3, 0.7]];
        let frameprob = array![[0.9, 0.2], [0.9, 0.2], [0.1, 0.8], [0.9, 0.2], [0.9, 0.2]];
        (startprob, transmat, frameprob)
    }

    #[test]
    fn forward_log_matches_wikipedia() {
        let (sp, tm, fp) = wikipedia();
        let lfp = fp.mapv(f64::ln);
        let (log_prob, fwd) = forward_log(sp.view(), tm.view(), lfp.view());
        assert!((log_prob - (-3.3725)).abs() < 5e-5);
        // exp(unscaled log-forward) equals the reference alpha values.
        let ref_fwd = array![
            [0.4500, 0.1000],
            [0.3105, 0.0410],
            [0.0230, 0.0975],
            [0.0408, 0.0150],
            [0.0298, 0.0046]
        ];
        assert_close(&fwd.mapv(f64::exp), &ref_fwd, 1e-3);
    }

    #[test]
    fn forward_scaling_matches_wikipedia() {
        let (sp, tm, fp) = wikipedia();
        let (log_prob, fwd, _scaling) = forward_scaling(sp.view(), tm.view(), fp.view()).unwrap();
        assert!((log_prob - (-3.3725)).abs() < 5e-5);
        // Scaled rows sum to 1.
        for row in fwd.rows() {
            assert!((row.sum() - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn backward_log_matches_wikipedia() {
        let (sp, tm, fp) = wikipedia();
        let lfp = fp.mapv(f64::ln);
        let bwd = backward_log(sp.view(), tm.view(), lfp.view());
        let ref_bwd = array![
            [0.0661, 0.0455],
            [0.0906, 0.1503],
            [0.4593, 0.2437],
            [0.6900, 0.4100],
            [1.0000, 1.0000]
        ];
        assert_close(&bwd.mapv(f64::exp), &ref_bwd, 1e-3);
    }

    #[test]
    fn backward_scaling_reconstructs_wikipedia() {
        let (sp, tm, fp) = wikipedia();
        let (_lp, _fwd, scaling) = forward_scaling(sp.view(), tm.view(), fp.view()).unwrap();
        let bwd = backward_scaling(sp.view(), tm.view(), fp.view(), scaling.view());
        // cumprod(scaling[::-1])[::-1]
        let ns = scaling.len();
        let mut rev_cumprod = vec![0.0; ns];
        let mut acc = 1.0;
        for t in (0..ns).rev() {
            acc *= scaling[t];
            rev_cumprod[t] = acc;
        }
        let mut reconstructed = bwd.clone();
        for t in 0..ns {
            for c in 0..2 {
                reconstructed[[t, c]] /= rev_cumprod[t];
            }
        }
        let ref_bwd = array![
            [0.0661, 0.0455],
            [0.0906, 0.1503],
            [0.4593, 0.2437],
            [0.6900, 0.4100],
            [1.0000, 1.0000]
        ];
        assert_close(&reconstructed, &ref_bwd, 1e-3);
    }

    #[test]
    fn viterbi_matches_wikipedia() {
        let (sp, tm, fp) = wikipedia();
        let lfp = fp.mapv(f64::ln);
        let (log_prob, seq) = viterbi(sp.view(), tm.view(), lfp.view());
        assert_eq!(seq.to_vec(), vec![0, 0, 1, 0, 0]);
        assert!((log_prob - (-4.4590)).abs() < 5e-5);
    }

    #[test]
    fn score_samples_posteriors_match_wikipedia() {
        let (sp, tm, fp) = wikipedia();
        let lfp = fp.mapv(f64::ln);
        let (log_prob, fwd) = forward_log(sp.view(), tm.view(), lfp.view());
        let bwd = backward_log(sp.view(), tm.view(), lfp.view());
        let post = compute_posteriors_log(fwd.view(), bwd.view());
        assert!((log_prob - (-3.3725)).abs() < 5e-5);
        for row in post.rows() {
            assert!((row.sum() - 1.0).abs() < 1e-12);
        }
        let ref_post = array![
            [0.8673, 0.1327],
            [0.8204, 0.1796],
            [0.3075, 0.6925],
            [0.8204, 0.1796],
            [0.8673, 0.1327]
        ];
        assert_close(&post, &ref_post, 1e-4);
    }

    #[test]
    fn uniform_transmat_reduces_to_softmax_posteriors() {
        // test_base.py::TestBaseConsistentWithGMM: uniform start/trans => posteriors
        // are the row-softmax of the frame log-likelihoods.
        let nc = 4;
        let sp = Array1::from_elem(nc, 1.0 / nc as f64);
        let tm = Array2::from_elem((nc, nc), 1.0 / nc as f64);
        let lfp = array![
            [-0.2, -1.0, -0.5, -2.0],
            [-1.5, -0.3, -0.7, -0.1],
            [-0.9, -0.9, -0.2, -1.1]
        ];
        let (_lp, fwd) = forward_log(sp.view(), tm.view(), lfp.view());
        let bwd = backward_log(sp.view(), tm.view(), lfp.view());
        let post = compute_posteriors_log(fwd.view(), bwd.view());
        // expected: softmax of each row of lfp
        let mut expected = lfp.clone();
        log_normalize_axis(&mut expected, 1);
        expected.mapv_inplace(f64::exp);
        assert_close(&post, &expected, 1e-12);
    }
}
