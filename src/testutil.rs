//! Shared assertions for unit tests (compiled only under `cfg(test)`).

use crate::core::emission::EmissionModel;
use crate::core::hmm::Hmm;
use crate::core::params::ParamSet;
use ndarray::{Array, ArrayView2, Dimension};

/// Assert two arrays have the same shape and differ by at most `eps` elementwise.
pub fn assert_close<D: Dimension>(a: &Array<f64, D>, b: &Array<f64, D>, eps: f64) {
    assert_eq!(
        a.shape(),
        b.shape(),
        "shape mismatch: {:?} vs {:?}",
        a.shape(),
        b.shape()
    );
    let max_diff = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_diff <= eps,
        "arrays differ by {max_diff} > {eps}\n left = {a:?}\n right = {b:?}"
    );
}

/// Port of `assert_log_likelihood_increasing`: fit one EM iteration at a time
/// (without re-initializing parameters) and require the data log-likelihood to
/// make at least one improvement larger than `sqrt(eps)`.
pub fn assert_ll_increasing<E: EmissionModel>(
    mut model: Hmm<E>,
    x: ArrayView2<f64>,
    lengths: Option<&[usize]>,
    n_iter: usize,
) {
    model.n_iter = 1;
    model.init_params = ParamSet::empty();
    let mut lls = Vec::with_capacity(n_iter);
    let mut fitted = model.fit(x, lengths).unwrap();
    lls.push(fitted.score(x, lengths).unwrap());
    for _ in 1..n_iter {
        fitted = fitted.into_inner().fit(x, lengths).unwrap();
        lls.push(fitted.score(x, lengths).unwrap());
    }
    let eps_sqrt = f64::EPSILON.sqrt();
    let diff_max = lls
        .windows(2)
        .map(|w| w[1] - w[0])
        .fold(f64::NEG_INFINITY, f64::max);
    assert!(
        diff_max > eps_sqrt,
        "non-increasing log-likelihoods: {lls:?}"
    );
}
