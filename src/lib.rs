//! `hmmlearn` — a native Rust port of the Python library `hmmlearn`.
//!
//! Hidden Markov Models with categorical, Gaussian, Gaussian-mixture, Poisson,
//! and multinomial emissions, trained by Expectation–Maximization or variational
//! inference.
//!
//! Fitting proceeds by Expectation–Maximization (maximum-likelihood) or
//! mean-field variational inference. The concrete models and their builders live
//! in [`models`], backed by the forward–backward and Viterbi routines in
//! [`core`], the covariance representations in [`covariance`], and the
//! supporting numerics (special functions, linear algebra, statistics, and array
//! utilities) in the remaining modules.
//!
//! # Quick start
//!
//! Every model is built with a fluent builder, fit with
//! [`fit`](models::GaussianHmm::fit), and returns a [`Fitted`](core::Fitted)
//! handle exposing `predict`, `score`, `sample`, `decode`, `aic`/`bic`, and the
//! learned parameters.
//!
//! ```
//! use hmmlearn::models::GaussianHmm;
//! use hmmlearn::ndarray::array;
//!
//! // One feature per column; each row is an observation.
//! let x = array![[0.0], [0.2], [-0.1], [5.0], [5.2], [4.8]];
//!
//! let model = GaussianHmm::new(2)
//!     .n_iter(20)
//!     .random_state(42)
//!     .fit(x.view(), None)?;
//!
//! let states = model.predict(x.view(), None)?;
//! assert_eq!(states.len(), x.nrows());
//!
//! let log_likelihood = model.score(x.view(), None)?;
//! assert!(log_likelihood.is_finite());
//! # Ok::<(), hmmlearn::HmmError>(())
//! ```
//!
//! Multiple independent sequences are concatenated along the rows and delimited
//! by a `lengths` slice, exactly as in `hmmlearn`:
//! `model.fit(x.view(), Some(&[100, 200, 150]))`.
//!
//! # Relationship to `hmmlearn`
//!
//! This is a faithful, pure-Rust port of
//! [`hmmlearn`](https://github.com/hmmlearn/hmmlearn) 0.3.3 with no LAPACK/BLAS
//! dependency. It deviates from upstream in three documented, test-validated
//! ways, all confined to initialization and sampling: k-means++ replaces
//! scikit-learn's `KMeans` for mean initialization; multivariate-normal sampling
//! uses a Cholesky factor rather than NumPy's SVD; and the binomial sampler omits
//! NumPy's BTPE branch in favor of the inversion fallback. Fitted parameters and
//! inference results match upstream.

pub use ndarray;

pub mod cluster;
pub mod core;
pub mod covariance;
pub mod error;
pub mod kl;
pub mod linalg;
pub mod models;
pub mod rng;
pub mod special;
pub mod stats;
pub mod util;

#[cfg(test)]
pub(crate) mod testutil;

pub use error::{HmmError, Result};
