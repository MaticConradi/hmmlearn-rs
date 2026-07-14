//! Error type shared across the crate.

use std::error::Error;
use std::fmt;

/// Errors produced while building, fitting, or using a model.
#[derive(Debug, Clone, PartialEq)]
pub enum HmmError {
    /// An operation requiring fitted parameters was called on an unfitted model.
    NotFitted,
    /// `lengths` did not sum to the number of provided samples.
    LengthsMismatch {
        /// Sum of the supplied `lengths` array.
        total: usize,
        /// Number of samples actually provided.
        n_samples: usize,
    },
    /// The `scaling` forward pass underflowed (a column summed below `1e-300`).
    ///
    /// Mirrors the `ValueError` raised by hmmlearn's C++ `forward_scaling`.
    ScalingUnderflow,
    /// A constructor or input parameter was invalid.
    InvalidParameter(String),
    /// Input array had the wrong shape or was inconsistent with the model.
    DimensionMismatch(String),
    /// A covariance / scale matrix was not symmetric positive-definite.
    NotPositiveDefinite(String),
}

impl fmt::Display for HmmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HmmError::NotFitted => write!(f, "model is not fitted"),
            HmmError::LengthsMismatch { total, n_samples } => write!(
                f,
                "lengths array sums to {total} but {n_samples} samples were provided"
            ),
            HmmError::ScalingUnderflow => write!(
                f,
                "forward pass failed; for sequences with negligible probability under \
                 the model consider the `log` implementation"
            ),
            HmmError::InvalidParameter(msg) => write!(f, "invalid parameter: {msg}"),
            HmmError::DimensionMismatch(msg) => write!(f, "dimension mismatch: {msg}"),
            HmmError::NotPositiveDefinite(msg) => {
                write!(f, "matrix is not symmetric positive-definite: {msg}")
            }
        }
    }
}

impl Error for HmmError {}

/// Convenience alias for results returned by this crate.
pub type Result<T> = std::result::Result<T, HmmError>;
