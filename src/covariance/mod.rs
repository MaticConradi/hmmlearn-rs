//! Covariance parameterization: the type tag, compressed storage, and validation.

pub mod store;
pub mod types;
pub mod validate;

pub use store::{distribute_covar, CovarStore};
pub use types::CovarianceType;
pub use validate::validate_covars;
