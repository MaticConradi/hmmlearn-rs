//! Variational-Bayes HMM models.

pub mod categorical;
pub mod gaussian;

pub use categorical::{VariationalCategoricalEm, VariationalCategoricalHmm};
pub use gaussian::{Dof, VariationalGaussianEm, VariationalGaussianHmm};
