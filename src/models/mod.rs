//! Concrete HMM models and their builders.

pub mod categorical;
pub mod gaussian;
pub mod gmm;
pub mod multinomial;
pub mod poisson;
pub mod variational;

pub use categorical::{CategoricalEm, CategoricalHmm};
pub use gaussian::{GaussianEm, GaussianHmm};
pub use gmm::{GmmCovars, GmmEm, GmmHmm};
pub use multinomial::{MultinomialEm, MultinomialHmm};
pub use poisson::{PoissonEm, PoissonHmm};
pub use variational::{
    Dof, VariationalCategoricalEm, VariationalCategoricalHmm, VariationalGaussianEm,
    VariationalGaussianHmm,
};
