//! The emission axis: everything specific to how each model produces
//! observations, expressed as a trait so the generic [`super::hmm::Hmm`] core can
//! drive any emission type.
//!
//! Each emission fixes an [`Inference`] strategy
//! (EM or variational) via the associated `Inference` type, and owns a concrete
//! sufficient-statistics type — the typed replacement for hmmlearn's `stats`
//! dict.

use crate::core::inference::Inference;
use crate::core::params::{Param, ParamSet};
use crate::error::Result;
use crate::rng::NumpyRandomState;
use ndarray::{Array1, Array2, ArrayView2};

/// An emission distribution family (categorical, Gaussian, Poisson, …).
pub trait EmissionModel: Clone {
    /// The start/transition inference strategy this emission is wired to.
    type Inference: Inference;
    /// Emission-specific sufficient statistics accumulated during the E-step.
    type Stats;

    /// The parameter groups this emission owns, beyond start/trans.
    ///
    /// # Returns
    /// A static slice of the emission-specific [`Param`] codes (e.g. means and
    /// covariances for a Gaussian model).
    fn emission_params() -> &'static [Param];

    /// Number of feature columns expected in `X`.
    ///
    /// # Returns
    /// The feature dimension of observations this model consumes.
    fn n_features(&self) -> usize;

    /// Validate `X` and infer/confirm the feature dimension.
    ///
    /// # Arguments
    /// * `x` — observation matrix, `(n_samples, n_features)`.
    ///
    /// # Errors
    /// An error if `x`'s column count disagrees with a feature dimension already
    /// fixed on this model.
    fn check_and_set_n_features(&mut self, x: ArrayView2<f64>) -> Result<()>;

    /// Initialize emission parameters for the groups selected by `init`.
    ///
    /// # Arguments
    /// * `x` — observation matrix, `(n_samples, n_features)`, used to seed
    ///   parameter estimates.
    /// * `init` — which emission parameter groups to initialize.
    /// * `seed` — RNG seed; matches hmmlearn's per-`_init` `check_random_state`,
    ///   seeding the emission's own generator independently of the start/trans
    ///   initializer.
    ///
    /// # Errors
    /// An error if initialization fails (e.g. an inconsistent or degenerate `x`).
    fn init(&mut self, x: ArrayView2<f64>, init: ParamSet, seed: Option<u32>) -> Result<()>;

    /// Validate emission parameters.
    ///
    /// # Arguments
    /// * `n_components` — number of hidden states the parameters must match.
    ///
    /// # Errors
    /// An error if an emission parameter array has the wrong shape or holds
    /// invalid values.
    fn check(&self, n_components: usize) -> Result<()>;

    /// Free scalars contributed by the emission parameters in `params`.
    ///
    /// # Arguments
    /// * `n_components` — number of hidden states.
    /// * `params` — which parameter groups are being fitted.
    ///
    /// # Returns
    /// The count of independently fitted emission scalars, for AIC/BIC and the
    /// degenerate-fit warning.
    fn n_fit_scalars(&self, n_components: usize, params: ParamSet) -> usize;

    /// Per-component emission log-likelihood used for scoring.
    ///
    /// # Arguments
    /// * `x` — observation matrix, `(n_samples, n_features)`.
    ///
    /// # Returns
    /// The `(n_samples, n_components)` log-likelihood `log p(x_t | state)`.
    fn log_likelihood(&self, x: ArrayView2<f64>) -> Array2<f64>;

    /// Per-component emission likelihood for the scaling path.
    ///
    /// Defaults to `exp(log_likelihood)`; count models override for fidelity to
    /// scipy.
    ///
    /// # Arguments
    /// * `x` — observation matrix, `(n_samples, n_features)`.
    ///
    /// # Returns
    /// The `(n_samples, n_components)` likelihood `p(x_t | state)`.
    fn likelihood(&self, x: ArrayView2<f64>) -> Array2<f64> {
        self.log_likelihood(x).mapv(f64::exp)
    }

    /// Emission log-likelihood used inside the E-step.
    ///
    /// Same as [`log_likelihood`](Self::log_likelihood) for EM; variational
    /// models override this to return the *sub-normalized* log-likelihood.
    ///
    /// # Arguments
    /// * `x` — observation matrix, `(n_samples, n_features)`.
    ///
    /// # Returns
    /// The `(n_samples, n_components)` log-likelihood to feed forward/backward
    /// during fitting.
    fn fit_log_likelihood(&self, x: ArrayView2<f64>) -> Array2<f64> {
        self.log_likelihood(x)
    }

    /// Scaling-path counterpart of [`fit_log_likelihood`](Self::fit_log_likelihood).
    ///
    /// # Arguments
    /// * `x` — observation matrix, `(n_samples, n_features)`.
    ///
    /// # Returns
    /// The `(n_samples, n_components)` likelihood to feed the scaling
    /// forward/backward during fitting.
    fn fit_likelihood(&self, x: ArrayView2<f64>) -> Array2<f64> {
        self.likelihood(x)
    }

    /// Allocate zeroed emission sufficient statistics.
    ///
    /// # Returns
    /// A fresh `Self::Stats` ready to accumulate one E-step.
    fn init_stats(&self) -> Self::Stats;

    /// Accumulate emission statistics from one sequence's posteriors.
    ///
    /// # Arguments
    /// * `stats` — running sufficient statistics, updated in place.
    /// * `x` — one sequence's observations, `(n_samples, n_features)`.
    /// * `posteriors` — per-sample state posteriors, `(n_samples, n_components)`.
    /// * `params` — which emission parameter groups are being fitted.
    fn accumulate(
        &self,
        stats: &mut Self::Stats,
        x: ArrayView2<f64>,
        posteriors: ArrayView2<f64>,
        params: ParamSet,
    );

    /// Update emission parameters from accumulated statistics.
    ///
    /// # Arguments
    /// * `stats` — sufficient statistics accumulated over all sequences.
    /// * `params` — which emission parameter groups to update.
    ///
    /// # Errors
    /// An error if the update produces invalid parameters (e.g. a non-positive-
    /// definite covariance).
    fn mstep(&mut self, stats: &Self::Stats, params: ParamSet) -> Result<()>;

    /// Draw one observation (a row of features) from the given state.
    ///
    /// # Arguments
    /// * `state` — hidden-state index to condition on.
    /// * `rng` — NumPy-compatible generator supplying the draw.
    ///
    /// # Returns
    /// A length-`n_features` sample from the state's emission distribution.
    fn sample_state(&self, state: usize, rng: &mut NumpyRandomState) -> Array1<f64>;

    /// Emission contribution to the variational lower bound.
    ///
    /// # Returns
    /// `0.0` for EM emissions; variational emissions override with their KL term.
    fn lower_bound_contribution(&self) -> f64 {
        0.0
    }

    /// Per-iteration hook, called at the start of each E-step.
    ///
    /// The default is a no-op; the variational categorical emission overrides it
    /// to cache its digamma terms.
    fn estep_begin(&mut self) {}
}
