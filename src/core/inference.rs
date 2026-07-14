//! The inference axis: how start/transition parameters are initialized, used in
//! the E-step, updated in the M-step, and scored in the lower bound.
//!
//! `Em` implements classic Expectation–Maximization (the `BaseHMM` behavior).
//! The variational counterpart is added in a later phase; both implement
//! [`Inference`] so the generic [`super::hmm::Hmm`] core is agnostic to which is
//! in use.

use crate::core::params::{Param, ParamSet};
use crate::error::{HmmError, Result};
use crate::kl::kl_dirichlet;
use crate::rng::NumpyRandomState;
use crate::special::digamma;
use crate::util::{normalize1, normalize_axis};
use ndarray::{Array1, Array2};

/// Shared start/transition sufficient statistics (the base `nobs/start/trans`).
#[derive(Debug, Clone)]
pub struct CoreStats {
    /// Number of sequences accumulated so far.
    pub nobs: usize,
    /// Expected initial-state occupancy, length `n_components`.
    pub start: Array1<f64>,
    /// Expected transition counts, `(n_components, n_components)`.
    pub trans: Array2<f64>,
}

impl CoreStats {
    /// Zeroed start/transition statistics for `n_components` states.
    ///
    /// # Arguments
    /// * `n_components` — number of hidden states.
    ///
    /// # Returns
    /// A `CoreStats` with `nobs = 0` and zero-filled `start`/`trans`.
    pub fn zeros(n_components: usize) -> Self {
        CoreStats {
            nobs: 0,
            start: Array1::zeros(n_components),
            trans: Array2::zeros((n_components, n_components)),
        }
    }
}

/// Number of free scalars contributed by the start/transition parameters.
///
/// A distribution over `n_components` states has `n_components - 1` free scalars
/// (the last is fixed by the sum-to-one constraint); the transition matrix
/// contributes one such distribution per row.
///
/// # Arguments
/// * `n_components` — number of hidden states.
/// * `params` — which parameter groups are being fitted; only [`Param::Start`]
///   and [`Param::Trans`] contribute here.
///
/// # Returns
/// `n_components - 1` for a fitted start distribution plus
/// `n_components * (n_components - 1)` for a fitted transition matrix.
pub fn start_trans_fit_scalars(n_components: usize, params: ParamSet) -> usize {
    let mut n = 0;
    if params.contains(Param::Start) {
        n += n_components - 1;
    }
    if params.contains(Param::Trans) {
        n += n_components * (n_components - 1);
    }
    n
}

/// The start/transition inference strategy (EM or variational).
pub trait Inference: Clone {
    /// Number of hidden states.
    ///
    /// # Returns
    /// The state count this inference core was built for.
    fn n_components(&self) -> usize;

    /// Normalized initial-state distribution (for scoring/sampling/stationary).
    ///
    /// # Returns
    /// A reference to the length-`n_components` start distribution, summing to 1.
    fn start_prob(&self) -> &Array1<f64>;

    /// Normalized transition matrix.
    ///
    /// # Returns
    /// A reference to the row-stochastic `(n_components, n_components)` matrix.
    fn trans_mat(&self) -> &Array2<f64>;

    /// Initialize start/trans for the parameters selected by `init`.
    ///
    /// # Arguments
    /// * `init` — parameter groups to initialize; groups not yet set are always
    ///   initialized regardless of membership.
    /// * `seed` — RNG seed; matches hmmlearn's `check_random_state(self.random_state)`,
    ///   creating a fresh generator here, independent of the emission's initializer.
    /// * `n_sequences` — number of sequences, used by the variational initializer.
    /// * `n_samples` — total number of samples, used by the variational initializer.
    fn init(&mut self, init: ParamSet, seed: Option<u32>, n_sequences: usize, n_samples: usize);

    /// Per-iteration hook, called at the start of each E-step.
    ///
    /// The default is a no-op; the variational strategy overrides it to recompute
    /// the sub-normalized parameters from the current posteriors.
    fn estep_begin(&mut self) {}

    /// The `(start, trans)` pair used to drive forward/backward and score
    /// transition counts.
    ///
    /// # Returns
    /// `(start, trans)`: for EM the normalized parameters, for variational the
    /// sub-normalized ones. Both are owned copies.
    fn estep_start_trans(&self) -> (Array1<f64>, Array2<f64>);

    /// Update start/trans from the accumulated core statistics.
    ///
    /// # Arguments
    /// * `stats` — accumulated [`CoreStats`] from the E-step.
    /// * `params` — which of [`Param::Start`]/[`Param::Trans`] to update.
    fn mstep(&mut self, stats: &CoreStats, params: ParamSet);

    /// Objective value for this iteration, from the data log-probability.
    ///
    /// # Arguments
    /// * `curr_logprob` — summed sequence log-probabilities from the E-step.
    ///
    /// # Returns
    /// The iteration objective: `curr_logprob` itself for EM, or the variational
    /// lower bound (log-probability minus the parameter KL terms).
    fn lower_bound(&self, curr_logprob: f64) -> f64;

    /// Validate the start/transition parameters.
    ///
    /// # Errors
    /// [`HmmError::DimensionMismatch`] if a parameter array has the wrong shape,
    /// and (for EM) [`HmmError::InvalidParameter`] if the start distribution or a
    /// transition row does not sum to 1.
    fn check(&self) -> Result<()>;
}

/// Classic Expectation–Maximization inference for start/transition parameters.
#[derive(Debug, Clone)]
pub struct Em {
    /// Number of hidden states.
    n_components: usize,
    /// Initial-state distribution, length `n_components`.
    start_prob: Array1<f64>,
    /// Transition matrix, `(n_components, n_components)`.
    trans_mat: Array2<f64>,
    /// Whether `start_prob` is already set and must not be re-initialized.
    start_preset: bool,
    /// Whether `trans_mat` is already set and must not be re-initialized.
    trans_preset: bool,
    /// Dirichlet concentration prior on the initial-state distribution.
    start_prior: f64,
    /// Dirichlet concentration prior on each transition row.
    trans_prior: f64,
}

impl Em {
    /// A fresh EM core with the given priors and (optionally) preset parameters.
    ///
    /// # Arguments
    /// * `n_components` — number of hidden states.
    /// * `start_prior` — Dirichlet concentration prior on the start distribution.
    /// * `trans_prior` — Dirichlet concentration prior on each transition row.
    /// * `start_prob` — optional preset start distribution; `None` defers to
    ///   initialization.
    /// * `trans_mat` — optional preset transition matrix; `None` defers to
    ///   initialization.
    ///
    /// # Returns
    /// An `Em` core; presets are flagged so a later fit with empty `init_params`
    /// keeps them.
    pub fn new(
        n_components: usize,
        start_prior: f64,
        trans_prior: f64,
        start_prob: Option<Array1<f64>>,
        trans_mat: Option<Array2<f64>>,
    ) -> Self {
        let start_preset = start_prob.is_some();
        let trans_preset = trans_mat.is_some();
        Em {
            n_components,
            start_prob: start_prob.unwrap_or_else(|| Array1::zeros(n_components)),
            trans_mat: trans_mat.unwrap_or_else(|| Array2::zeros((n_components, n_components))),
            start_preset,
            trans_preset,
            start_prior,
            trans_prior,
        }
    }
}

impl Inference for Em {
    fn n_components(&self) -> usize {
        self.n_components
    }
    fn start_prob(&self) -> &Array1<f64> {
        &self.start_prob
    }
    fn trans_mat(&self) -> &Array2<f64> {
        &self.trans_mat
    }

    fn init(&mut self, init: ParamSet, seed: Option<u32>, _n_sequences: usize, _n_samples: usize) {
        let nc = self.n_components;
        let alpha = Array1::from_elem(nc, 1.0 / nc as f64);
        let mut rng = NumpyRandomState::new(seed.unwrap_or(0));
        if init.contains(Param::Start) || !self.start_preset {
            self.start_prob = rng.dirichlet(alpha.view());
        }
        if init.contains(Param::Trans) || !self.trans_preset {
            let mut tm = Array2::zeros((nc, nc));
            for i in 0..nc {
                tm.row_mut(i).assign(&rng.dirichlet(alpha.view()));
            }
            self.trans_mat = tm;
        }
        // After initialization the parameters are set; later fits with empty
        // `init_params` must not re-initialize them (hmmlearn's `_needs_init`).
        self.start_preset = true;
        self.trans_preset = true;
    }

    fn estep_start_trans(&self) -> (Array1<f64>, Array2<f64>) {
        (self.start_prob.clone(), self.trans_mat.clone())
    }

    /// Preserves structural zeros: any start entry or transition already exactly
    /// `0.0` stays zero (mirrors hmmlearn's `np.where`), so forbidden states or
    /// transitions in e.g. a left-to-right model are never revived.
    fn mstep(&mut self, stats: &CoreStats, params: ParamSet) {
        if params.contains(Param::Start) {
            let prior = self.start_prior;
            let mut sp = stats.start.mapv(|s| (prior - 1.0 + s).max(0.0));
            for i in 0..self.n_components {
                if self.start_prob[i] == 0.0 {
                    sp[i] = 0.0;
                }
            }
            normalize1(&mut sp);
            self.start_prob = sp;
        }
        if params.contains(Param::Trans) {
            let prior = self.trans_prior;
            let mut tm = stats.trans.mapv(|s| (prior - 1.0 + s).max(0.0));
            for i in 0..self.n_components {
                for j in 0..self.n_components {
                    if self.trans_mat[[i, j]] == 0.0 {
                        tm[[i, j]] = 0.0;
                    }
                }
            }
            normalize_axis(&mut tm, 1);
            self.trans_mat = tm;
        }
    }

    fn lower_bound(&self, curr_logprob: f64) -> f64 {
        curr_logprob
    }

    fn check(&self) -> Result<()> {
        let nc = self.n_components;
        if self.start_prob.len() != nc {
            return Err(HmmError::DimensionMismatch(
                "startprob_ must have length n_components".into(),
            ));
        }
        if (self.start_prob.sum() - 1.0).abs() > 1e-5 {
            return Err(HmmError::InvalidParameter(format!(
                "startprob_ must sum to 1 (got {:.4})",
                self.start_prob.sum()
            )));
        }
        if self.trans_mat.dim() != (nc, nc) {
            return Err(HmmError::DimensionMismatch(
                "transmat_ must have shape (n_components, n_components)".into(),
            ));
        }
        for (i, row) in self.trans_mat.rows().into_iter().enumerate() {
            if (row.sum() - 1.0).abs() > 1e-5 {
                return Err(HmmError::InvalidParameter(format!(
                    "transmat_ row {i} must sum to 1 (got {:.4})",
                    row.sum()
                )));
            }
        }
        Ok(())
    }
}

/// Variational-Bayes inference: the start/transition parameters are Dirichlet
/// distributions (prior + posterior), and the E-step uses sub-normalized
/// parameters derived from the posterior via `digamma`.
#[derive(Debug, Clone)]
pub struct Variational {
    /// Number of hidden states.
    n_components: usize,
    /// Dirichlet prior concentrations for the start distribution, length
    /// `n_components`.
    startprob_prior: Array1<f64>,
    /// Dirichlet posterior concentrations for the start distribution, length
    /// `n_components`.
    startprob_posterior: Array1<f64>,
    /// Dirichlet prior concentrations for each transition row,
    /// `(n_components, n_components)`.
    transmat_prior: Array2<f64>,
    /// Dirichlet posterior concentrations for each transition row,
    /// `(n_components, n_components)`.
    transmat_posterior: Array2<f64>,
    /// Normalized posterior-mean start distribution, length `n_components`.
    startprob: Array1<f64>,
    /// Normalized posterior-mean transition matrix,
    /// `(n_components, n_components)`.
    transmat: Array2<f64>,
    /// Sub-normalized start distribution (from `digamma`), used in the E-step.
    startprob_subnorm: Array1<f64>,
    /// Sub-normalized transition matrix (from `digamma`), used in the E-step.
    transmat_subnorm: Array2<f64>,
    /// Scalar Dirichlet prior for the start distribution; `None` uses the
    /// uniform `1/n_components` default.
    startprob_prior_scalar: Option<f64>,
    /// Scalar Dirichlet prior for the transition rows; `None` uses the uniform
    /// `1/n_components` default.
    transmat_prior_scalar: Option<f64>,
    /// Whether the start posterior is already set and must not be re-initialized.
    start_preset: bool,
    /// Whether the transition posterior is already set and must not be
    /// re-initialized.
    trans_preset: bool,
}

impl Variational {
    /// A variational core with the given Dirichlet-prior scalars (or `None` for
    /// the uniform `1/n_components` default) and optional preset posteriors.
    ///
    /// # Arguments
    /// * `n_components` — number of hidden states.
    /// * `startprob_prior_scalar` — scalar prior for the start distribution;
    ///   `None` defaults to `1/n_components`.
    /// * `transmat_prior_scalar` — scalar prior for the transition rows; `None`
    ///   defaults to `1/n_components`.
    /// * `startprob_prior` — optional full start prior array.
    /// * `startprob_posterior` — optional preset start posterior.
    /// * `transmat_prior` — optional full transition prior matrix.
    /// * `transmat_posterior` — optional preset transition posterior.
    ///
    /// # Returns
    /// A `Variational` core; if either posterior is preset the normalized means
    /// are computed immediately.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        n_components: usize,
        startprob_prior_scalar: Option<f64>,
        transmat_prior_scalar: Option<f64>,
        startprob_prior: Option<Array1<f64>>,
        startprob_posterior: Option<Array1<f64>>,
        transmat_prior: Option<Array2<f64>>,
        transmat_posterior: Option<Array2<f64>>,
    ) -> Self {
        let start_preset = startprob_posterior.is_some();
        let trans_preset = transmat_posterior.is_some();
        let sp_post = startprob_posterior.unwrap_or_else(|| Array1::zeros(n_components));
        let tm_post =
            transmat_posterior.unwrap_or_else(|| Array2::zeros((n_components, n_components)));
        let mut v = Variational {
            n_components,
            startprob_prior: startprob_prior.unwrap_or_else(|| Array1::zeros(n_components)),
            startprob_posterior: sp_post,
            transmat_prior: transmat_prior
                .unwrap_or_else(|| Array2::zeros((n_components, n_components))),
            transmat_posterior: tm_post,
            startprob: Array1::zeros(n_components),
            transmat: Array2::zeros((n_components, n_components)),
            startprob_subnorm: Array1::zeros(n_components),
            transmat_subnorm: Array2::zeros((n_components, n_components)),
            startprob_prior_scalar,
            transmat_prior_scalar,
            start_preset,
            trans_preset,
        };
        if start_preset || trans_preset {
            v.update_normalized();
        }
        v
    }

    /// Recompute the normalized posterior means `startprob_`/`transmat_`.
    fn update_normalized(&mut self) {
        let s = self.startprob_posterior.sum();
        if s != 0.0 {
            self.startprob = self.startprob_posterior.mapv(|x| x / s);
        }
        self.transmat = self.transmat_posterior.clone();
        normalize_axis(&mut self.transmat, 1);
    }

    /// The Dirichlet posterior concentrations for the initial-state distribution.
    pub fn startprob_posterior(&self) -> &Array1<f64> {
        &self.startprob_posterior
    }
}

impl Inference for Variational {
    fn n_components(&self) -> usize {
        self.n_components
    }
    fn start_prob(&self) -> &Array1<f64> {
        &self.startprob
    }
    fn trans_mat(&self) -> &Array2<f64> {
        &self.transmat
    }

    /// Seeds the priors from the scalar priors (or the uniform default) and
    /// draws Dirichlet posteriors scaled by the data volume: the start posterior
    /// by `n_sequences`, each transition row by `n_samples / n_components`.
    fn init(&mut self, init: ParamSet, seed: Option<u32>, n_sequences: usize, n_samples: usize) {
        let nc = self.n_components;
        let uniform = 1.0 / nc as f64;
        let alpha = Array1::from_elem(nc, uniform);
        let mut rng = NumpyRandomState::new(seed.unwrap_or(0));
        if init.contains(Param::Start) || !self.start_preset {
            let sp_init = self.startprob_prior_scalar.unwrap_or(uniform);
            self.startprob_prior = Array1::from_elem(nc, sp_init);
            self.startprob_posterior = rng.dirichlet(alpha.view()) * n_sequences as f64;
        }
        if init.contains(Param::Trans) || !self.trans_preset {
            let tm_init = self.transmat_prior_scalar.unwrap_or(uniform);
            self.transmat_prior = Array2::from_elem((nc, nc), tm_init);
            let mut tm = Array2::zeros((nc, nc));
            for i in 0..nc {
                tm.row_mut(i).assign(&rng.dirichlet(alpha.view()));
            }
            self.transmat_posterior = tm * (n_samples as f64 / nc as f64);
        }
        self.start_preset = true;
        self.trans_preset = true;
        self.update_normalized();
    }

    /// Computes the sub-normalized parameters `exp(digamma(alpha) -
    /// digamma(sum alpha))` from the current Dirichlet posteriors — the
    /// geometric-mean expectations used in place of probabilities in the E-step.
    fn estep_begin(&mut self) {
        let dg_ssum = digamma(self.startprob_posterior.sum());
        self.startprob_subnorm = self
            .startprob_posterior
            .mapv(|x| (digamma(x) - dg_ssum).exp());
        let mut tsub = Array2::zeros((self.n_components, self.n_components));
        for i in 0..self.n_components {
            let dg_rsum = digamma(self.transmat_posterior.row(i).sum());
            for j in 0..self.n_components {
                tsub[[i, j]] = (digamma(self.transmat_posterior[[i, j]]) - dg_rsum).exp();
            }
        }
        self.transmat_subnorm = tsub;
    }

    fn estep_start_trans(&self) -> (Array1<f64>, Array2<f64>) {
        (
            self.startprob_subnorm.clone(),
            self.transmat_subnorm.clone(),
        )
    }

    fn mstep(&mut self, stats: &CoreStats, params: ParamSet) {
        if params.contains(Param::Start) {
            self.startprob_posterior = &self.startprob_prior + &stats.start;
        }
        if params.contains(Param::Trans) {
            self.transmat_posterior = &self.transmat_prior + &stats.trans;
        }
        self.update_normalized();
    }

    /// Subtracts the KL divergence of each Dirichlet posterior from its prior
    /// (start distribution plus every transition row) from `curr_logprob`.
    fn lower_bound(&self, curr_logprob: f64) -> f64 {
        let mut lb = curr_logprob
            - kl_dirichlet(self.startprob_posterior.view(), self.startprob_prior.view());
        for i in 0..self.n_components {
            lb -= kl_dirichlet(self.transmat_posterior.row(i), self.transmat_prior.row(i));
        }
        lb
    }

    fn check(&self) -> Result<()> {
        let nc = self.n_components;
        if self.startprob_prior.len() != nc || self.startprob_posterior.len() != nc {
            return Err(HmmError::DimensionMismatch(
                "startprob prior/posterior must have length n_components".into(),
            ));
        }
        if self.transmat_prior.dim() != (nc, nc) || self.transmat_posterior.dim() != (nc, nc) {
            return Err(HmmError::DimensionMismatch(
                "transmat prior/posterior must have shape (n_components, n_components)".into(),
            ));
        }
        Ok(())
    }
}
