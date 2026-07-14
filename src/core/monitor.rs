//! EM convergence tracking — port of `hmmlearn.base.ConvergenceMonitor`.

/// Tracks per-iteration log-probabilities and decides when EM has converged.
#[derive(Debug, Clone)]
pub struct ConvergenceMonitor {
    /// Convergence threshold on the log-probability improvement.
    pub tol: f64,
    /// Maximum number of iterations.
    pub n_iter: usize,
    /// Whether to print per-iteration reports to stderr.
    pub verbose: bool,
    /// Log-probability of every iteration so far.
    pub history: Vec<f64>,
    /// Number of iterations performed.
    pub iter: usize,
}

impl ConvergenceMonitor {
    /// Create a monitor with the given tolerance, iteration cap, and verbosity.
    ///
    /// # Arguments
    /// * `tol` — convergence threshold on the log-probability improvement.
    /// * `n_iter` — maximum number of iterations.
    /// * `verbose` — whether to print per-iteration reports to stderr.
    ///
    /// # Returns
    /// A monitor with an empty history and iteration count zero.
    pub fn new(tol: f64, n_iter: usize, verbose: bool) -> Self {
        ConvergenceMonitor {
            tol,
            n_iter,
            verbose,
            history: Vec::new(),
            iter: 0,
        }
    }

    /// Reset iteration count and history.
    pub fn reset(&mut self) {
        self.iter = 0;
        self.history.clear();
    }

    /// Format one report line, matching hmmlearn's `_template`.
    ///
    /// # Arguments
    /// * `iter` — 1-based iteration number.
    /// * `log_prob` — log-probability at this iteration.
    /// * `delta` — improvement over the previous iteration; a NaN is rendered as
    ///   the right-aligned literal `+nan`.
    ///
    /// # Returns
    /// The formatted line: iteration, log-probability, and signed delta, each
    /// right-aligned to hmmlearn's column widths.
    fn format_report(iter: usize, log_prob: f64, delta: f64) -> String {
        let delta_field = if delta.is_nan() {
            format!("{:>16}", "+nan")
        } else {
            format!("{:>+16.8}", delta)
        };
        format!("{iter:>10} {log_prob:>16.8} {delta_field}")
    }

    /// Record an iteration's log-probability, printing a report if verbose and
    /// warning on a non-trivial decrease.
    ///
    /// Appends `log_prob` to `history` and increments `iter`. A decrease larger
    /// than `sqrt(f64::EPSILON)` relative to the previous entry emits a
    /// "not converging" warning to stderr.
    ///
    /// # Arguments
    /// * `log_prob` — the log-probability (EM) or lower bound (variational) for
    ///   the current iteration.
    pub fn report(&mut self, log_prob: f64) {
        if self.verbose {
            let delta = self.history.last().map_or(f64::NAN, |&h| log_prob - h);
            eprintln!("{}", Self::format_report(self.iter + 1, log_prob, delta));
        }
        let precision = f64::EPSILON.sqrt();
        if let Some(&last) = self.history.last() {
            if log_prob - last < -precision {
                eprintln!(
                    "Model is not converging.  Current: {log_prob} is not greater than {last}. \
                     Delta is {}",
                    log_prob - last
                );
            }
        }
        self.history.push(log_prob);
        self.iter += 1;
    }

    /// Whether EM has converged: the iteration cap is reached, or the last two
    /// log-probabilities improved by less than `tol`.
    pub fn converged(&self) -> bool {
        self.iter == self.n_iter
            || (self.history.len() >= 2
                && self.history[self.history.len() - 1] - self.history[self.history.len() - 2]
                    < self.tol)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converged_by_iterations() {
        let mut m = ConvergenceMonitor::new(1e-3, 2, false);
        assert!(!m.converged());
        m.report(-0.01);
        assert!(!m.converged());
        m.report(-0.1);
        assert!(m.converged());
    }

    #[test]
    fn converged_by_log_prob() {
        let mut m = ConvergenceMonitor::new(1e-3, 10, false);
        for &lp in &[-0.03, -0.02, -0.01] {
            m.report(lp);
            assert!(!m.converged());
        }
        m.report(-0.0101);
        assert!(m.converged());
    }

    #[test]
    fn reset_clears_state() {
        let mut m = ConvergenceMonitor::new(1e-3, 10, false);
        m.iter = 1;
        m.history.push(-0.01);
        m.reset();
        assert_eq!(m.iter, 0);
        assert!(m.history.is_empty());
    }

    #[test]
    fn report_builds_history() {
        let mut m = ConvergenceMonitor::new(1e-3, 10, false);
        for i in (0..10).rev() {
            m.report(-0.01 * i as f64);
        }
        assert_eq!(m.history.len(), 10);
        assert_eq!(m.iter, 10);
    }

    #[test]
    fn report_format_matches_hmmlearn() {
        assert_eq!(
            ConvergenceMonitor::format_report(1, -0.01, f64::NAN),
            "         1      -0.01000000             +nan"
        );
        assert_eq!(
            ConvergenceMonitor::format_report(2, -0.1, -0.09),
            "         2      -0.10000000      -0.09000000"
        );
        assert_eq!(
            ConvergenceMonitor::format_report(3, 1.5, 0.25),
            "         3       1.50000000      +0.25000000"
        );
    }
}
