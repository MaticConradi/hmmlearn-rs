//! Parameter-gating bitset and the algorithm/implementation enums.
//!
//! hmmlearn uses substring tests on a `params` / `init_params` string (e.g.
//! `"stmc"`) to decide which parameters to initialize and update. We model that
//! as a small typed bitset.

use crate::error::HmmError;
use std::str::FromStr;

/// A single fittable parameter group, identified by its hmmlearn character code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Param {
    /// `'s'` — initial state distribution.
    Start,
    /// `'t'` — transition matrix.
    Trans,
    /// `'m'` — Gaussian means.
    Means,
    /// `'c'` — Gaussian covariances.
    Covars,
    /// `'e'` — categorical / multinomial emission probabilities.
    Emit,
    /// `'l'` — Poisson rates.
    Lambdas,
    /// `'w'` — GMM mixture weights.
    Weights,
}

impl Param {
    /// The single-bit mask identifying this parameter within a [`ParamSet`].
    #[inline]
    fn bit(self) -> u8 {
        match self {
            Param::Start => 1 << 0,
            Param::Trans => 1 << 1,
            Param::Means => 1 << 2,
            Param::Covars => 1 << 3,
            Param::Emit => 1 << 4,
            Param::Lambdas => 1 << 5,
            Param::Weights => 1 << 6,
        }
    }

    /// The hmmlearn character code for this parameter.
    pub fn code(self) -> char {
        match self {
            Param::Start => 's',
            Param::Trans => 't',
            Param::Means => 'm',
            Param::Covars => 'c',
            Param::Emit => 'e',
            Param::Lambdas => 'l',
            Param::Weights => 'w',
        }
    }
}

/// The set of parameters a model should initialize or update.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParamSet(u8);

impl ParamSet {
    /// The empty set (initialize/update nothing).
    pub const fn empty() -> Self {
        ParamSet(0)
    }

    /// Build from an hmmlearn-style code string such as `"stmc"`.
    ///
    /// Recognized codes (`s t m c e l w`) set their bit; any other character is
    /// ignored, matching hmmlearn's `'x' in self.params` semantics.
    pub fn from_codes(s: &str) -> Self {
        let mut bits = 0u8;
        for ch in s.chars() {
            bits |= match ch {
                's' => Param::Start.bit(),
                't' => Param::Trans.bit(),
                'm' => Param::Means.bit(),
                'c' => Param::Covars.bit(),
                'e' => Param::Emit.bit(),
                'l' => Param::Lambdas.bit(),
                'w' => Param::Weights.bit(),
                _ => 0,
            };
        }
        ParamSet(bits)
    }

    /// Whether `p` is in the set.
    #[inline]
    pub fn contains(self, p: Param) -> bool {
        self.0 & p.bit() != 0
    }

    /// Add `p` to the set.
    pub fn insert(&mut self, p: Param) {
        self.0 |= p.bit();
    }

    /// Whether the set is empty.
    pub fn is_empty(self) -> bool {
        self.0 == 0
    }
}

/// The decoding algorithm used by `decode` / `predict`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecoderAlgorithm {
    /// Maximum-likelihood state sequence (Viterbi).
    Viterbi,
    /// Per-sample most likely state (maximum a posteriori).
    Map,
}

impl FromStr for DecoderAlgorithm {
    type Err = HmmError;
    /// Parses `"viterbi"` or `"map"`; any other string yields
    /// [`HmmError::InvalidParameter`].
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "viterbi" => Ok(DecoderAlgorithm::Viterbi),
            "map" => Ok(DecoderAlgorithm::Map),
            other => Err(HmmError::InvalidParameter(format!(
                "algorithm must be 'viterbi' or 'map', got {other:?}"
            ))),
        }
    }
}

/// Numerical strategy for the forward-backward recursions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Implementation {
    /// Work in log space (default; robust to underflow).
    Log,
    /// Work with per-step scaling factors.
    Scaling,
}

impl FromStr for Implementation {
    type Err = HmmError;
    /// Parses `"log"` or `"scaling"`; any other string yields
    /// [`HmmError::InvalidParameter`].
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "log" => Ok(Implementation::Log),
            "scaling" => Ok(Implementation::Scaling),
            other => Err(HmmError::InvalidParameter(format!(
                "implementation must be 'log' or 'scaling', got {other:?}"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paramset_from_codes() {
        let p = ParamSet::from_codes("stmc");
        assert!(p.contains(Param::Start));
        assert!(p.contains(Param::Trans));
        assert!(p.contains(Param::Means));
        assert!(p.contains(Param::Covars));
        assert!(!p.contains(Param::Emit));
        assert!(!p.contains(Param::Weights));
    }

    #[test]
    fn paramset_ignores_unknown_and_empty() {
        assert!(ParamSet::from_codes("").is_empty());
        let p = ParamSet::from_codes("sZt9");
        assert!(p.contains(Param::Start));
        assert!(p.contains(Param::Trans));
        assert!(!p.contains(Param::Means));
    }

    #[test]
    fn enum_parsing() {
        assert_eq!(
            "viterbi".parse::<DecoderAlgorithm>().unwrap(),
            DecoderAlgorithm::Viterbi
        );
        assert_eq!(
            "log".parse::<Implementation>().unwrap(),
            Implementation::Log
        );
        assert!("nope".parse::<Implementation>().is_err());
    }
}
