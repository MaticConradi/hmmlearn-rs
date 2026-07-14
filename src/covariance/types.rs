//! The covariance-parameterization tag shared by the Gaussian-family models.

use crate::error::HmmError;
use std::fmt;
use std::str::FromStr;

/// How a Gaussian model parameterizes its covariance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CovarianceType {
    /// One scalar variance per state: stored `(n_components,)`.
    Spherical,
    /// One variance per feature per state: stored `(n_components, n_features)`.
    Diag,
    /// A full matrix per state: stored `(n_components, n_features, n_features)`.
    Full,
    /// A single full matrix shared by all states: stored `(n_features, n_features)`.
    Tied,
}

impl CovarianceType {
    /// The lowercase name used by hmmlearn (`"spherical"`, `"diag"`, …).
    pub fn as_str(self) -> &'static str {
        match self {
            CovarianceType::Spherical => "spherical",
            CovarianceType::Diag => "diag",
            CovarianceType::Full => "full",
            CovarianceType::Tied => "tied",
        }
    }
}

impl fmt::Display for CovarianceType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for CovarianceType {
    type Err = HmmError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "spherical" => Ok(CovarianceType::Spherical),
            "diag" => Ok(CovarianceType::Diag),
            "full" => Ok(CovarianceType::Full),
            "tied" => Ok(CovarianceType::Tied),
            other => Err(HmmError::InvalidParameter(format!(
                "covariance_type must be one of spherical/diag/full/tied, got {other:?}"
            ))),
        }
    }
}
