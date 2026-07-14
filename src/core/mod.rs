//! Inference engine: shared parameters, algorithms, and the generic HMM core.
//!
//! Phase 0 establishes the parameter/enum types; the algorithms, convergence
//! monitor, and `Hmm<E>` core land in later phases.

pub mod algorithms;
pub mod emission;
pub mod fitted;
pub mod hmm;
pub mod inference;
pub mod monitor;
pub mod params;

pub use emission::EmissionModel;
pub use fitted::Fitted;
pub use hmm::Hmm;
pub use inference::{Em, Inference, Variational};
pub use monitor::ConvergenceMonitor;
pub use params::{DecoderAlgorithm, Implementation, Param, ParamSet};
