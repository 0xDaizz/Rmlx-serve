//! Re-exports for backwards compatibility.
//!
//! All types are defined in `lib.rs` and re-exported here so that
//! internal modules can use `crate::traits::Engine` if they prefer.

pub use crate::{Engine, EngineError, EngineHealth, EngineStats};
