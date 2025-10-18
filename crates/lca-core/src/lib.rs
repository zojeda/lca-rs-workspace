//! # Solver Core Library
//!
//! Core data structures and GPU compute ops for LCA and linear algebra.

// Declare modules (new structure)
#[path = "context/mod.rs"]
pub mod context;
#[path = "devices/mod.rs"]
pub mod devices;
pub mod error;
#[path = "math/mod.rs"]
pub mod math; // traits, vector, sparse_matrix, ops // cpu/gpu backends

// Backward-compat shim: old `device` module path
#[path = "device/mod.rs"]
pub mod device;

#[path = "models/mod.rs"]
pub mod models; // LCA domain models

// Re-export public types (stable API surface)
pub use devices::GpuDevice;
pub use error::LcaCoreError;
pub use math::{
    sparse_matrix::{SparseMatrix, SparseMatrixGpu},
    traits::{Matrix, Vector},
    vector::GpuVector,
};
pub use models::{DemandItem, InterSystemLink, LcaMatrix, LcaSystem};

// Backward-compat shims for old module paths
pub mod sparse_matrix {
    pub use crate::math::sparse_matrix::*;
}
pub mod vector {
    pub use crate::math::vector::*;
}
pub mod traits {
    pub use crate::math::traits::*;
}
#[allow(unused_imports)]
pub mod ops {
    pub use crate::math::ops::*;
}

// High-level async operations for SparseMatrixGpu live here for convenience
impl math::sparse_matrix::SparseMatrixGpu {
    pub async fn spmv(
        &self,
        x: &math::vector::GpuVector,
        y: &mut math::vector::GpuVector,
    ) -> Result<(), LcaCoreError> {
        math::ops::internal_spmv(&self.context, self, x, y).await
    }
    pub async fn spmv_transpose(
        &self,
        x: &math::vector::GpuVector,
        y: &mut math::vector::GpuVector,
    ) -> Result<(), LcaCoreError> {
        math::ops::internal_spmv_transpose(&self.context, self, x, y).await
    }
}
