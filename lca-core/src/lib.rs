//! # Solver Core Library
//!
//! Provides core data structures and GPU compute operations for linear solvers.

// Declare modules
pub mod context;
pub mod error;
pub mod sparse_matrix;
// pub mod matrix;
pub mod device;
pub mod ops; // Keep internal ops module private for now
pub mod traits;
pub mod vector;

mod lca_matrix;
mod lca_system;

// Re-export public types
// GpuContext is now internal
pub use lca_matrix::LcaMatrix;
pub use lca_system::{LcaSystem, DemandItem, InterSystemLink};
pub use device::GpuDevice; // Export the main entry point
pub use error::LcaCoreError;
pub use sparse_matrix::{SparseMatrix, SparseMatrixGpu}; // Keep SparseMatrixGpu public for now, might revisit
pub use vector::GpuVector; // Keep GpuVector public

pub use traits::{Matrix, Vector};
// GpuVector is already exported above

// Needed for trait bounds in impls

// Operations like axpy and dot are now methods on GpuDevice.
// Vector I/O operations (read/write/clone) will be methods on GpuVector, using an internal context.

// --- High-level async operations for SparseMatrixGpu ---
// spmv remains here as it's specific to the matrix data.

impl SparseMatrixGpu {
    /// Performs the sparse matrix-vector multiplication `y = self * x` on the GPU.
    /// Uses the internal GPU context associated with this matrix.
    ///
    /// # Arguments
    /// * `x` - The input vector `x`.
    /// * `y` - The output vector `y` (mutable).
    ///
    /// # Errors
    /// Returns `LsolverError` if dimensions mismatch or GPU execution fails.
    pub async fn spmv(&self, x: &GpuVector, y: &mut GpuVector) -> Result<(), LcaCoreError> {
        // Context is already part of SparseMatrixGpu
        ops::internal_spmv(&self.context, self, x, y).await
    }

    /// Performs the sparse matrix-vector multiplication `y = self^T * x` on the GPU,
    /// where `self` is matrix A.
    /// Uses the internal GPU context associated with this matrix.
    ///
    /// # Arguments
    /// * `x` - The input vector `x` (corresponds to rows of A, so columns of A^T).
    /// * `y` - The output vector `y` (corresponds to columns of A, so rows of A^T, mutable).
    ///
    /// # Errors
    /// Returns `LcaCoreError` if dimensions mismatch or GPU execution fails.
    pub async fn spmv_transpose(&self, x: &GpuVector, y: &mut GpuVector) -> Result<(), LcaCoreError> {
        ops::internal_spmv_transpose(&self.context, self, x, y).await
    }
}
