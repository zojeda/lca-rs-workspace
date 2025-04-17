//! `lca-lsolver`: A linear equation solver library leveraging WGPU for GPU acceleration.
//!
//! This library provides tools to solve systems of linear equations of the form Ax = b,
//! using different algorithms, matrix types, and execution devices (CPU/GPU).

// Core modules
pub mod algorithms;
pub mod dense_matrix; // Assuming this should be public if used externally

// Re-export from lca_core
pub use lca_core::{
    device::GpuDevice, // Export the public device
    GpuVector,
    LcaCoreError,
    Matrix,
    SparseMatrix,
    SparseMatrixGpu,
    Vector,
    // GpuContext is no longer public
};
