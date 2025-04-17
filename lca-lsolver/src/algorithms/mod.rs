use lca_core::{device::Device, LcaCoreError, Matrix}; // Use error and matrix trait from lca_core

pub struct SolveResult<V: Float, M> {
    pub x: Vec<V>,   // Solution vector
    pub metadata: M, // Metadata about the solve process
}

// --- Algorithm Trait Definition ---
/// Trait representing a specific linear system solving algorithm.
/// Generic over the Device (CPU/GPU) and Matrix type (Sparse/Dense) it supports.
pub trait SolveAlgorithm<D: Device, M: Matrix> {
    /// The numeric type the algorithm operates on (e.g., f32, f32).
    /// Must match the Matrix::Value type.
    type Value: Float + Copy + Send + Sync + std::fmt::Debug + Default + bytemuck::Pod;
    type Metadata: std::fmt::Debug;

    /// Solves the linear system Ax = b for x.
    ///
    /// # Arguments
    ///
    /// * `device` - The execution device (CPU or GPU).
    /// * `a` - The coefficient matrix A.
    /// * `b` - The right-hand side vector b.
    ///
    /// # Returns
    ///
    /// A `Result` containing the solution vector x or an `LsolverError`.
    fn solve(
        &self,
        device: &D,
        a: &M,
        b: &[Self::Value],
    ) -> impl std::future::Future<
        Output = Result<SolveResult<Self::Value, Self::Metadata>, LcaCoreError>,
    > + Send;

    // Helper for input validation, can be called by implementations.
    // Note: This validation assumes the trait's Value matches the Matrix's Value.
    // Rust's type system doesn't easily enforce M::Value == Self::Value directly in the trait definition,
    // but implementations should ensure this consistency.
    fn validate_inputs(&self, a: &M, b: &[Self::Value]) -> Result<(), LcaCoreError> {
        let (rows, cols) = a.dims();
        if !a.is_square() {
            return Err(LcaCoreError::InvalidDimensions(format!(
                "Matrix A must be square (dims: {}x{})",
                rows, cols
            )));
        }
        if rows != b.len() {
            return Err(LcaCoreError::InvalidDimensions(format!(
                "Matrix A rows ({}) must match RHS vector b length ({})",
                rows,
                b.len()
            )));
        }
        // TODO: Potentially check if M::Value matches Self::Value if possible,
        // maybe via a PhantomData marker or runtime check if absolutely necessary,
        // but ideally the generic impl constraints handle this.
        Ok(())
    }
}

// --- Algorithm Implementations ---

// Declare the modules for specific algorithm implementations
// Removed: pub mod gpu_ops;
pub mod gpu_sparse_bicgstab; // GPU Sparse BiConjugate Gradient Stabilized
pub mod gpu_sparse_cg_checked; // GPU Sparse Conjugate Gradient with SPD check

use num_traits::Float;

// --- Algorithm Struct Definitions ---
// Define structs that represent specific algorithms and hold their parameters.

/// BiConjugate Gradient Stabilized Algorithm.
#[derive(Debug, Clone)]
pub struct BiCGSTAB {
    pub tolerance: f32, // Or make generic T?
    pub max_iterations: usize,
    pub use_preconditioner: bool, // Flag for Jacobi preconditioner
}

impl Default for BiCGSTAB {
    fn default() -> Self {
        Self {
            tolerance: 1e-6,           // Default tolerance
            max_iterations: 1000,      // Default max iterations
            use_preconditioner: false, // Default to no preconditioner
        }
    }
}

impl BiCGSTAB {
    /// Creates a new instance of the BiCGSTAB algorithm with default parameters.
    pub fn new() -> Self {
        Self::default()
    }
    /// Creates a new instance of the BiCGSTAB algorithm with specified parameters.
    pub fn with_params(tolerance: f32, max_iterations: usize, use_preconditioner: bool) -> Self {
        Self {
            tolerance,
            max_iterations,
            use_preconditioner,
        }
    }
}

/// Conjugate Gradient Algorithm.
#[derive(Debug, Clone)]
pub struct ConjugateGradient {
    pub tolerance: f32, // Or make generic T?
    pub max_iterations: usize,
    // Optional: Preconditioner type could go here
}

impl Default for ConjugateGradient {
    fn default() -> Self {
        Self {
            tolerance: 1e-6,      // Default tolerance
            max_iterations: 1000, // Default max iterations
        }
    }
}

impl ConjugateGradient {
    /// Creates a new instance of the Conjugate Gradient algorithm with default parameters.
    pub fn new() -> Self {
        Self::default()
    }
    /// Creates a new instance of the Conjugate Gradient algorithm with specified parameters.
    pub fn with_params(tolerance: f32, max_iterations: usize) -> Self {
        Self {
            tolerance,
            max_iterations,
        }
    }
}
