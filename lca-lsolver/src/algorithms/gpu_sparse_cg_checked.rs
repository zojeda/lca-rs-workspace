// Import necessary types from lca_core
use lca_core::{
    device::GpuDevice,
    GpuVector,
    LcaCoreError,
    Matrix,
    SparseMatrixGpu, // Use GPU matrix directly
                     // GpuContext is no longer public
};
// Keep local imports
use log::{info, warn};

// Import local algorithm traits
use super::{
    // Removed gpu_ops import
    ConjugateGradient,
    SolveAlgorithm,
    SolveResult,
};

#[derive(Debug, Clone, Copy)]
pub struct ConjugateGradientMetadata {
    pub iterations: usize,
    pub residual_norm: f32,
}

// Implement for SparseMatrixGpu now, as input is expected to be on GPU
impl SolveAlgorithm<GpuDevice, SparseMatrixGpu> for ConjugateGradient {
    type Value = f32;
    type Metadata = ConjugateGradientMetadata;

    async fn solve(
        &self,
        device: &GpuDevice,
        a: &SparseMatrixGpu, // Changed input type
        b: &[f32],           // Keep b as slice, will be uploaded
    ) -> Result<SolveResult<Self::Value, Self::Metadata>, LcaCoreError> {
        // Validation
        if a.rows() != b.len() {
            return Err(LcaCoreError::InvalidDimensions(format!(
                "Matrix rows ({}) must match b vector length ({})",
                a.rows(),
                b.len()
            )));
        }
        if !a.is_square() {
            return Err(LcaCoreError::InvalidDimensions(
                "Matrix must be square".to_string(),
            ));
        }

        // Context is no longer accessed directly. Use device methods for creation.

        // Create GPU vectors for b and x using the device
        let n = a.rows();
        let b_gpu = device.create_vector("b_gpu", b)?;
        let mut x_gpu = device.create_empty_vector("x_gpu (initial_guess/solution)", n)?;
        // TODO: Initialize x_gpu with zeros if needed, or accept initial guess.
        // If initial guess is non-zero, use device.create_vector instead of create_empty_vector.
        // For now, assuming initial guess is zero, which is handled by the algorithm start.

        // Use pre-defined parameters for the CG solver.
        let max_iterations = self.max_iterations;
        let tolerance = self.tolerance;

        // Execute the Conjugate Gradient solver on GPU.
        // Pass the device along.
        let solve_info = solve_cg_checked(
            device, // Pass the device
            a,
            &b_gpu,
            &mut x_gpu,
            max_iterations,
            tolerance,
        )
        .await?;

        // Read back the solution vector from the GPU buffer using GpuVector method (context is now internal)
        let result = x_gpu.read_contents().await?; // Remove context argument

        Ok(SolveResult {
            x: result,
            metadata: ConjugateGradientMetadata {
                iterations: solve_info.iterations,
                residual_norm: solve_info.residual_norm,
            },
        })
    }
}

/// Information about the solve process.
#[derive(Debug, Clone, Copy)]
pub struct SolveInfo {
    pub iterations: usize,
    pub residual_norm: f32,
}

// --- Main Solver Logic ---

/// Checks if a sparse matrix is symmetric and has positive diagonal elements.
/// Note: This check is currently DEFERRED.
pub async fn check_symmetry_and_positive_diagonal(
    _matrix: &SparseMatrixGpu,
) -> Result<(), LcaCoreError> {
    warn!("Symmetry and positive diagonal checks are currently deferred.");
    // TODO: Implement these checks. Symmetry check requires reading GPU buffers back to CPU (slow).
    // Positive diagonal check can be done on GPU but needs buffer access logic.
    Ok(()) // Assume matrix is valid for now
}

/// Solves Ax = b using GPU Conjugate Gradient, assuming A is SPD.
pub async fn solve_cg_checked(
    device: &GpuDevice, // Add device parameter
    matrix: &SparseMatrixGpu,
    b: &GpuVector,
    x: &mut GpuVector, // Initial guess, overwritten with solution
    max_iterations: usize,
    tolerance: f32,
) -> Result<SolveInfo, LcaCoreError> {
    // Context is no longer needed explicitly here, use device methods or vector/matrix internal context.
    // let context = matrix.context(); // Cannot access private context
    // 1. Check (Deferred)
    // check_symmetry_and_positive_diagonal(matrix, context).await?; // Cannot pass context
    check_symmetry_and_positive_diagonal(matrix).await?; // Update signature if needed

    // Need a GpuDevice instance to create vectors - it's now passed in.
    // let device = GpuDevice { context: matrix.context().clone() }; // No longer needed

    // 2. Initialize Algorithm
    let n = matrix.cols();

    // Use device.create_empty_vector
    let mut r = device.create_empty_vector("r (residual)", n)?;
    let mut p = device.create_empty_vector("p (direction)", n)?;
    let mut ap = device.create_empty_vector("ap (A*p)", n)?;
    let mut ax_tmp = device.create_empty_vector("ax_tmp", n)?;

    // Calculate initial residual: r = b - A*x
    // 1. ax_tmp = A*x
    matrix.spmv(x, &mut ax_tmp).await?;
    // 2. r = b (copy b into r)
    r.clone_from(b)?; // Use internal context
                      // 3. r = r - ax_tmp  (which is r = b - ax_tmp)
    device.axpy(-1.0, &ax_tmp, &mut r).await?; // Use device.axpy

    // Initialize p = r
    p.clone_from(&r)?; // Use internal context

    let mut rs_old = device.dot(&r, &r).await?; // Use device.dot
    let initial_residual_norm = rs_old.sqrt();
    info!("Initial residual norm: {}", initial_residual_norm);

    if initial_residual_norm < tolerance {
        info!("Initial guess is already within tolerance.");
        return Ok(SolveInfo {
            iterations: 0,
            residual_norm: initial_residual_norm,
        });
    }

    // 3. CG Iteration Loop
    let mut iterations = 0;
    loop {
        if iterations >= max_iterations {
            warn!(
                "CG reached maximum iterations ({}) without converging.",
                max_iterations
            );
            return Err(LcaCoreError::NonConvergence);
        }

        // Ap = A * p
        matrix.spmv(&p, &mut ap).await?; // Use SparseMatrixGpu::spmv

        // alpha = rs_old / (p^T * Ap)
        let pt_ap = device.dot(&p, &ap).await?; // Use device.dot
        if pt_ap.abs() < f32::EPSILON {
            warn!("CG breakdown: p^T * Ap is close to zero.");
            return Err(LcaCoreError::Internal(
                "CG breakdown: p^T * Ap near zero".to_string(),
            ));
        }
        let alpha = rs_old / pt_ap;

        // x = x + alpha * p
        device.axpy(alpha, &p, x).await?; // Use device.axpy

        // r = r - alpha * Ap
        device.axpy(-alpha, &ap, &mut r).await?; // Use device.axpy

        // rs_new = r^T * r
        let rs_new = device.dot(&r, &r).await?; // Use device.dot
        let residual_norm = rs_new.sqrt();
        info!(
            "Iteration {}: residual norm = {}",
            iterations + 1,
            residual_norm
        );

        // Check convergence
        if residual_norm < tolerance {
            iterations += 1;
            info!("CG converged in {} iterations.", iterations);
            return Ok(SolveInfo {
                iterations,
                residual_norm,
            });
        }

        // beta = rs_new / rs_old
        if rs_old.abs() < f32::EPSILON {
            warn!("CG breakdown: rs_old is close to zero.");
            return Err(LcaCoreError::Internal(
                "CG breakdown: rs_old near zero".to_string(),
            ));
        }
        let beta = rs_new / rs_old;

        // Update p = r + beta * p
        // Use axpy: p = 1.0 * r + beta * p (but axpy modifies the second arg, so need intermediate step)
        // 1. p_new = beta * p_old
        // 2. p_new = p_new + r
        if beta == 0.0 {
            // If beta is 0, p = r.
            p.clone_from(&r)?; // Use internal context
        } else {
            // Calculate beta * p into ap (reuse buffer)
            // Need temporary copy of p
            let mut p_old = device.create_empty_vector("p_old", n)?;
            p_old.clone_from(&p)?; // Use internal context

            // ap = beta * p_old (using axpy trick: ap = 0*p_old + beta*p_old -> copy then scale)
            // First copy p_old to ap
            ap.clone_from(&p_old)?; // Use internal context
                                    // Now scale ap by beta (using axpy: ap = (beta-1)*ap + ap) - This seems wrong.
                                    // Let's rethink: We want p = r + beta * p_old
                                    // 1. Copy r into p: p = r
            p.clone_from(&r)?; // Use internal context
                               // 2. Add beta * p_old to p: p = p + beta * p_old
            device.axpy(beta, &p_old, &mut p).await?; // Use device.axpy
        }

        // Update rs_old
        rs_old = rs_new;
        iterations += 1;
    }
}
