// Import necessary types from lca_core
use lca_core::{
    device::GpuDevice,
    GpuVector,
    LcaCoreError,
    Matrix,
    SparseMatrixGpu, // Use GPU matrix directly
};
// Keep local imports
use log::{info, warn};
// Import local algorithm traits
use super::{
    // Removed gpu_ops import
    BiCGSTAB,
    SolveAlgorithm,
    SolveResult,
};

#[derive(Debug, Clone, Copy)]
pub struct BiCGSTABMetadata {
    pub iterations: usize,
    pub residual_norm: f32,
}

// Implement for SparseMatrixGpu now
impl SolveAlgorithm<GpuDevice, SparseMatrixGpu> for BiCGSTAB {
    type Value = f32;
    type Metadata = BiCGSTABMetadata;

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
        let b_gpu = device.create_vector("b_gpu", b)?;
        // TODO: Initialize x_gpu with zeros if needed. Assuming the demand vector start for now.
        let mut x_gpu = device.create_vector("x_gpu (initial_guess)", b)?;

        // Use pre-defined parameters
        let max_iterations = self.max_iterations;
        let tolerance = self.tolerance;
        // Read preconditioner flag from self
        let use_preconditioner = self.use_preconditioner;

        // Execute the BiCGSTAB solver on GPU.
        let (iterations, residual_norm) = gpu_sparse_bicgstab(
            device, // Pass the device
            a,
            &b_gpu,
            &mut x_gpu,
            tolerance,
            max_iterations,
            use_preconditioner, // Pass flag
        )
        .await?;

        // Read back the solution vector from the GPU buffer.
        let result = x_gpu.read_contents().await?; // Use internal context
        Ok(SolveResult {
            x: result,
            metadata: BiCGSTABMetadata {
                iterations,
                residual_norm,
            },
        })
    }
}

// --- Main BiCGSTAB Solver Logic ---

/// Solves the sparse linear system Ax = b using the BiConjugate Gradient Stabilized (BiCGSTAB) method on the GPU,
/// optionally with Jacobi preconditioning.
///
/// # Arguments
/// * `device` - The GpuDevice used for operations.
/// * `matrix` - The sparse matrix A on the GPU (`SparseMatrixGpu`).
/// * `b` - The right-hand side vector b on the GPU (`GpuVector`).
/// * `x` - The initial guess vector x on the GPU (`GpuVector`, will be modified in place).
/// * `tolerance` - The convergence tolerance for the residual norm.
/// * `max_iterations` - The maximum number of iterations allowed.
/// * `use_preconditioner` - Flag to enable Jacobi preconditioning.
///
/// # Returns
///
/// A `Result` containing a tuple `(iterations, residual_norm)` on success,
/// or an `LsolverError` on failure.
pub(super) async fn gpu_sparse_bicgstab(
    device: &GpuDevice, // Add device parameter
    matrix: &SparseMatrixGpu,
    b: &GpuVector,
    x: &mut GpuVector,
    tolerance: f32,
    max_iterations: usize,
    use_preconditioner: bool, // Added flag
) -> Result<(usize, f32), LcaCoreError> {
    // Context is no longer accessed directly. Use device methods.
    let n = matrix.rows();
    // Dimension checks using GpuVector::size()
    if matrix.cols() != n || b.size() != n || x.size() != n {
        return Err(LcaCoreError::InvalidDimensions(format!(
            "Matrix and vector dimensions are incompatible: A=({}x{}), b={}, x={}",
            matrix.rows(),
            matrix.cols(),
            b.size(),
            x.size()
        )));
    }

    // Allocate necessary vectors using device.create_empty_vector
    let mut r = device.create_empty_vector("r", n)?;
    let mut r_hat_0 = device.create_empty_vector("r_hat_0", n)?;
    let mut p = device.create_empty_vector("p", n)?;
    let mut v = device.create_empty_vector("v", n)?;
    let mut s = device.create_empty_vector("s", n)?;
    let mut t = device.create_empty_vector("t", n)?;
    let mut temp_ax = device.create_empty_vector("temp_ax", n)?;
    // Preconditioner vectors
    let mut inv_diag_gpu = device.create_empty_vector("inv_diag", n)?;
    let mut z = device.create_empty_vector("z_prec", n)?; // For Mz = r
    let mut phat = device.create_empty_vector("phat_prec", n)?; // Intermediate for p update
    let mut shat = device.create_empty_vector("shat_prec", n)?; // Intermediate for s update

    // --- Preconditioner Setup ---
    if use_preconditioner {
        info!("Setting up Jacobi preconditioner...");
        let mut diag_gpu = device.create_empty_vector("diag", n)?;
        device.extract_diagonal(matrix, &mut diag_gpu).await?;
        device.invert_elements(&diag_gpu, &mut inv_diag_gpu).await?;
        // diag_gpu is no longer needed
    } else {
        // If not using preconditioner, set inv_diag to ones (identity preconditioner)
        // This is less efficient than a dedicated non-preconditioned path, but simpler for now.
        warn!("Preconditioner disabled, using identity (inefficient setup).");
        let ones = vec![1.0; n];
        inv_diag_gpu = device.create_vector("inv_diag_ones", &ones)?;
    }
    // --- End Preconditioner Setup ---

    // Initial calculation: r = b - A*x
    // 1. temp_ax = A*x
    matrix.spmv(x, &mut temp_ax).await?; // Use SparseMatrixGpu::spmv
                                         // 2. r = b (copy b into r)
    r.clone_from(b)?; // Use internal context
                      // 3. r = r - temp_ax
    device.axpy(-1.0, &temp_ax, &mut r).await?; // Use device.axpy

    // r_hat_0 = r (copy) - Note: For preconditioned BiCGSTAB, r_hat_0 is often chosen differently,
    // but using the initial residual is a common starting point.
    r_hat_0.clone_from(&r)?; // Use internal context

    // Initial scalars
    let mut rho: f32;
    let mut rho_prev: f32 = 1.0;
    let mut alpha: f32 = 1.0;
    let mut omega: f32 = 1.0;

    // Calculate initial residual norm (||r||)
    let r_dot_r = device.dot(&r, &r).await?; // Use device.dot
    let mut residual_norm = r_dot_r.sqrt();
    info!("BiCGSTAB Initial Residual Norm: {}", residual_norm);

    if residual_norm < tolerance {
        info!("BiCGSTAB converged in 0 iterations.");
        return Ok((0, residual_norm));
    }

    let mut iterations = 0;
    for i in 1..=max_iterations {
        iterations = i;

        // --- Preconditioned Step: Solve Mz = r ---
        // z = inv_diag * r
        device.elementwise_mul(&inv_diag_gpu, &r, &mut z).await?;
        // --- End Preconditioned Step ---

        rho = device.dot(&r_hat_0, &z).await?; // rho = r_hat_0^T * z

        if rho.abs() < f32::EPSILON {
            warn!(
                "BiCGSTAB breakdown: rho ({}) is near zero at iteration {}",
                rho, i
            );
            return Err(LcaCoreError::BiCGSTABBreakdown {
                iteration: i,
                value_name: "rho".to_string(),
                value: rho,
            });
        }

        if i == 1 {
            // p = z
            p.clone_from(&z)?;
        } else {
            if omega.abs() < f32::EPSILON {
                warn!(
                    "BiCGSTAB breakdown: omega ({}) is near zero at iteration {}",
                    omega, i
                );
                return Err(LcaCoreError::BiCGSTABBreakdown {
                    iteration: i,
                    value_name: "omega".to_string(),
                    value: omega,
                });
            }
            if rho_prev.abs() < f32::EPSILON {
                warn!(
                    "BiCGSTAB breakdown: rho_prev ({}) is near zero at iteration {}",
                    rho_prev, i
                );
                return Err(LcaCoreError::BiCGSTABBreakdown {
                    iteration: i,
                    value_name: "rho_prev".to_string(),
                    value: rho_prev,
                });
            }
            let beta = (rho / rho_prev) * (alpha / omega);
            // p = z + beta * (p - omega * v)
            // Step 1: p_temp = p - omega * v
            phat.clone_from(&p)?; // Use phat as p_temp
            device.axpy(-omega, &v, &mut phat).await?; // phat now holds p - omega * v

            // Step 2: p = z + beta * phat
            p.clone_from(&z)?; // p = z
            device.axpy(beta, &phat, &mut p).await?; // p = z + beta * phat
        }

        // --- Preconditioned Step: Solve M*phat = p ---
        // phat = inv_diag * p
        device.elementwise_mul(&inv_diag_gpu, &p, &mut phat).await?;
        // --- End Preconditioned Step ---

        // v = A*phat (Note: using phat which incorporates M^-1)
        matrix.spmv(&phat, &mut v).await?;

        let r_hat_0_dot_v = device.dot(&r_hat_0, &v).await?;
        if r_hat_0_dot_v.abs() < f32::EPSILON {
            warn!(
                "BiCGSTAB breakdown: r_hat_0_dot_v ({}) is near zero at iteration {}",
                r_hat_0_dot_v, i
            );
            return Err(LcaCoreError::BiCGSTABBreakdown {
                iteration: i,
                value_name: "r_hat_0_dot_v".to_string(),
                value: r_hat_0_dot_v,
            });
        }
        alpha = rho / r_hat_0_dot_v;

        // s = r - alpha * v
        s.clone_from(&r)?;
        device.axpy(-alpha, &v, &mut s).await?;

        // Check convergence of s (intermediate residual)
        let s_norm = device.dot(&s, &s).await?.sqrt();
        if s_norm < tolerance {
            // x = x + alpha * phat (Update using preconditioned direction)
            device.axpy(alpha, &phat, x).await?;
            residual_norm = s_norm; // Use s_norm as final residual
            info!("BiCGSTAB converged early on s norm at iteration {}", i);
            break;
        }

        // --- Preconditioned Step: Solve M*shat = s ---
        // shat = inv_diag * s
        device.elementwise_mul(&inv_diag_gpu, &s, &mut shat).await?;
        // --- End Preconditioned Step ---

        // t = A*shat (Note: using shat which incorporates M^-1)
        matrix.spmv(&shat, &mut t).await?;

        let t_dot_t = device.dot(&t, &t).await?;
        if t_dot_t.abs() < f32::EPSILON {
            // Avoid division by zero if t is zero
            warn!(
                "BiCGSTAB breakdown: t_dot_t ({}) is near zero at iteration {}",
                t_dot_t, i
            );
            // If t is zero, omega calculation fails. Can sometimes set omega=0 and continue,
            // but often indicates stagnation. Let's treat as breakdown.
            return Err(LcaCoreError::BiCGSTABBreakdown {
                iteration: i,
                value_name: "t_dot_t".to_string(),
                value: t_dot_t,
            });
        }
        let t_dot_s = device.dot(&t, &s).await?;
        omega = t_dot_s / t_dot_t;

        // x = x + alpha * phat + omega * shat (Update using preconditioned directions)
        device.axpy(alpha, &phat, x).await?;
        device.axpy(omega, &shat, x).await?;

        // r = s - omega * t
        r.clone_from(&s)?;
        device.axpy(-omega, &t, &mut r).await?;

        // Check convergence of r (final residual for the iteration)
        residual_norm = device.dot(&r, &r).await?.sqrt();
        info!(
            "Iteration {}: residual norm = {}",
            iterations, // Use current iteration number
            residual_norm
        );
        if residual_norm < tolerance {
            info!("BiCGSTAB converged on r norm at iteration {}", i);
            break;
        }

        // Check for stagnation related to omega
        if omega.abs() < f32::EPSILON {
            warn!(
                "BiCGSTAB breakdown: omega ({}) is near zero at iteration {}",
                omega, i
            );
            return Err(LcaCoreError::BiCGSTABBreakdown {
                iteration: i,
                value_name: "omega".to_string(),
                value: omega,
            });
        }

        rho_prev = rho;

        if i == max_iterations {
            warn!(
                "BiCGSTAB reached maximum iterations ({}) without converging. Residual norm: {}",
                max_iterations, residual_norm
            );
            return Err(LcaCoreError::NonConvergence);
        }
    }

    info!(
        "BiCGSTAB finished ({} iterations, Residual Norm: {})",
        iterations, residual_norm
    );

    Ok((iterations, residual_norm))
}
