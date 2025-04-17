use lca_core::{device::{GpuDevice, TransferStats}, sparse_matrix::Triplete};
use lca_lsolver::{
    algorithms::{BiCGSTAB, SolveAlgorithm},
    SparseMatrix,
};
use std::time::Instant;

/// Creates a pentadiagonal sparse matrix A of size n x n.
/// Diagonals:
/// - Main: 4.0
/// - Adjacent (+1, -1): -1.0
/// - Outer (+2, -2): -0.5
fn create_pentadiagonal_matrix(n: usize) -> SparseMatrix {
    let mut triplets = Vec::new();

    for i in 0..n {
        let row = i as usize;

        // Diagonal -2
        if i >= 2 {
            triplets.push(Triplete::new(row, i - 2, -0.5));
        }
        // Diagonal -1
        if i >= 1 {
            triplets.push(Triplete::new(row, i - 1, -1.0));
        }
        // Main Diagonal
        triplets.push(Triplete::new(row, i, 4.0));
        // Diagonal +1
        if i + 1 < n {
            triplets.push(Triplete::new(row, i + 1, -1.0));
        }
        // Diagonal +2
        if i + 2 < n {
            triplets.push(Triplete::new(row, i + 2, -0.5));
        }
    }

    SparseMatrix::from_triplets(n, n, triplets)
        .expect("Failed to create sparse matrix from COO")
}

/// Creates a vector b of size n with b[i] = sin(i).
fn create_sin_vector(n: usize) -> Vec<f32> {
    (0..n).map(|i| (i as f32 / n as f32).sin()).collect()
}

#[tokio::main]
async fn main() {
    // Initialize logging based on RUST_LOG environment variable
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .filter_module("wgpu", log::LevelFilter::Off)
        .init();

    let n = 500;
    log::info!(
        "Setting up {}x{} pentadiagonal matrix A and sin vector b...",
        n,
        n
    );

    let a = create_pentadiagonal_matrix(n);

    let b = create_sin_vector(n);

    // 1. Setup Device
    let gpu_device = GpuDevice::new().await.expect("Failed to create GPU device");
    let a = gpu_device
        .create_sparse_matrix(&a)
        .expect("Failed to create GPU matrix");
    let tolerance = 1e-3;
    let max_iterations = n * 5; // A common heuristic

    log::info!("Running GPU Conjugate Gradient solver...");
    log::info!("  Size: {}", n);
    log::info!("  Tolerance: {}", tolerance);
    log::info!("  Max Iterations: {}", max_iterations);

    let start_time = Instant::now();

    // 3. Setup Algorithm
    // Update constructor call to include use_preconditioner flag (set to false for this example)
    let algorithm = BiCGSTAB::with_params(tolerance, max_iterations, false);

    // 4. Reset counters and Solve
    log::info!("Resetting GPU transfer counters...");
    gpu_device.reset_transfer_stats();

    log::info!("Starting solve...");
    let x_result = algorithm.solve(&gpu_device, &a, &b).await;

    let duration = start_time.elapsed();

    // 5. Get and Log Transfer Stats
    let TransferStats {
        bytes_to_gpu,
        bytes_from_gpu,
    } = gpu_device.get_transfer_stats();
    log::info!("GPU Transfer Stats:");
    log::info!("  Bytes CPU -> GPU: {}", bytes_to_gpu);
    log::info!("  Bytes GPU -> CPU: {}", bytes_from_gpu);

    match x_result {
        Ok(result) => {
            log::info!("\nSolver finished successfully!");
            log::info!("  Iterations: {}", result.metadata.iterations);
            log::info!(
                "  Final Residual Norm: {:.6e}",
                result.metadata.residual_norm
            );
            log::info!("  Time elapsed: {:?}", duration);
            // Optionally print parts of the solution vector x
            // log::debug!("Solution vector x (first 10 elements): {:?}", &result.x[..10.min(n)]);
            // log::debug!("Solution vector x (last 10 elements): {:?}", &result.x[n.saturating_sub(10)..]);
        }
        Err(e) => {
            // Use log::error for errors
            log::error!("\nSolver failed: {:?}", e);
        }
    }
}
