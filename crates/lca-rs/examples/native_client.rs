use lca_core::{GpuDevice, SparseMatrix};

// Removed error import: use lca_lsolver::error::LsolverWgpuError;

// Use a generic error type for the example
type ExampleError = Box<dyn std::error::Error>;

#[tokio::main]
async fn main() -> Result<(), ExampleError> {
    // Initialize logging for native execution, ignoring wgpu messages
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .filter_module("wgpu", log::LevelFilter::Off)
        .init();

    log::info!("Starting Native LCA Solver Example...");

    // --- 1. Initialize Device ---
    log::debug!("Initializing native GPU device...");
    let device = GpuDevice::new().await?; // Use native constructor

    // --- 2. Define Sample Data (CPU) ---
    // Example: A = [[2, -1], [-1, 2]], B = [[1, 1]], C = [[1], [1]], f = [1, 1]
    // Expected: x = A^-1 * f = [1, 1]^T
    //           g = B * x = [1*1 + 1*1] = [2]
    //           h = C * g = [[1*2], [1*2]] = [2, 2]^T

    // // A: 2x2 sparse matrix
    // let a_rows = 2;
    // let a_cols = 2;
    // let a_indptr = vec![0, 2, 4]; // CSR row pointers
    // let a_indices = vec![0, 1, 0, 1]; // CSR column indices
    // let a_data = vec![2.0, -1.0, -1.0, 2.0]; // CSR data
    // let a_cpu = SparseMatrix::from_csr(a_rows, a_cols, a_data, a_indices, a_indptr)?;

    // // B: 1x2 sparse matrix
    // let b_rows = 1;
    // let b_cols = 2;
    // let b_indptr = vec![0, 2];
    // let b_indices = vec![0, 1];
    // let b_data = vec![1.0, 1.0];
    // let b_cpu = SparseMatrix::from_csr(b_rows, b_cols, b_data, b_indices, b_indptr)?;

    // // C: 2x1 sparse matrix
    // let c_rows = 2;
    // let c_cols = 1;
    // let c_indptr = vec![0, 1, 2];
    // let c_indices = vec![0, 0]; // Column index is 0 for both rows
    // let c_data = vec![1.0, 1.0];
    // let c_cpu = SparseMatrix::from_csr(c_rows, c_cols, c_data, c_indices, c_indptr)?;

    // // f: Dense vector
    // let f = vec![1.0, 1.0];

    // log::info!("Sample data defined.");
    // log::info!("A = {:?}", a_cpu);
    // log::info!("B = {:?}", b_cpu);
    // log::info!("C = {:?}", c_cpu);
    // log::info!("f = {:?}", f);

    // let h = calculate_lca(&device, a_cpu, b_cpu, c_cpu, f, 1000, 0.001).await?;
    // // --- 9. Print Result ---
    // log::info!("Native LCA calculation completed successfully.");
    // println!("\n======================================");
    // println!("Final Result h = {:?}", h);
    // println!("======================================");

    // // Optional: Print transfer stats
    // let ts = device.get_transfer_stats();
    // log::info!("Bytes transferred To GPU: {}", ts.bytes_to_gpu);
    // log::info!("Bytes transferred From GPU: {}", ts.bytes_from_gpu);

    Ok(())
}
