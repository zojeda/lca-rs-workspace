// Run with:
//   cargo run -p lca-lsolver --example pardiso_solve --features pardiso
// Requires Intel MKL runtime available (e.g., oneAPI: set MKLROOT/LD_LIBRARY_PATH).

#[cfg(all(feature = "pardiso", not(target_arch = "wasm32")))]
fn main() {
    use lca_core::sparse_matrix::{SparseMatrix, Triplete};
    use lca_lsolver::algorithms::pardiso_direct::{PardisoConfig, PardisoDirect, PardisoMatrixType};
    use lca_core::devices::CpuDevice;

    // Simple 3x3 general system
    // [ 4 -1  0 ] [x0]   [ 3 ]
    // [-1  4 -1 ] [x1] = [ 2 ]
    // [ 0 -1  3 ] [x2]   [ 1 ]
    let a = SparseMatrix::from_triplets(
        3,
        3,
        vec![
            Triplete::new(0, 0, 4.0),
            Triplete::new(0, 1, -1.0),
            Triplete::new(1, 0, -1.0),
            Triplete::new(1, 1, 4.0),
            Triplete::new(1, 2, -1.0),
            Triplete::new(2, 1, -1.0),
            Triplete::new(2, 2, 3.0),
        ],
    )
    .expect("build matrix");
    let b = vec![3.0, 2.0, 1.0];

    let cfg = PardisoConfig { matrix_type: PardisoMatrixType::General, max_threads: None };
    let solver = PardisoDirect::new(cfg);
    let cpu = CpuDevice::default();

    let result = pollster::block_on(async { solver.solve(&cpu, &a, &b).await });
    match result {
        Ok(sol) => {
            println!("x = {:?}", sol.x);
        }
        Err(e) => {
            eprintln!("PARDISO example failed: {e}");
            eprintln!("Hint: ensure MKL is installed and available (MKLROOT/LD_LIBRARY_PATH).\n");
        }
    }
}

#[cfg(any(not(feature = "pardiso"), target_arch = "wasm32"))]
fn main() {
    eprintln!("This example requires the 'pardiso' feature and a native target.");
}
