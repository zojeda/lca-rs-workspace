// This test file is currently empty as the only test relied on the removed
// ILU(0)-based solver (previously named SparseLU).
// Add new tests here when other GPU algorithms (e.g., CG) are implemented.

use lca_core::device::GpuDevice;
use lca_lsolver::{
    algorithms::{ConjugateGradient, SolveAlgorithm},
    LcaCoreError,
    SparseMatrix, // Removed SparseLU and lu_sparse imports
};
use pollster::block_on;

use lca_lsolver::algorithms::BiCGSTAB; // Import the new solver
                                       // Import Buffer

// Helper for float comparison in tests
fn assert_approx_eq_vec(a: &[f32], b: &[f32], tolerance: f32) {
    assert_eq!(a.len(), b.len(), "Vector lengths differ");
    for i in 0..a.len() {
        let diff = (a[i] - b[i]).abs();
        assert!(
            diff <= tolerance,
            "Verification failed at index {}: expected {}, got {}, diff {}",
            i,
            b[i],
            a[i],
            diff
        );
    }
}

#[test]
fn test_gpu_solve_conjugate_gradient() -> Result<(), LcaCoreError> {
    block_on(async {
        // 1. Setup Device
        let gpu_device = GpuDevice::new().await?;

        // 2. Setup Matrix & Vector
        // Example:
        let a_dense = vec![
            vec![4.0, -1.0, 0.0],
            vec![-1.0, 4.0, -1.0],
            vec![0.0, -1.0, 4.0],
        ];
        let a_sparse = SparseMatrix::from_dense(&a_dense);
        let a_sparse = gpu_device.create_sparse_matrix(&a_sparse)?;
        let b = vec![1.0, 2.0, 3.0];

        // 3. Setup Algorithm
        let algorithm = ConjugateGradient::default();

        // 4. Solve
        let x_result = algorithm.solve(&gpu_device, &a_sparse, &b).await?;

        // 5. Verify
        let expected_x = vec![0.464, 0.857, 0.964];
        assert_approx_eq_vec(&x_result.x, &expected_x, 1e-3);

        Ok(())
    })
}

#[test]
fn test_gpu_solve_bicgstab() -> Result<(), LcaCoreError> {
    block_on(async {
        // 1. Setup Context & Device

        let gpu_device = GpuDevice::new().await?;

        // 2. Setup Matrix & Vector
        // Example:
        let a_dense = vec![
            vec![4.0, -1.0, 0.0],
            vec![-1.0, 4.0, -1.0],
            vec![0.0, -1.0, 4.0],
        ];
        let a_sparse = SparseMatrix::from_dense(&a_dense);
        let a_sparse = gpu_device.create_sparse_matrix(&a_sparse)?;
        let b = vec![1.0, 2.0, 3.0];

        // 3. Setup Algorithm
        let algorithm = BiCGSTAB::default();

        // 4. Solve
        let x_result = algorithm.solve(&gpu_device, &a_sparse, &b).await?;

        // 5. Verify
        let expected_x = vec![0.464, 0.857, 0.964];
        assert_approx_eq_vec(&x_result.x, &expected_x, 1e-3);

        Ok(())
    })
}
