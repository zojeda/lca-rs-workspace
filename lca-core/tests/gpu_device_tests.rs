#![cfg(not(target_arch = "wasm32"))]
use lca_core::{GpuDevice, SparseMatrix};
use pollster::block_on;

// Helper for float comparison
fn assert_approx_eq_slice(a: &[f64], b: &[f64], tol: f64) {
    assert_eq!(a.len(), b.len());
    for (i, (ai, bi)) in a.iter().zip(b.iter()).enumerate() {
        let d = (ai - bi).abs();
        assert!(d <= tol, "idx {i}: {ai} != {bi} (|diff|={d})");
    }
}

#[test]
fn axpy_and_dot_work() {
    // Skip if no GPU is available by returning early on init error
    let device = match block_on(GpuDevice::new()) {
        Ok(d) => d,
        Err(_) => return, // gracefully skip
    };

    let x_data = vec![1.0, 2.0, 3.0, 4.0];
    let y_data = vec![10.0, 20.0, 30.0, 40.0];

    let x = device.create_vector("x", &x_data).unwrap();
    let mut y = device.create_vector("y", &y_data).unwrap();

    block_on(device.axpy(0.5, &x, &mut y)).unwrap();

    let y_out = block_on(y.read_contents()).unwrap();
    let expected = vec![10.5, 21.0, 31.5, 42.0];
    assert_approx_eq_slice(&y_out, &expected, 1e-9);

    let dot = block_on(device.dot(&x, &y)).unwrap();
    // x · y = 1*10.5 + 2*21 + 3*31.5 + 4*42 = 10.5 + 42 + 94.5 + 168 = 315.0
    assert!((dot - 315.0).abs() <= 1e-9);
}

#[test]
fn invert_and_elementwise_mul_work() {
    let device = match block_on(GpuDevice::new()) {
        Ok(d) => d,
        Err(_) => return,
    };

    let a = device.create_vector("a", &[1.0, 2.0, 4.0, 0.0]).unwrap();
    let mut inv = device.create_empty_vector("inv", 4).unwrap();
    block_on(device.invert_elements(&a, &mut inv)).unwrap();
    let inv_out = block_on(inv.read_contents()).unwrap();
    assert_approx_eq_slice(&inv_out, &[1.0, 0.5, 0.25, 0.0], 1e-12);

    let b = device.create_vector("b", &[2.0, 3.0, 4.0, 5.0]).unwrap();
    let mut c = device.create_empty_vector("c", 4).unwrap();
    block_on(device.elementwise_mul(&a, &b, &mut c)).unwrap();
    let c_out = block_on(c.read_contents()).unwrap();
    assert_approx_eq_slice(&c_out, &[2.0, 6.0, 16.0, 0.0], 1e-12);
}

#[test]
fn extract_diagonal_works() {
    let device = match block_on(GpuDevice::new()) {
        Ok(d) => d,
        Err(_) => return,
    };

    // Create a 3x3 matrix with diagonal [1,2,3]
    let triplets = vec![
        lca_core::sparse_matrix::Triplete::new(0, 0, 1.0),
        lca_core::sparse_matrix::Triplete::new(1, 1, 2.0),
        lca_core::sparse_matrix::Triplete::new(2, 2, 3.0),
        lca_core::sparse_matrix::Triplete::new(0, 2, 5.0), // off-diagonal
    ];
    let cpu = SparseMatrix::from_triplets(3, 3, triplets).unwrap();
    let gpu = device.create_sparse_matrix(&cpu).unwrap();

    let mut dvec = device.create_empty_vector("diag", 3).unwrap();
    block_on(device.extract_diagonal(&gpu, &mut dvec)).unwrap();
    let diag = block_on(dvec.read_contents()).unwrap();
    assert_approx_eq_slice(&diag, &[1.0, 2.0, 3.0], 1e-12);
}
