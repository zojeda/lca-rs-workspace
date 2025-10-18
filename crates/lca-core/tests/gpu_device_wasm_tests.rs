#![cfg(target_arch = "wasm32")]

use lca_core::{GpuDevice, SparseMatrix};
use wasm_bindgen_futures::spawn_local;
use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

fn assert_approx_eq_slice(a: &[f64], b: &[f64], tol: f64) {
    assert_eq!(a.len(), b.len());
    for (i, (ai, bi)) in a.iter().zip(b.iter()).enumerate() {
        let d = (ai - bi).abs();
        assert!(d <= tol, "idx {i}: {ai} != {bi} (|diff|={d})");
    }
}

#[wasm_bindgen_test]
fn wasm_axpy_and_dot_work() {
    spawn_local(async move {
        let device = match GpuDevice::new().await {
            Ok(d) => d,
            Err(_) => return, // skip
        };

        let x_data = vec![1.0, 2.0, 3.0, 4.0];
        let y_data = vec![10.0, 20.0, 30.0, 40.0];
        let x = device.create_vector("x", &x_data).unwrap();
        let mut y = device.create_vector("y", &y_data).unwrap();

        device.axpy(0.5, &x, &mut y).await.unwrap();
        let y_out = y.read_contents().await.unwrap();
        assert_approx_eq_slice(&y_out, &[10.5, 21.0, 31.5, 42.0], 1e-9);

        let dot = device.dot(&x, &y).await.unwrap();
        assert!((dot - 315.0).abs() <= 1e-9);
    });
}

#[wasm_bindgen_test]
fn wasm_invert_and_elementwise_mul_work() {
    spawn_local(async move {
        let device = match GpuDevice::new().await {
            Ok(d) => d,
            Err(_) => return,
        };

        let a = device.create_vector("a", &[1.0, 2.0, 4.0, 0.0]).unwrap();
        let mut inv = device.create_empty_vector("inv", 4).unwrap();
        device.invert_elements(&a, &mut inv).await.unwrap();
        let inv_out = inv.read_contents().await.unwrap();
        assert_approx_eq_slice(&inv_out, &[1.0, 0.5, 0.25, 0.0], 1e-12);

        let b = device.create_vector("b", &[2.0, 3.0, 4.0, 5.0]).unwrap();
        let mut c = device.create_empty_vector("c", 4).unwrap();
        device.elementwise_mul(&a, &b, &mut c).await.unwrap();
        let c_out = c.read_contents().await.unwrap();
        assert_approx_eq_slice(&c_out, &[2.0, 6.0, 16.0, 0.0], 1e-12);
    });
}

#[wasm_bindgen_test]
fn wasm_extract_diagonal_works() {
    spawn_local(async move {
        let device = match GpuDevice::new().await {
            Ok(d) => d,
            Err(_) => return,
        };

        let triplets = vec![
            lca_core::sparse_matrix::Triplete::new(0, 0, 1.0),
            lca_core::sparse_matrix::Triplete::new(1, 1, 2.0),
            lca_core::sparse_matrix::Triplete::new(2, 2, 3.0),
            lca_core::sparse_matrix::Triplete::new(0, 2, 5.0),
        ];
        let cpu = SparseMatrix::from_triplets(3, 3, triplets).unwrap();
        let gpu = device.create_sparse_matrix(&cpu).unwrap();

        let mut dvec = device.create_empty_vector("diag", 3).unwrap();
        device.extract_diagonal(&gpu, &mut dvec).await.unwrap();
        let diag = dvec.read_contents().await.unwrap();
        assert_approx_eq_slice(&diag, &[1.0, 2.0, 3.0], 1e-12);
    });
}
