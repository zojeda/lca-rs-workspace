#[test]
fn test_matrix_read_back() {
    // pollster::block_on(async {
    //     let context = setup_context().await;

    //     let rows = 2;
    //     let cols = 3;
    //     let original_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

    //     // Create matrix on GPU
    //     let matrix = DenseMatrixf32::from_data(context.clone(), &original_data, rows, cols)
    //         .expect("Failed to create matrix from data");

    //     assert_eq!(matrix.rows(), rows);
    //     assert_eq!(matrix.cols(), cols);

    //     // Read data back from GPU
    //     let read_back_data = matrix
    //         .read_back()
    //         .await
    //         .expect("Failed to read matrix back");

    //     // Verify dimensions and content
    //     assert_eq!(
    //         read_back_data.len(),
    //         rows * cols,
    //         "Read-back data length mismatch"
    //     );
    //     assert_eq!(
    //         read_back_data, original_data,
    //         "Read-back data content mismatch"
    //     );

    //     // Optional: Check with a small tolerance for floating-point comparisons if needed
    //     // for (original, read) in original_data.iter().zip(read_back_data.iter()) {
    //     //     assert!((original - read).abs() < f32::EPSILON, "Floating point mismatch");
    //     // }
    // });
}

// TODO: Add more tests for edge cases (e.g., empty matrix, large matrix)
// TODO: Add tests for actual computations once implemented.
