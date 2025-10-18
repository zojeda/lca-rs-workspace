use crate::{LcaCoreError, SparseMatrix};
use ndarray::{Array1, ArrayView1};
use sprs::{CsMatI, TriMatI};
use std::fmt::Debug;

pub trait Device: Debug {}

#[derive(Debug, Clone, Default)]
pub struct CpuDevice {}
impl Device for CpuDevice {}

impl CpuDevice {
    /// Perform y = A * x where A is in CSR format.
    /// Uses nalgebra-sparse for a stable, optimized CPU implementation.
    pub fn spmv_csr(&self, a: &SparseMatrix, x: &[f64]) -> Result<Vec<f64>, LcaCoreError> {
        let (rows, cols) = a.dims();
        if x.len() != cols {
            return Err(LcaCoreError::InvalidDimensions(format!(
                "Vector x length ({}) must equal A cols ({})",
                x.len(),
                cols
            )));
        }

        // sprs expects indptr (row_ptr), indices (col_indices) as index type; use usize variant
        // Build via triplets to ensure indices are sorted and duplicates are coalesced
        let mut tri: TriMatI<f64, usize> = TriMatI::with_capacity((rows, cols), a.nnz());
        for r in 0..rows {
            let start = a.row_ptr()[r];
            let end = a.row_ptr()[r + 1];
            for i in start..end {
                let c = a.col_indices()[i];
                let v = a.values()[i];
                tri.add_triplet(r, c, v);
            }
        }
        let csr: CsMatI<f64, usize> = tri.to_csr();

        let x_view: ArrayView1<'_, f64> = ArrayView1::from(x);
        let y_arr: Array1<f64> = &csr * &x_view;
        Ok(y_arr.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::sparse_matrix::Triplete;

    #[test]
    fn test_cpu_spmv_simple() {
        // A = [[1, 0, 0],
        //      [2, 3, 0],
        //      [0, 0, 4]]
        let a = SparseMatrix::from_triplets(
            3,
            3,
            vec![
                Triplete::new(0, 0, 1.0),
                Triplete::new(1, 0, 2.0),
                Triplete::new(1, 1, 3.0),
                Triplete::new(2, 2, 4.0),
            ],
        )
        .unwrap();
        let x = vec![1.0, 2.0, 3.0];
        let cpu = CpuDevice::default();
        let y = cpu.spmv_csr(&a, &x).expect("spmv failed");
        assert_eq!(y, vec![1.0, 8.0, 12.0]);
    }
}

pub mod gpu;
pub use gpu::{GpuDevice, TransferStats};
// Removed the placeholder traits module
// pub mod traits {
//     use super::Device;
//     pub trait ComputeDevice: Device {}
// }
