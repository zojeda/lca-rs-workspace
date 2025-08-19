//! PARDISO-based direct solver (native only).
//!
//! This module provides a CPU direct sparse solver backed by Intel MKL PARDISO via
//! the `pardiso-wrapper` crate. It is gated behind the `pardiso` feature and is not
//! available for `wasm32` targets.

#![cfg(all(feature = "pardiso", not(target_arch = "wasm32")))]

use crate::algorithms::{SolveAlgorithm, SolveResult};
use lca_core::{devices::CpuDevice, error::LcaCoreError, sparse_matrix::SparseMatrix};

/// Matrix structural type for PARDISO factorization.
#[derive(Debug, Clone, Copy)]
pub enum PardisoMatrixType {
    /// Symmetric Positive Definite (SPD)
    Spd,
    /// Symmetric Indefinite
    SymIndef,
    /// General Unsymmetric
    General,
}

impl PardisoMatrixType {
    fn mtype_code(self) -> i32 {
        match self {
            PardisoMatrixType::Spd => 2,          // real and symmetric positive definite
            PardisoMatrixType::SymIndef => -2,    // real and symmetric indefinite
            PardisoMatrixType::General => 11,     // real and unsymmetric
        }
    }
}

/// Configuration for the PARDISO direct solver.
#[derive(Debug, Clone)]
pub struct PardisoConfig {
    pub matrix_type: PardisoMatrixType,
    pub max_threads: Option<usize>,
}

impl Default for PardisoConfig {
    fn default() -> Self {
        Self { matrix_type: PardisoMatrixType::General, max_threads: None }
    }
}

/// Metadata returned by the PARDISO solver.
#[derive(Debug, Clone, Default)]
pub struct PardisoMetadata {
    pub n: usize,
    pub nnz: usize,
    pub iterations: Option<usize>,
}

/// Direct sparse solver using Intel MKL PARDISO.
#[derive(Debug, Clone)]
pub struct PardisoDirect {
    pub config: PardisoConfig,
}

impl PardisoDirect {
    pub fn new(config: PardisoConfig) -> Self { Self { config } }
}

impl SolveAlgorithm<CpuDevice, SparseMatrix> for PardisoDirect {
    type Value = f64;
    type Metadata = PardisoMetadata;

    fn solve(
        &self,
        _device: &CpuDevice,
        a: &SparseMatrix,
        b: &[Self::Value],
    ) -> impl std::future::Future<Output = Result<SolveResult<Self::Value, Self::Metadata>, LcaCoreError>> + Send {
    use pardiso_wrapper::{MatrixType, MKLPardisoSolver, PardisoInterface};
    let res = self.validate_inputs(a, b);
    let config = self.config.clone();
    let (n, _m) = a.dims();
    // Prepare CSR in 1-based indexing as expected by MKL PARDISO (defer error to async).
    let csr_conv = csr_usize_to_csr_i32_1based(a);
    let mut rhs = b.to_vec();

        async move {
            res?;
            let mtype = match config.matrix_type {
                PardisoMatrixType::Spd => MatrixType::RealSymmetricPositiveDefinite,
                PardisoMatrixType::SymIndef => MatrixType::RealSymmetricIndefinite,
                PardisoMatrixType::General => MatrixType::RealNonsymmetric,
            };

            // Unwrap CSR conversion result within async context
            let (ia, ja, a_vals) = csr_conv?;

            let mut solver = MKLPardisoSolver::new().map_err(|e| LcaCoreError::LinkError(format!("PARDISO init error: {e}")))?;
            solver.set_matrix_type(mtype);
            solver.pardisoinit().map_err(|e| LcaCoreError::LinkError(format!("PARDISO init error: {e}")))?;
            if let Some(t) = config.max_threads { let _ = solver.set_num_threads(t as i32); }

            // 1) Analysis
            solver.set_phase(pardiso_wrapper::Phase::Analysis);
            solver.pardiso(&a_vals, &ia, &ja, &mut [], &mut [], n as i32, 1)
                .map_err(|e| LcaCoreError::LinkError(format!("PARDISO analysis error: {e}")))?;

            // 2) Numeric factorization
            solver.set_phase(pardiso_wrapper::Phase::NumFact);
            solver.pardiso(&a_vals, &ia, &ja, &mut rhs.clone(), &mut vec![0.0; b.len()], n as i32, 1)
                .map_err(|e| LcaCoreError::LinkError(format!("PARDISO factorization error: {e}")))?;

            // 3) Solve
            let mut x = vec![0.0f64; b.len()];
            solver.set_phase(pardiso_wrapper::Phase::SolveIterativeRefine);
            solver.pardiso(&a_vals, &ia, &ja, &mut rhs, &mut x, n as i32, 1)
                .map_err(|e| LcaCoreError::LinkError(format!("PARDISO solve error: {e}")))?;

            Ok(SolveResult { x, metadata: PardisoMetadata { n, nnz: a.nnz(), iterations: None } })
        }
    }
}

fn csr_usize_to_csr_i32_1based(a: &SparseMatrix) -> Result<(Vec<i32>, Vec<i32>, Vec<f64>), LcaCoreError> {
    let n = a.rows();
    if a.cols() != n {
        return Err(LcaCoreError::InvalidDimensions("Matrix must be square".to_string()));
    }
    let nnz = a.nnz();
    // Convert row_ptr and col_indices from 0-based usize to 1-based i32
    let mut ia = Vec::with_capacity(n + 1);
    for rp in a.row_ptr() {
        let v = i32::try_from(*rp).map_err(|_| LcaCoreError::InvalidDimensions("row_ptr index overflow".into()))?;
        ia.push(v + 1);
    }
    let mut ja = Vec::with_capacity(nnz);
    for ci in a.col_indices() {
        let v = i32::try_from(*ci).map_err(|_| LcaCoreError::InvalidDimensions("col index overflow".into()))?;
        ja.push(v + 1);
    }
    Ok((ia, ja, a.values().to_vec()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use lca_core::sparse_matrix::Triplete;

    #[test]
    #[ignore]
    fn test_pardiso_small_general() {
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
        .unwrap();
        let b = vec![3.0, 2.0, 1.0];
        let cfg = PardisoConfig { matrix_type: PardisoMatrixType::General, max_threads: Some(1) };
        let solver = PardisoDirect::new(cfg);
        let cpu = CpuDevice::default();
        let out = pollster::block_on(async { solver.solve(&cpu, &a, &b).await });
        match out {
            Ok(res) => {
                assert_eq!(res.x.len(), 3);
                // expected approximately: [1.0, 1.0, 0.6666667]
                assert!((res.x[0] - 1.0).abs() < 1e-9);
                assert!((res.x[1] - 1.0).abs() < 1e-9);
                assert!((res.x[2] - (2.0/3.0)).abs() < 1e-9);
            }
            Err(e) => {
                // If MKL not available, this might fail at runtime; mark as ignored by default.
                eprintln!("PARDISO unit test failed: {e}");
                panic!("PARDISO not available or failed: {e}");
            }
        }
    }
}
