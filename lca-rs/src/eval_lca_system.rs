use std::collections::HashSet;

use lca_core::{sparse_matrix::Triplete, DemandItem, GpuDevice, LcaMatrix, LcaSystem, SparseMatrix}; // Removed Matrix
use lca_lsolver::algorithms::{BiCGSTAB, SolveAlgorithm};

use crate::error::{LcaError, Result};

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

pub struct EvalLCASystem {
    pub name: String,
    pub a_matrix: LcaMatrix,
    pub b_matrix: LcaMatrix,
    pub c_matrix: LcaMatrix,
    pub demand: Vec<DemandItem>,
    pub evaluation_methods: Vec<String>,
}

impl TryFrom<LcaSystem> for EvalLCASystem {
    type Error = LcaError;

    fn try_from(lca_system: LcaSystem) -> Result<Self> {
        // if there are required links, throw LinkError
        if !lca_system.a_links.is_empty() {
            return Err(LcaError::LinkError(format!(
                "LCA system '{}' has A links: {:?}",
                lca_system.name, lca_system.a_links
            )));
        }
        if !lca_system.b_links.is_empty() {
            return Err(LcaError::LinkError(format!(
                "LCA system '{}' has B links: {:?}",
                lca_system.name, lca_system.b_links
            )));
        }
        if !lca_system.c_links.is_empty() {
            return Err(LcaError::LinkError(format!(
                "LCA system '{}' has C links: {:?}",
                lca_system.name, lca_system.c_links
            )));
        }
        let a_matrix = lca_system.a_matrix;
        let b_matrix = lca_system.b_matrix;
        let c_matrix = lca_system.c_matrix;

        let evaluation_demand = if let Some(evaluation_demand) = lca_system.evaluation_demand {
            if evaluation_demand.is_empty() {
                return Err(LcaError::DimensionError(format!(
                    "LCA system '{}' has empty evaluation demand",
                    lca_system.name
                )));
            } else {
                for demand in &evaluation_demand {
                    if !a_matrix.col_ids.contains(&demand.product) {
                        return Err(LcaError::DimensionError(format!(
                            "LCA system '{}' has evaluation demand '{}' not found in A matrix column IDs",
                            lca_system.name, demand.product
                        )));
                    }
                }
                evaluation_demand
            }
        } else {
            Vec::new()
        };

        let evaluation_methods = if let Some(evaluation_methods) = lca_system.evaluation_methods {
            if evaluation_methods.is_empty() {
                return Err(LcaError::DimensionError(format!(
                    "LCA system '{}' has empty evaluation methods",
                    lca_system.name
                )));
            } else {
                for method in &evaluation_methods {
                    if !c_matrix.row_ids.contains(method) {
                        return Err(LcaError::DimensionError(format!(
                            "LCA system '{}' has evaluation method '{}' not found in C matrix row IDs",
                            lca_system.name, method
                        )));
                    }
                }
            }
            evaluation_methods
        } else {
            Vec::new()
        };

        Ok(Self {
            name: lca_system.name,
            a_matrix,
            b_matrix,
            c_matrix,
            demand: evaluation_demand,
            evaluation_methods,
        })
    }
}

impl EvalLCASystem {
    pub fn with_demand(self, demand: Vec<DemandItem>) -> Self {
        Self { demand, ..self }
    }
    pub fn with_evaluation_methods(self, evaluation_methods: Vec<String>) -> Self {
        Self {
            evaluation_methods,
            ..self
        }
    }
    pub async fn evaluate(&self, device: &GpuDevice) -> Result<Vec<f64>> {
        log::info!("Evaluating LCA system...");
        log::debug!("A matrix: {:?}", self.a_matrix.matrix.dims());
        log::debug!("B matrix: {:?}", self.b_matrix.matrix.dims());
        log::debug!("C matrix: {:?}", self.c_matrix.matrix.dims());
        
        if self.demand.is_empty() {
            return Err(LcaError::DimensionError(format!(
                "LCA system '{}' has no demand items to process",
                self.name
            )));
        }
        if self.evaluation_methods.is_empty() {
            return Err(LcaError::DimensionError(format!(
                "LCA system '{}' has no evaluation methods to process",
                self.name
            )));
        }
        // filter C matrix to only include rows for the evaluation methods
        let c_matrix = self.filtered_c_matrix()?;

        let f_vec = self.get_demand_vector(&self.demand)?;
        log::info!("f (demand) vector length: {}", f_vec.len());

        calculate_lca(
            device,
            &self.a_matrix.matrix,
            &self.b_matrix.matrix,
            &c_matrix.matrix,
            f_vec,
            1000, // FIXME hardcoded max_iterations
            8e-8,
        )
        .await
    }

    // FIXME ugly, this should be a view of the sparse matrix, without copying data
    fn filtered_c_matrix(&self) -> Result<LcaMatrix> {
        let new_c_matrix = self.c_matrix.filter_rows(&self.evaluation_methods)?;
        Ok(new_c_matrix)
    }

    fn get_demand_vector(&self, demand: &[DemandItem]) -> Result<Vec<f64>> {
        let demand_items_for_f_vec = demand;

        if demand_items_for_f_vec.is_empty() && self.a_matrix.matrix.rows() > 0 {
            log::warn!(
                "The list of demand items to process is empty. The demand vector (f_vec) will be all zeros."
            );
        }
        let mut f_vec = vec![0.0; self.a_matrix.matrix.rows()];

        for demand in demand_items_for_f_vec {
            if let Some(index) = self
                .a_matrix
                .col_ids
                .iter()
                .position(|id| id == &demand.product)
            {
                f_vec[index] = demand.amount;
            } else {
                return Err(LcaError::DimensionError(format!(
                    "Demand product '{}' not found in A matrix column IDs",
                    demand.product
                )));
            }
        }
        Ok(f_vec)
    }
}

// --- Main LCA Calculation Function ---
async fn calculate_lca(
    device: &GpuDevice,
    // A Matrix (CSR)
    a_cpu: &SparseMatrix,
    // B Matrix (CSR)
    b_cpu: &SparseMatrix,
    // C Matrix (CSR)
    c_cpu: &SparseMatrix,
    // f Vector (Dense)
    f: Vec<f64>,
    // Solver parameters
    max_iterations: usize,
    tolerance: f64, // Will be 8e-6 from evaluate, or 8e-4 from test_calculate_lca_system
) -> Result<Vec<f64>> {
    log::info!("Starting LCA calculation with tolerance: {}", tolerance); // Use log::info

    // --- 1. Dimension Checks ---
    log::debug!("Performing dimension checks...");
    // Check A * x = f
    if a_cpu.cols() != f.len() {
        return Err(LcaError::DimensionError(format!(
            "A matrix columns ({}) must match f vector length ({})",
            a_cpu.cols(),
            f.len()
        ))
        .into());
    }
    // Check B * x = g
    if b_cpu.cols() != a_cpu.cols() {
        // B's cols must match x's length (which is A's cols)
        return Err(LcaError::DimensionError(format!(
            "B matrix columns ({}) must match A matrix columns ({})",
            b_cpu.cols(),
            a_cpu.cols()
        ))
        .into());
    }
    // Check C * g = h
    if c_cpu.cols() != b_cpu.rows() {
        // C's cols must match g's length (which is B's rows)
        return Err(LcaError::DimensionError(format!(
            "C matrix columns ({}) must match B matrix rows ({})",
            c_cpu.cols(),
            b_cpu.rows()
        ))
        .into());
    }
    log::debug!("Dimension checks passed.");

    // --- 5. Transfer Data to GPU ---
    log::debug!("Transferring data to GPU...");
    // Use device.create_sparse_matrix and device.create_vector
    let a_gpu = device.create_sparse_matrix(&a_cpu)?;
    let b_gpu = device.create_sparse_matrix(&b_cpu)?;
    let c_gpu = device.create_sparse_matrix(&c_cpu)?;
    let f_gpu = device.create_vector("f_gpu", &f)?;
    log::debug!("Data transferred to GPU.");
    // TODO: Update the SolveAlgorithm trait in lca-lsolver to accept GpuVector directly for b.
    // For now, read f_gpu back to CPU to match the trait signature (inefficient).
    log::warn!(
        "Reading f vector back from GPU to CPU due to current SolveAlgorithm trait signature. This is inefficient."
    );
    let f_cpu_temp = f_gpu.read_contents().await?;

    // --- 6. Solve Ax = f for x ---

    let solver_bicgstab = BiCGSTAB::with_params(tolerance, max_iterations, true);

    log::info!("Attempting to solve Ax=f with BiCGSTAB...");

    let solution_x: Vec<f64>;

    log::info!("Attempting to solve Ax=f with BiCGSTAB..."); // Use log::info
    match solver_bicgstab.solve(&device, &a_gpu, &f_cpu_temp).await {
        Ok(result) => {
            log::info!("BiCGSTAB succeeded. Metadata: {:?}", result.metadata); // Use log::info
            solution_x = result.x;
        }
        Err(other_bicgstab_core_err) => {
            // Other LcaCoreError from BiCGSTAB
            log::error!(
                "BiCGSTAB failed with unexpected core error: {:?}",
                other_bicgstab_core_err
            ); // Use log::error
            return Err(LcaError::LcaCoreError(other_bicgstab_core_err));
        }
    }

    log::info!("System Ax=f solved."); // Use log::info

    // Create x_gpu from the solution vector returned by the solver
    let x_gpu = device.create_vector(
        "x_gpu_solution",
        &solution_x, // Use the obtained solution data
    )?;

    // --- 7. Calculate g = Bx ---
    log::debug!("Calculating g = Bx...");
    // Use device.create_empty_vector for the output vector g
    let mut g_gpu = device.create_empty_vector("g_gpu", b_gpu.rows())?;
    b_gpu
        .spmv(&x_gpu, &mut g_gpu)
        .await
        .map_err(LcaError::from)?; // Map error before '?' - Keep for spmv
    log::debug!("Calculated g = Bx.");

    // --- 8. Calculate h = Cg ---
    log::debug!("Calculating h = Cg...");
    // Us e device.create_empty_vector for the output vector h
    let mut h_gpu = device.create_empty_vector("h_gpu", c_gpu.rows())?;
    c_gpu
        .spmv(&g_gpu, &mut h_gpu)
        .await
        .map_err(LcaError::from)?; // Map error before '?' - Keep for spmv
    log::debug!("Calculated h = Cg.");

    // --- 9. Read Result Back ---
    log::debug!("Reading final result vector h back from GPU...");
    // Use read_contents with internal context (no context argument needed)
    let h_vec: Vec<f64> = h_gpu.read_contents().await?; // Map error before '?'
    log::debug!(
        "Result vector read back successfully ({} elements).",
        h_vec.len()
    );

    Ok(h_vec)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::LcaError;
    use lca_core::{InterSystemLink, SparseMatrix, sparse_matrix::Triplete};
    use std::collections::HashSet;

    #[test]
    fn test_calculate_lca_system() {
        // A[0,0] = 1.0 (Bike output)
        // A[1,1] = 1.0 (Carbon Fibre output)
        // A[2,2] = 1.0 (Natural Gas output)
        // A[1,0] = -2.5 (Carbon Fibre input to Bike)
        // A[2,1] = -237.0 (Natural Gas input to Carbon Fibre)
        // B[0,1] = 26.6 (CO2 emission from Carbon Fibre)
        // C[0,0] = 1.0 (GWP100 impact from CO2)
        let a = SparseMatrix::from_triplets(
            3,
            3,
            vec![
                Triplete::new(0, 0, 1.0),
                Triplete::new(1, 1, 1.0),
                Triplete::new(2, 2, 1.0),
                Triplete::new(1, 0, -2.5),
                Triplete::new(2, 1, -237.0),
            ],
        )
        .unwrap();
        let b = SparseMatrix::from_triplets(1, 3, vec![Triplete::new(0, 1, 26.6)]).unwrap();

        let c = SparseMatrix::from_triplets(1, 1, vec![Triplete::new(0, 0, 1.0)]).unwrap();

        let f = vec![1.0, 0.0, 0.0];
        pollster::block_on(async {
            let device = GpuDevice::new().await.unwrap();
            let result = calculate_lca(&device, &a, &b, &c, f, 1000, 1e-12)
                .await
                .expect("LCA calculation failed");
            assert_eq!(result.len(), 1);
            let expected_co2 = 66.5;
            assert!(
                (result[0] - expected_co2).abs() < 1e-3,
                "LCA result for GWP100 is incorrect. Expected: {}, Got: {}",
                expected_co2,
                result[0]
            );
        })
    }
}
