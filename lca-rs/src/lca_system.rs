use lca_core::{GpuDevice, LcaCoreError, Matrix, SparseMatrix, sparse_matrix::Triplete};
use lca_lsolver::algorithms::{BiCGSTAB, SolveAlgorithm};
use serde::{Deserialize, Serialize};

use crate::{
    LcaMatrix,
    error::{LcaError, Result},
};

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

#[cfg_attr(feature = "wasm", wasm_bindgen)]
#[derive(Debug, Clone)]
pub struct LcaSystem {
    a_matrix: LcaMatrix,
    b_matrix: LcaMatrix,
    c_matrix: LcaMatrix,
    evaluation_demand_process_names: Option<Vec<String>>,
}

#[cfg_attr(feature = "wasm", wasm_bindgen)]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DemandItem {
    pub(crate) product: String,
    pub(crate) amount: f32,
}

impl DemandItem {
    pub fn new(product: String, amount: f32) -> Self {
        Self { product, amount }
    }
}

#[cfg_attr(feature = "wasm", wasm_bindgen)]
impl LcaSystem {
    pub fn new(
        a_matrix: LcaMatrix,
        b_matrix: LcaMatrix,
        c_matrix: LcaMatrix,
        evaluation_demand_process_names: Option<Vec<String>>,
    ) -> Result<Self> {
        Ok(Self {
            a_matrix,
            b_matrix,
            c_matrix,
            evaluation_demand_process_names,
        })
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn a_matrix(&self) -> &LcaMatrix {
        &self.a_matrix
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn b_matrix(&self) -> &LcaMatrix {
        &self.b_matrix
    }
    #[cfg(not(target_arch = "wasm32"))]
    pub fn c_matrix(&self) -> &LcaMatrix {
        &self.c_matrix
    }
    #[cfg(not(target_arch = "wasm32"))]
    pub fn evaluation_demand_process_names(&self) -> &Option<Vec<String>> {
        &self.evaluation_demand_process_names
    }

    pub async fn evaluate(
        &self,
        device: &GpuDevice,
        demand: Option<Vec<DemandItem>>,
        methods: Option<Vec<String>>,
    ) -> Result<Vec<f32>> {
        log::info!("Evaluating LCA system...");
        log::debug!("A matrix: {:?}", self.a_matrix.matrix.dims());
        log::debug!("B matrix: {:?}", self.b_matrix.matrix.dims());
        log::debug!("C matrix: {:?}", self.c_matrix.matrix.dims());

        let f_vec = self.get_demand_vector(demand)?;
        log::debug!("f (demand) vector length: {}", f_vec.len());

        // Create a new C matrix with rows filtered by the methods
        let new_c_matrix = if let Some(methods) = methods {
            let filtered_c_matrix = self.c_matrix.clone().filter_rows(methods)?;
            filtered_c_matrix
        } else {
            self.c_matrix.clone()
        };

        calculate_lca(
            device,
            &self.a_matrix.matrix,
            &self.b_matrix.matrix,
            &new_c_matrix.matrix,
            f_vec,
            1000, // FIXME hardcoded max_iterations
            8e-3, // FIXME hardcoded tolerance
        )
        .await
    }

    fn get_demand_vector(&self, demand: Option<Vec<DemandItem>>) -> Result<Vec<f32>> {
        let demand_items_for_f_vec = match demand {
            Some(actual_demand) if !actual_demand.is_empty() => {
                log::debug!("Using provided explicit demand.");
                actual_demand
            }
            _ => {
                // Covers None or Some(empty_vec)
                if let Some(evaluation_demand_process_names) = &self.evaluation_demand_process_names
                {
                    if !evaluation_demand_process_names.is_empty() {
                        log::info!(
                            "No explicit/empty demand provided; using default demand for evaluation processes."
                        );
                        evaluation_demand_process_names
                            .iter()
                            .map(|name| DemandItem::new(name.clone(), 1.0))
                            .collect::<Vec<DemandItem>>()
                    } else {
                        log::warn!(
                            "No explicit/empty demand provided and no default evaluation processes defined. The demand vector (f_vec) will be all zeros."
                        );
                        Vec::new()
                    }
                } else {
                    log::warn!(
                        "No explicit/empty demand provided and no default evaluation processes defined. The demand vector (f_vec) will be all zeros."
                    );
                    Vec::new()
                }
            }
        };

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
                ))
                .into());
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
    f: Vec<f32>,
    // Solver parameters
    max_iterations: usize,
    tolerance: f32,
) -> Result<Vec<f32>> {
    log::info!("Starting LCA calculation...");

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
    let mut a_gpu = device.create_sparse_matrix(&a_cpu)?;
    let b_gpu = device.create_sparse_matrix(&b_cpu)?;
    let c_gpu = device.create_sparse_matrix(&c_cpu)?;
    let f_gpu = device.create_vector("f_gpu", &f)?;
    log::debug!("Data transferred to GPU.");

    // --- 6. Solve Ax = f for x ---
    log::debug!(
        "Solving Ax = f using BiCGSTAB (with Jacobi Preconditioner) via SolveAlgorithm trait..."
    );
    // Instantiate the solver with parameters, enabling the preconditioner
    let solver = BiCGSTAB::with_params(tolerance, max_iterations, true); // Set use_preconditioner to true
    // TODO: Update the SolveAlgorithm trait in lca-lsolver to accept GpuVector directly for b.
    // For now, read f_gpu back to CPU to match the trait signature (inefficient).
    log::warn!(
        "Reading f vector back from GPU to CPU due to current SolveAlgorithm trait signature. This is inefficient."
    );
    let f_cpu_temp = f_gpu.read_contents().await?;

    let mut solve_result = solver.solve(&device, &a_gpu, &f_cpu_temp).await;
    let mut regularization = 1e-4;
    let mut attempts = 0;

    while let Err(LcaCoreError::BiCGSTABBreakdown { .. }) = solve_result {
        if attempts >= 4 {
            break;
        }

        log::warn!(
            "BiCGSTAB breakdown detected. Applying Tikhonov regularization (lambda = {})...",
            regularization
        );

        // Apply Tikhonov regularization to the diagonal of A
        let a_cpu = a_cpu.clone();
        let new_triplets = a_cpu
            .iter()
            .map(|triplet| {
                let mut triplet = triplet.clone();
                if triplet.row() == triplet.col() {
                    triplet = Triplete::new(
                        triplet.row(),
                        triplet.col(),
                        triplet.value() + regularization,
                    );
                }
                triplet
            })
            .collect::<Vec<Triplete>>();
        let a_cpu = SparseMatrix::from_triplets(a_cpu.rows(), a_cpu.cols(), new_triplets)?;
        // Transfer the modified A matrix back to GPU
        log::debug!("Transferring modified A matrix back to GPU...");
        a_gpu = device.create_sparse_matrix(&a_cpu)?;

        regularization *= 10.0;
        attempts += 1;
        solve_result = solver.solve(&device, &a_gpu, &f_cpu_temp).await;
    }

    // The trait solve method returns the solution vector x directly.
    let solve_result = solve_result?;

    log::info!("System Ax=f solved. Metadata: {:?}", solve_result.metadata);

    // Create x_gpu from the solution vector returned by the solver
    let x_gpu = device.create_vector(
        "x_gpu_solution",
        &solve_result.x, // Use the returned solution data
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
    let h_vec: Vec<f32> = h_gpu.read_contents().await?; // Map error before '?'
    log::debug!(
        "Result vector read back successfully ({} elements).",
        h_vec.len()
    );

    Ok(h_vec)
}
