#[cfg(all(feature = "pardiso", not(target_arch = "wasm32")))]
use lca_core::devices::CpuDevice;
use lca_core::{
    DemandItem, GpuDevice, GpuVector, LcaMatrix, LcaSystem, SparseMatrix, SparseMatrixGpu,
    models::lca_system::Demand,
};
#[cfg(all(feature = "pardiso", not(target_arch = "wasm32")))]
use lca_lsolver::algorithms::pardiso_direct::{PardisoConfig, PardisoDirect, PardisoMatrixType};
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
    /// Optional list of multiple demand sets for MultiLCA runs
    pub demands: Vec<Demand>,
    pub evaluation_methods: Vec<String>,
    /// Solver backend selection (GPU BiCGSTAB by default)
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    solver_backend: SolverBackend,
    /// Device backend for Bx and Cg multiplications. If None, defaults based on solver backend at evaluate time.
    device_backend: Option<DeviceBackend>,
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

        // Seed `demands` list based on the single evaluation_demand, computed before moving it
        let seeded_demands = if !evaluation_methods.is_empty() && !evaluation_demand.is_empty() {
            vec![evaluation_demand.clone()]
        } else {
            Vec::new()
        };

        Ok(Self {
            name: lca_system.name,
            a_matrix,
            b_matrix,
            c_matrix,
            demand: evaluation_demand,
            demands: seeded_demands,
            evaluation_methods,
            solver_backend: SolverBackend::GpuBiCGSTAB,
            device_backend: None,
        })
    }
}

impl EvalLCASystem {
    pub fn with_demand(self, demand: Demand) -> Self {
        // Keep backward compatibility for single-demand API
        let mut demands = self.demands;
        demands.clear();
        demands.push(demand.clone());
        Self {
            demand,
            demands,
            ..self
        }
    }
    /// Set multiple demands for MultiLCA evaluation (replaces any previous list)
    pub fn with_demands(self, demands: Vec<Demand>) -> Self {
        // Also set the first demand (if any) as the single-demand field for backwards compatibility
        let demand = demands.first().cloned().unwrap_or_default();
        Self {
            demands,
            demand,
            ..self
        }
    }
    pub fn with_evaluation_methods(self, evaluation_methods: Vec<String>) -> Self {
        Self {
            evaluation_methods,
            ..self
        }
    }
    /// Prefer CPU PARDISO direct solver instead of GPU BiCGSTAB (native only)
    #[cfg(all(feature = "pardiso", not(target_arch = "wasm32")))]
    pub fn with_cpu_pardiso(
        self,
        matrix_type: Option<PardisoMatrixType>,
        max_threads: Option<usize>,
    ) -> Self {
        // If device backend not explicitly set later, we'll default to CPU device at evaluate time.
        Self {
            solver_backend: SolverBackend::CpuPardiso {
                matrix_type,
                max_threads,
            },
            ..self
        }
    }
    /// Provide an explicit GPU device to be used for Bx/Cg; overrides any internal device.
    pub fn with_gpu_device(mut self, device: GpuDevice) -> Self {
        self.device_backend = Some(DeviceBackend::Gpu(device));
        self
    }
    /// Provide an explicit CPU device to be used for Bx/Cg; overrides any internal device.
    #[cfg(all(feature = "pardiso", not(target_arch = "wasm32")))]
    pub fn with_cpu_device(mut self, device: CpuDevice) -> Self {
        self.device_backend = Some(DeviceBackend::Cpu(device));
        self
    }
    /// Evaluate using an internal device backend; no need to pass a device.
    pub async fn evaluate(&self) -> Result<Vec<f64>> {
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

        // Pick device backend: use provided one or default based on solver backend
        let device_backend = self.ensure_device_backend().await?;

        calculate_lca_with_backend(
            &device_backend,
            &self.a_matrix.matrix,
            &self.b_matrix.matrix,
            &c_matrix.matrix,
            f_vec,
            1000,
            8e-8,
            &self.solver_backend,
        )
        .await
    }

    /// Evaluate the system for multiple demand sets, returning one impact vector per demand
    pub async fn evaluate_multi(&self) -> Result<Vec<Vec<f64>>> {
        log::info!(
            "Evaluating MultiLCA for {} demands...",
            if !self.demands.is_empty() {
                self.demands.len()
            } else {
                1
            }
        );
        log::debug!("A matrix: {:?}", self.a_matrix.matrix.dims());
        log::debug!("B matrix: {:?}", self.b_matrix.matrix.dims());
        log::debug!("C matrix: {:?}", self.c_matrix.matrix.dims());

        if self.evaluation_methods.is_empty() {
            return Err(LcaError::DimensionError(format!(
                "LCA system '{}' has no evaluation methods to process",
                self.name
            )));
        }

        // Determine which demand sets to use: prefer `self.demands`, else fall back to single `self.demand`
        let demand_sets: Vec<Demand> = if !self.demands.is_empty() {
            self.demands.clone()
        } else if !self.demand.is_empty() {
            vec![self.demand.clone()]
        } else {
            return Err(LcaError::DimensionError(format!(
                "LCA system '{}' has no demand items to process",
                self.name
            )));
        };

        // Filter C matrix once for all evaluations
        let c_matrix = self.filtered_c_matrix()?;

        // Ensure device backend for Bx/Cg
        let device_backend = self.ensure_device_backend().await?;
        // Prepare GPU resources only if needed
        let mut gpu_preload: Option<GpuPreload> = None;
        if let DeviceBackend::Gpu(ref device) = device_backend {
            let a_gpu = device.create_sparse_matrix(&self.a_matrix.matrix)?;
            let b_gpu = device.create_sparse_matrix(&self.b_matrix.matrix)?;
            let c_gpu = device.create_sparse_matrix(&c_matrix.matrix)?;
            let x_gpu = device.create_empty_vector("x_gpu_solution_multi", a_gpu.cols())?;
            let g_gpu = device.create_empty_vector("g_gpu", b_gpu.rows())?;
            let h_gpu = device.create_empty_vector("h_gpu", c_gpu.rows())?;
            gpu_preload = Some(GpuPreload {
                device: device.clone(),
                a_gpu,
                b_gpu,
                c_gpu,
                x_gpu,
                g_gpu,
                h_gpu,
            });
        }

        // Reuse a single solver instance for GPU
        let solver_bicgstab = BiCGSTAB::with_params(8e-8, 1000, true);

        let mut results: Vec<Vec<f64>> = Vec::with_capacity(demand_sets.len());
        for (i, demand) in demand_sets.iter().enumerate() {
            log::info!(
                "Evaluating demand set {} ({} items)...",
                i + 1,
                demand.len()
            );
            let f_vec = self.get_demand_vector(demand)?;
            let res = match (&device_backend, &mut gpu_preload) {
                (DeviceBackend::Gpu(_), Some(preload)) => {
                    calculate_lca_preloaded(
                        &preload.device,
                        &preload.a_gpu,
                        &preload.b_gpu,
                        &preload.c_gpu,
                        &f_vec,
                        &solver_bicgstab,
                        &mut preload.x_gpu,
                        &mut preload.g_gpu,
                        &mut preload.h_gpu,
                    )
                    .await?
                }
                #[cfg(all(feature = "pardiso", not(target_arch = "wasm32")))]
                (DeviceBackend::Cpu(cpu), _) => {
                    // CPU solve path depending on solver backend
                    let x = match self.solver_backend {
                        SolverBackend::GpuBiCGSTAB => {
                            // Fallback: perform solve on GPU once for solution only
                            let gpu = GpuDevice::new().await.map_err(LcaError::LcaCoreError)?;
                            let a_gpu = gpu.create_sparse_matrix(&self.a_matrix.matrix)?;

                            solver_bicgstab
                                .solve(&gpu, &a_gpu, &f_vec)
                                .await
                                .map_err(LcaError::LcaCoreError)?
                                .x
                        }
                        #[cfg(all(feature = "pardiso", not(target_arch = "wasm32")))]
                        SolverBackend::CpuPardiso {
                            matrix_type,
                            max_threads,
                        } => {
                            let cfg = PardisoConfig {
                                matrix_type: matrix_type.unwrap_or(PardisoMatrixType::General),
                                max_threads,
                            };
                            let solver = PardisoDirect::new(cfg);
                            solver
                                .solve(cpu, &self.a_matrix.matrix, &f_vec)
                                .await
                                .map_err(LcaError::LcaCoreError)?
                                .x
                        }
                    };
                    let g = cpu
                        .spmv_csr(&self.b_matrix.matrix, &x)
                        .map_err(LcaError::LcaCoreError)?;

                    cpu.spmv_csr(&c_matrix.matrix, &g)
                        .map_err(LcaError::LcaCoreError)?
                }
                _ => unreachable!("Unsupported backend combination"),
            };
            results.push(res);
        }

        Ok(results)
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
#[derive(Debug, Clone)]
enum SolverBackend {
    GpuBiCGSTAB,
    #[cfg(all(feature = "pardiso", not(target_arch = "wasm32")))]
    CpuPardiso {
        matrix_type: Option<PardisoMatrixType>,
        max_threads: Option<usize>,
    },
}

#[derive(Clone)]
enum DeviceBackend {
    Gpu(GpuDevice),
    #[cfg(all(feature = "pardiso", not(target_arch = "wasm32")))]
    Cpu(CpuDevice),
}

async fn calculate_lca_with_backend(
    device_backend: &DeviceBackend,
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
    solver_backend: &SolverBackend,
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
        )));
    }
    // Check B * x = g
    if b_cpu.cols() != a_cpu.cols() {
        // B's cols must match x's length (which is A's cols)
        return Err(LcaError::DimensionError(format!(
            "B matrix columns ({}) must match A matrix columns ({})",
            b_cpu.cols(),
            a_cpu.cols()
        )));
    }
    // Check C * g = h
    if c_cpu.cols() != b_cpu.rows() {
        // C's cols must match g's length (which is B's rows)
        return Err(LcaError::DimensionError(format!(
            "C matrix columns ({}) must match B matrix rows ({})",
            c_cpu.cols(),
            b_cpu.rows()
        )));
    }
    log::debug!("Dimension checks passed.");

    // Prepare GPU matrices only if needed for Bx/Cg
    let mut gpu_mats: Option<(GpuDevice, SparseMatrixGpu, SparseMatrixGpu, SparseMatrixGpu)> = None;
    if let DeviceBackend::Gpu(device) = device_backend {
        log::debug!("Transferring matrices to GPU...");
        let a_gpu = device.create_sparse_matrix(a_cpu)?;
        let b_gpu = device.create_sparse_matrix(b_cpu)?;
        let c_gpu = device.create_sparse_matrix(c_cpu)?;
        log::debug!("Matrices transferred to GPU.");
        gpu_mats = Some((device.clone(), a_gpu, b_gpu, c_gpu));
    }

    // --- 6. Solve Ax = f for x ---

    let solution_x: Vec<f64> = match solver_backend {
        SolverBackend::GpuBiCGSTAB => {
            let maybe_refs = gpu_mats
                .as_ref()
                .map(|(device, a_gpu, _, _)| (device, a_gpu));
            let solver_bicgstab = BiCGSTAB::with_params(tolerance, max_iterations, true);
            log::info!("Attempting to solve Ax=f with BiCGSTAB...");
            let solve_res = match maybe_refs {
                Some((device, a_gpu)) => solver_bicgstab.solve(device, a_gpu, &f).await,
                None => {
                    let device = GpuDevice::new().await.map_err(LcaError::LcaCoreError)?;
                    let a_gpu = device.create_sparse_matrix(a_cpu)?;
                    solver_bicgstab.solve(&device, &a_gpu, &f).await
                }
            };
            match solve_res {
                Ok(result) => {
                    log::info!("BiCGSTAB succeeded. Metadata: {:?}", result.metadata);
                    result.x
                }
                Err(e) => {
                    log::error!("BiCGSTAB failed: {:?}", e);
                    return Err(LcaError::LcaCoreError(e));
                }
            }
        }
        #[cfg(all(feature = "pardiso", not(target_arch = "wasm32")))]
        SolverBackend::CpuPardiso {
            matrix_type,
            max_threads,
        } => {
            log::info!("Attempting to solve Ax=f with CPU PARDISO...");
            let cfg = PardisoConfig {
                matrix_type: matrix_type.unwrap_or(PardisoMatrixType::General),
                max_threads: *max_threads,
            };
            let solver = PardisoDirect::new(cfg);
            let cpu = CpuDevice::default();
            match solver.solve(&cpu, a_cpu, &f).await {
                Ok(result) => result.x,
                Err(e) => return Err(LcaError::LcaCoreError(e)),
            }
        }
    };

    log::info!("System Ax=f solved."); // Use log::info

    // Compute g and h using selected device backend
    match device_backend {
        DeviceBackend::Gpu(device) => {
            let (_, _a_gpu, b_gpu, c_gpu) = gpu_mats.expect("GPU mats must exist for GPU backend");
            // upload x
            let x_gpu = device.create_vector("x_gpu_solution", &solution_x)?;
            // g = Bx
            let mut g_gpu = device.create_empty_vector("g_gpu", b_gpu.rows())?;
            b_gpu
                .spmv(&x_gpu, &mut g_gpu)
                .await
                .map_err(LcaError::from)?;
            // h = Cg
            let mut h_gpu = device.create_empty_vector("h_gpu", c_gpu.rows())?;
            c_gpu
                .spmv(&g_gpu, &mut h_gpu)
                .await
                .map_err(LcaError::from)?;
            // read back
            let h_vec: Vec<f64> = h_gpu.read_contents().await?;
            Ok(h_vec)
        }
        #[cfg(all(feature = "pardiso", not(target_arch = "wasm32")))]
        DeviceBackend::Cpu(cpu) => {
            let g = cpu
                .spmv_csr(b_cpu, &solution_x)
                .map_err(LcaError::LcaCoreError)?;
            let h = cpu.spmv_csr(c_cpu, &g).map_err(LcaError::LcaCoreError)?;
            Ok(h)
        }
    }
}

// Optimized path: preloaded GPU matrices and reusable GPU vectors
async fn calculate_lca_preloaded(
    device: &GpuDevice,
    a_gpu: &SparseMatrixGpu,
    b_gpu: &SparseMatrixGpu,
    c_gpu: &SparseMatrixGpu,
    f: &[f64],
    solver_bicgstab: &BiCGSTAB,
    x_gpu: &mut GpuVector,
    g_gpu: &mut GpuVector,
    h_gpu: &mut GpuVector,
) -> Result<Vec<f64>> {
    // Solve Ax=f on GPU-backed solver, returns CPU solution
    let solution_x = match solver_bicgstab.solve(device, a_gpu, f).await {
        Ok(result) => result.x,
        Err(e) => return Err(LcaError::LcaCoreError(e)),
    };

    // Upload x to GPU without reallocating buffers
    x_gpu
        .write_contents(&solution_x)
        .await
        .map_err(LcaError::from)?;

    // g = Bx
    b_gpu.spmv(x_gpu, g_gpu).await.map_err(LcaError::from)?;
    // h = Cg
    c_gpu.spmv(g_gpu, h_gpu).await.map_err(LcaError::from)?;
    // Read back h
    let h_vec = h_gpu.read_contents().await?;
    Ok(h_vec)
}

// Small helper to hold preloaded GPU resources for multi evaluation
struct GpuPreload {
    device: GpuDevice,
    a_gpu: SparseMatrixGpu,
    b_gpu: SparseMatrixGpu,
    c_gpu: SparseMatrixGpu,
    x_gpu: GpuVector,
    g_gpu: GpuVector,
    h_gpu: GpuVector,
}

impl EvalLCASystem {
    async fn ensure_device_backend(&self) -> Result<DeviceBackend> {
        if let Some(ref backend) = self.device_backend {
            return Ok(backend.clone());
        }
        match self.solver_backend {
            SolverBackend::GpuBiCGSTAB => {
                // Prefer GPU device
                let gpu = GpuDevice::new().await.map_err(LcaError::LcaCoreError)?;
                Ok(DeviceBackend::Gpu(gpu))
            }
            #[cfg(all(feature = "pardiso", not(target_arch = "wasm32")))]
            SolverBackend::CpuPardiso { .. } => Ok(DeviceBackend::Cpu(CpuDevice::default())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lca_core::{SparseMatrix, sparse_matrix::Triplete};

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
            let backend = DeviceBackend::Gpu(GpuDevice::new().await.unwrap());
            let result = calculate_lca_with_backend(
                &backend,
                &a,
                &b,
                &c,
                f,
                1000,
                1e-12,
                &SolverBackend::GpuBiCGSTAB,
            )
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

    #[test]
    fn test_evaluate_multi_on_small_system() {
        // Build a tiny system identical to test_calculate_lca_system but via EvalLCASystem
        // Matrices
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

        // Wrap into LcaMatrix with IDs
        let proc_ids = vec![
            "Bike production DK|Bike".to_string(),
            "Carbon fibre DE|Carbon fibre DE".to_string(),
            "Natural gas NO|Natural gas NO".to_string(),
        ];
        let sub_ids = vec!["CO2".to_string()];
        let impact_ids = vec!["GWP100".to_string()];

        let a_lca = LcaMatrix::new(a, proc_ids.clone(), proc_ids.clone()).unwrap();
        let b_lca = LcaMatrix::new(b, proc_ids.clone(), sub_ids.clone()).unwrap();
        let c_lca = LcaMatrix::new(c, sub_ids.clone(), impact_ids.clone()).unwrap();

        let system = LcaSystem::new(
            "SmallSys".to_string(),
            a_lca,
            b_lca,
            c_lca,
            None,                             // we'll provide demands via EvalLCASystem API
            Some(vec!["GWP100".to_string()]), // evaluation methods
            vec![],
            vec![],
            vec![],
        )
        .unwrap();

        pollster::block_on(async {
            let eval_sys: EvalLCASystem = system.try_into().unwrap();

            // Two demand sets: Proc0=1, and Proc1=1
            let demands = vec![
                vec![DemandItem::new("Bike production DK|Bike".to_string(), 1.0)],
                vec![DemandItem::new(
                    "Carbon fibre DE|Carbon fibre DE".to_string(),
                    1.0,
                )],
            ];

            let results = eval_sys
                .with_demands(demands)
                .evaluate_multi()
                .await
                .unwrap();
            assert_eq!(results.len(), 2);
            assert_eq!(results[0].len(), 1);
            assert_eq!(results[1].len(), 1);
            let expected0 = 66.5_f64; // same as single-demand test for Bike
            let expected1 = 26.6_f64; // demand on Carbon Fibre only
            assert!((results[0][0] - expected0).abs() < 1e-3);
            assert!((results[1][0] - expected1).abs() < 1e-3);
        });
    }
}
