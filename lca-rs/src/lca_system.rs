use lca_core::{GpuDevice, Matrix, SparseMatrix, sparse_matrix::Triplete};
use lca_lsolver::algorithms::{BiCGSTAB, SolveAlgorithm}; // Added QMR here and SolveResult
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};


use crate::{
    LcaMatrix,
    error::{LcaError, Result},
};

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InterSystemLink {
    pub local_source_row_id: String,
    pub target_system_name: String,
    pub target_system_row_id: String,
    pub value: f64,
}

impl InterSystemLink {
    pub fn new(local_source_row_id: String, target_system_name: String, target_system_col_id: String, value: f64) -> Self {
        Self { local_source_row_id, target_system_name, target_system_row_id: target_system_col_id, value }
    }
}


#[cfg_attr(feature = "wasm", wasm_bindgen)]
#[derive(Debug, Clone)]
pub struct LcaSystem {
    pub name: String,
    pub a_matrix: LcaMatrix,
    pub b_matrix: LcaMatrix,
    pub c_matrix: LcaMatrix,
    pub evaluation_demand_process_names: Option<Vec<String>>,
    pub a_links: Vec<InterSystemLink>,
    pub b_links: Vec<InterSystemLink>,
    pub c_links: Vec<InterSystemLink>,
}

#[cfg_attr(feature = "wasm", wasm_bindgen)]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DemandItem {
    pub(crate) product: String,
    pub(crate) amount: f64,
}

impl DemandItem {
    pub fn new(product: String, amount: f64) -> Self {
        Self { product, amount }
    }
}

#[cfg_attr(feature = "wasm", wasm_bindgen)]
impl LcaSystem {
    pub fn new(
        name: String,
        a_matrix: LcaMatrix,
        b_matrix: LcaMatrix,
        c_matrix: LcaMatrix,
        evaluation_demand_process_names: Option<Vec<String>>,
        a_links: Vec<InterSystemLink>,
        b_links: Vec<InterSystemLink>,
        c_links: Vec<InterSystemLink>,
    ) -> Result<Self> {
        // TODO: Add dimension checks if necessary, e.g., ensuring links refer to valid local IDs.
        Ok(Self {
            name,
            a_matrix,
            b_matrix,
            c_matrix,
            evaluation_demand_process_names,
            a_links,
            b_links,
            c_links,
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

    #[cfg(not(target_arch = "wasm32"))]
    pub fn name(&self) -> &String {
        &self.name
    }
    
    #[cfg(not(target_arch = "wasm32"))]
    pub fn a_links(&self) -> &Vec<InterSystemLink> {
        &self.a_links
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn b_links(&self) -> &Vec<InterSystemLink> {
        &self.b_links
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn c_links(&self) -> &Vec<InterSystemLink> {
        &self.c_links
    }


    pub async fn evaluate(
        &self,
        device: &GpuDevice,
        demand: Option<Vec<DemandItem>>,
        methods: Option<Vec<String>>,
    ) -> Result<Vec<f64>> {
        log::info!("Evaluating LCA system...");
        log::debug!("A matrix: {:?}", self.a_matrix.matrix.dims());
        log::debug!("B matrix: {:?}", self.b_matrix.matrix.dims());
        log::debug!("C matrix: {:?}", self.c_matrix.matrix.dims());

        if !self.a_links.is_empty() || !self.b_links.is_empty() || !self.c_links.is_empty() {
            return Err(LcaError::LinkError(
                "Cannot evaluate a system with unresolved inter-system links.".to_string(),
            ));
        }

        let f_vec = self.get_demand_vector(demand)?;
        log::info!("f (demand) vector length: {}", f_vec.len());

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
            8e-6, // Reverted tolerance back
        )
        .await
    }

    fn get_demand_vector(&self, demand: Option<Vec<DemandItem>>) -> Result<Vec<f64>> {
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

    pub fn combine(systems: Vec<LcaSystem>) -> Result<Self> {
        if systems.is_empty() {
            return Err(LcaError::Generic("Cannot combine an empty list of systems".to_string()).into());
        }

        // Check for unique system names
        let mut system_names = HashSet::new();
        for sys in &systems {
            if !system_names.insert(sys.name.clone()) {
                return Err(LcaError::Generic(format!("Duplicate system name found: {}. System names must be unique for combination.", sys.name)).into());
            }
        }

        let combined_a_matrix = Self::combine_single_matrix_type(
            &systems,
            |sys| &sys.a_matrix,
            |sys| &sys.a_links,
            "A",
        )?;

        let combined_b_matrix = Self::combine_single_matrix_type(
            &systems,
            |sys| &sys.b_matrix,
            |sys| &sys.b_links,
            "B",
        )?;

        let combined_c_matrix = Self::combine_single_matrix_type(
            &systems,
            |sys| &sys.c_matrix,
            |sys| &sys.c_links,
            "C",
        )?;

        let mut combined_eval_demand_names_set = HashSet::new();
        for sys in &systems {
            if let Some(names) = &sys.evaluation_demand_process_names {
                for local_name in names {
                    // Ensure namespacing for demand names
                    combined_eval_demand_names_set.insert(format!("{}::{}", sys.name, local_name));
                }
            }
        }
        
        let final_eval_demand_names = if combined_eval_demand_names_set.is_empty() {
            None
        } else {
            Some(combined_eval_demand_names_set.into_iter().collect())
        };

        // Create a meaningful name for the combined system
        let combined_system_name = systems.iter().map(|s| s.name.as_str()).collect::<Vec<&str>>().join("_plus_");

        Ok(LcaSystem {
            name: combined_system_name,
            a_matrix: combined_a_matrix,
            b_matrix: combined_b_matrix,
            c_matrix: combined_c_matrix,
            evaluation_demand_process_names: final_eval_demand_names,
            a_links: Vec::new(), // Links are resolved into the new matrices
            b_links: Vec::new(),
            c_links: Vec::new(),
        })
    }

    fn combine_single_matrix_type<'a>(
        systems: &'a [LcaSystem],
        get_matrix_fn: impl Fn(&'a LcaSystem) -> &'a LcaMatrix,
        get_links_fn: impl Fn(&'a LcaSystem) -> &'a Vec<InterSystemLink>,
        matrix_type_for_log: &str, // For logging/error messages
    ) -> Result<LcaMatrix> {
        log::info!("Combining {} matrices...", matrix_type_for_log);
        let mut final_row_ids: Vec<String> = Vec::new();
        let mut final_col_ids: Vec<String> = Vec::new();
        let mut global_row_id_to_idx: HashMap<String, usize> = HashMap::new();
        let mut global_col_id_to_idx: HashMap<String, usize> = HashMap::new();
        let mut current_final_row_idx = 0;
        let mut current_final_col_idx = 0;

        // Phase 1: Collect all unique global row and column IDs and map them to indices
        log::debug!("Phase 1: Collecting and mapping global IDs for {} matrix.", matrix_type_for_log);
        for sys in systems {
            let matrix = get_matrix_fn(sys);
            for local_row_id in &matrix.row_ids {
                let global_id = format!("{}::{}", sys.name, local_row_id);
                if !global_row_id_to_idx.contains_key(&global_id) {
                    final_row_ids.push(global_id.clone());
                    global_row_id_to_idx.insert(global_id, current_final_row_idx);
                    current_final_row_idx += 1;
                }
            }
            for local_col_id in &matrix.col_ids {
                let global_id = format!("{}::{}", sys.name, local_col_id);
                if !global_col_id_to_idx.contains_key(&global_id) {
                    final_col_ids.push(global_id.clone());
                    global_col_id_to_idx.insert(global_id, current_final_col_idx);
                    current_final_col_idx += 1;
                }
            }
        }

        log::debug!("Collected {} unique global row IDs and {} unique global col IDs for {} matrix.", 
                   final_row_ids.len(), final_col_ids.len(), matrix_type_for_log);


        let mut combined_triplets: Vec<Triplete> = Vec::new();

        // Phase 2: Process diagonal blocks (original matrices' triplets)
        log::debug!("Phase 2: Processing diagonal blocks for {} matrix.", matrix_type_for_log);
        for sys in systems {
            let matrix = get_matrix_fn(sys);
            for triplet in matrix.matrix.iter() {
                let local_row_id_str = matrix.row_ids.get(triplet.row()).ok_or_else(|| LcaError::Generic(format!("Invalid local row index {} for system '{}' in {} matrix during diagonal processing.", triplet.row(), sys.name, matrix_type_for_log)))?;
                let global_row_id_str = format!("{}::{}", sys.name, local_row_id_str);
                let final_r = *global_row_id_to_idx.get(&global_row_id_str)
                    .ok_or_else(|| LcaError::Generic(format!("Global row ID '{}' (from local '{}' in system '{}') not found in map for {} matrix.", global_row_id_str, local_row_id_str, sys.name, matrix_type_for_log)))?;

                let local_col_id_str = matrix.col_ids.get(triplet.col()).ok_or_else(|| LcaError::Generic(format!("Invalid local col index {} for system '{}' in {} matrix during diagonal processing.", triplet.col(), sys.name, matrix_type_for_log)))?;
                let global_col_id_str = format!("{}::{}", sys.name, local_col_id_str);
                let final_c = *global_col_id_to_idx.get(&global_col_id_str)
                    .ok_or_else(|| LcaError::Generic(format!("Global col ID '{}' (from local '{}' in system '{}') not found in map for {} matrix.", global_col_id_str, local_col_id_str, sys.name, matrix_type_for_log)))?;
                
                combined_triplets.push(Triplete::new(final_r, final_c, triplet.value()));
            }
        }

        // Phase 3: Process off-diagonal blocks (InterSystemLinks)
        // Activities/Processes (or Substances for B) are always columns, and Products are always columns.
        log::debug!("Phase 3: Processing off-diagonal links for {} matrix.", matrix_type_for_log);
        for sys in systems { // Iterate through systems providing the links
            let links = get_links_fn(sys);
            for link in links {
                // Source row comes from the current 'sys' (activity/process/substance)
                let global_source_row_id = format!("{}::{}", sys.name, link.local_source_row_id);
                // Target row comes from 'link.target_system_name' (required product input)
                let global_target_row_id = format!("{}::{}", link.target_system_name, link.target_system_row_id);

                let final_c = *global_row_id_to_idx.get(&global_source_row_id)
                    .ok_or_else(|| LcaError::LinkError(format!("Link source row ID '{}' (local '{}' from system '{}') not found in global map for {} matrix.", global_source_row_id, link.local_source_row_id, sys.name, matrix_type_for_log)))?;
                
                let final_r = *global_col_id_to_idx.get(&global_target_row_id)
                    .ok_or_else(|| LcaError::LinkError(format!("Link target row ID '{}' (local '{}' in system '{}') not found in global map for {} matrix.", global_target_row_id, link.target_system_row_id, link.target_system_name, matrix_type_for_log)))?;

                // The current design assumes links are specific to the matrix type being combined.

                let link_resolution = Triplete::new(final_r, final_c, link.value);
                combined_triplets.push(link_resolution);
            }
        }
        
        let num_rows = final_row_ids.len();
        let num_cols = final_col_ids.len();
        log::debug!("Creating final {} matrix with {} rows and {} columns from {} triplets.", 
                   matrix_type_for_log, num_rows, num_cols, combined_triplets.len());

        let sparse_matrix = SparseMatrix::from_triplets(num_rows, num_cols, combined_triplets)?;
        LcaMatrix::new(sparse_matrix, final_col_ids, final_row_ids)
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
        Err(other_bicgstab_core_err) => { // Other LcaCoreError from BiCGSTAB
             log::error!("BiCGSTAB failed with unexpected core error: {:?}", other_bicgstab_core_err); // Use log::error
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
    use lca_core::{SparseMatrix, sparse_matrix::Triplete};
    use std::collections::HashSet;

    // Helper function to create an LcaSystem for testing
    fn create_test_system(
        name: &str,
        n_procs: usize,
        n_subs: usize,
        n_impacts: usize,
        a_triplets: Vec<(usize, usize, f64)>, // (proc_idx, proc_idx, val)
        b_triplets: Vec<(usize, usize, f64)>, // (sub_idx, proc_idx, val)
        c_triplets: Vec<(usize, usize, f64)>, // (impact_idx, sub_idx, val)
        eval_demands: Option<Vec<String>>, // Local process names like "Proc0"
        a_links: Vec<InterSystemLink>,
        b_links: Vec<InterSystemLink>,
        c_links: Vec<InterSystemLink>,
    ) -> LcaSystem {
        let proc_ids: Vec<String> = (0..n_procs).map(|i| format!("Proc{}", i)).collect();
        let sub_ids: Vec<String> = (0..n_subs).map(|i| format!("Sub{}", i)).collect();
        let impact_ids: Vec<String> = (0..n_impacts).map(|i| format!("Impact{}", i)).collect();

        let a_matrix_triplets = a_triplets.into_iter().map(|(r,c,v)| Triplete::new(r,c,v)).collect();
        let a_sm = SparseMatrix::from_triplets(n_procs, n_procs, a_matrix_triplets).unwrap();
        let a_lca_matrix = LcaMatrix::new(a_sm, proc_ids.clone(), proc_ids.clone()).unwrap();

        let b_matrix_triplets = b_triplets.into_iter().map(|(r,c,v)| Triplete::new(r,c,v)).collect();
        let b_sm = SparseMatrix::from_triplets(n_subs, n_procs, b_matrix_triplets).unwrap();
        let b_lca_matrix = LcaMatrix::new(b_sm, proc_ids.clone(), sub_ids.clone()).unwrap();

        let c_matrix_triplets = c_triplets.into_iter().map(|(r,c,v)| Triplete::new(r,c,v)).collect();
        let c_sm = SparseMatrix::from_triplets(n_impacts, n_subs, c_matrix_triplets).unwrap();
        let c_lca_matrix = LcaMatrix::new(c_sm, sub_ids.clone(), impact_ids.clone()).unwrap();

        LcaSystem::new(
            name.to_string(),
            a_lca_matrix,
            b_lca_matrix,
            c_lca_matrix,
            eval_demands,
            a_links,
            b_links,
            c_links,
        )
        .unwrap()
    }

    // Helper to get a value from a matrix using string IDs
    fn get_matrix_value(matrix: &LcaMatrix, row_id_str: &str, col_id_str: &str) -> Option<f64> {
        let r_idx = matrix.row_ids.iter().position(|id| id == row_id_str)?;
        let c_idx = matrix.col_ids.iter().position(|id| id == col_id_str)?;
        matrix.matrix.get(r_idx, c_idx)
    }

    #[test]
    fn test_combine_two_simple_systems_no_links() {
        let sys_a = create_test_system(
            "SysA", 2, 1, 1, // P, S, I counts
            vec![(0,0,1.0), (1,1,1.0)], // A
            vec![(0,0,0.5)],             // B (Sub0-Proc0)
            vec![(0,0,2.0)],             // C (Impact0-Sub0)
            Some(vec!["Proc0".to_string()]), vec![], vec![], vec![]
        );
        let sys_b = create_test_system(
            "SysB", 1, 1, 1,
            vec![(0,0,1.0)], vec![(0,0,0.2)], vec![(0,0,3.0)],
            Some(vec!["Proc0".to_string()]), vec![], vec![], vec![]
        );

        let combined = LcaSystem::combine(vec![sys_a, sys_b]).unwrap();

        assert_eq!(combined.name, "SysA_plus_SysB");

        // A Matrix
        assert_eq!(combined.a_matrix.row_ids.len(), 3); // SysA::Proc0, SysA::Proc1, SysB::Proc0
        assert_eq!(get_matrix_value(&combined.a_matrix, "SysA::Proc0", "SysA::Proc0"), Some(1.0));
        assert_eq!(get_matrix_value(&combined.a_matrix, "SysA::Proc1", "SysA::Proc1"), Some(1.0));
        assert_eq!(get_matrix_value(&combined.a_matrix, "SysB::Proc0", "SysB::Proc0"), Some(1.0));
        assert_eq!(get_matrix_value(&combined.a_matrix, "SysA::Proc0", "SysB::Proc0"), None); // No link

        // B Matrix
        assert_eq!(combined.b_matrix.row_ids.len(), 2); // SysA::Sub0, SysB::Sub0
        assert_eq!(combined.b_matrix.col_ids.len(), 3); // SysA::Proc0, SysA::Proc1, SysB::Proc0
        assert_eq!(get_matrix_value(&combined.b_matrix, "SysA::Sub0", "SysA::Proc0"), Some(0.5));
        assert_eq!(get_matrix_value(&combined.b_matrix, "SysB::Sub0", "SysB::Proc0"), Some(0.2));

        // C Matrix
        assert_eq!(combined.c_matrix.row_ids.len(), 2); // SysA::Impact0, SysB::Impact0
        assert_eq!(combined.c_matrix.col_ids.len(), 2); // SysA::Sub0, SysB::Sub0
        assert_eq!(get_matrix_value(&combined.c_matrix, "SysA::Impact0", "SysA::Sub0"), Some(2.0));
        assert_eq!(get_matrix_value(&combined.c_matrix, "SysB::Impact0", "SysB::Sub0"), Some(3.0));

        // Eval demands
        let eval_demands_set: HashSet<String> = combined.evaluation_demand_process_names.unwrap().into_iter().collect();
        let expected_eval_demands: HashSet<String> = ["SysA::Proc0", "SysB::Proc0"].iter().map(|s| s.to_string()).collect();
        assert_eq!(eval_demands_set, expected_eval_demands);

        assert!(combined.a_links.is_empty() && combined.b_links.is_empty() && combined.c_links.is_empty());
    }

    #[test]
    fn test_combine_with_a_links() {
        let sys_a = create_test_system("SysA", 1,0,0, vec![(0,0,1.0)], vec![], vec![], None,
            vec![InterSystemLink::new("Proc0".to_string(), "SysB".to_string(), "Proc0".to_string(), -0.5)],
            vec![], vec![]);
        let sys_b = create_test_system("SysB", 1,0,0, vec![(0,0,1.0)], vec![], vec![], None, vec![], vec![], vec![]);
        let combined = LcaSystem::combine(vec![sys_a, sys_b]).unwrap();

        assert_eq!(get_matrix_value(&combined.a_matrix, "SysA::Proc0", "SysA::Proc0"), Some(1.0));
        assert_eq!(get_matrix_value(&combined.a_matrix, "SysB::Proc0", "SysB::Proc0"), Some(1.0));
        assert_eq!(get_matrix_value(&combined.a_matrix, "SysB::Proc0", "SysA::Proc0"), Some(-0.5)); // Link
    }

    #[test]
    fn test_combine_with_b_links() {
        let sys_a = create_test_system("SysA", 1,1,0, vec![(0,0,1.0)], vec![(0,0,0.1)], vec![], None, vec![],
            vec![InterSystemLink::new("Sub0".to_string(), "SysB".to_string(), "Proc0".to_string(), 0.7)],
            vec![]);
        let sys_b = create_test_system("SysB", 1,1,0, vec![(0,0,1.0)], vec![(0,0,0.2)], vec![], None, vec![], vec![], vec![]);
        let combined = LcaSystem::combine(vec![sys_a, sys_b]).unwrap();

        assert_eq!(get_matrix_value(&combined.b_matrix, "SysA::Sub0", "SysA::Proc0"), Some(0.1));
        assert_eq!(get_matrix_value(&combined.b_matrix, "SysB::Sub0", "SysB::Proc0"), Some(0.2));
        assert_eq!(get_matrix_value(&combined.b_matrix, "SysB::Sub0", "SysA::Proc0"), Some(0.7)); // Link
    }

    #[test]
    fn test_combine_with_c_links() {
        let sys_a = create_test_system("SysA", 0,1,1, vec![], vec![], vec![(0,0,10.0)], None, vec![], vec![],
            vec![InterSystemLink::new("Impact0".to_string(), "SysB".to_string(), "Sub0".to_string(), 1.5)]);
        let sys_b = create_test_system("SysB", 0,1,1, vec![], vec![], vec![(0,0,20.0)], None, vec![], vec![], vec![]);
        let combined = LcaSystem::combine(vec![sys_a, sys_b]).unwrap();

        assert_eq!(get_matrix_value(&combined.c_matrix, "SysA::Impact0", "SysA::Sub0"), Some(10.0));
        assert_eq!(get_matrix_value(&combined.c_matrix, "SysB::Impact0", "SysB::Sub0"), Some(20.0));
        assert_eq!(get_matrix_value(&combined.c_matrix, "SysB::Impact0", "SysA::Sub0"), Some(1.5)); // Link
    }

    #[test]
    fn test_combine_evaluation_demands_logic() {
        let sys_a = create_test_system("SysA", 1,0,0, vec![(0,0,1.0)], vec![],vec![], Some(vec!["Proc0".to_string()]), vec![],vec![],vec![]);
        let sys_b = create_test_system("SysB", 1,0,0, vec![(0,0,1.0)], vec![],vec![], None, vec![],vec![],vec![]); // No eval demands
        let sys_c = create_test_system("SysC", 1,0,0, vec![(0,0,1.0)], vec![],vec![], Some(vec!["Proc0".to_string()]), vec![],vec![],vec![]);

        // A and B
        let combined_ab = LcaSystem::combine(vec![sys_a.clone(), sys_b.clone()]).unwrap();
        let eval_ab_set: HashSet<String> = combined_ab.evaluation_demand_process_names.unwrap().into_iter().collect();
        assert_eq!(eval_ab_set, ["SysA::Proc0"].iter().map(|s|s.to_string()).collect::<HashSet<String>>());

        // A and C
        let combined_ac = LcaSystem::combine(vec![sys_a.clone(), sys_c.clone()]).unwrap();
        let eval_ac_set: HashSet<String> = combined_ac.evaluation_demand_process_names.unwrap().into_iter().collect();
        assert_eq!(eval_ac_set, ["SysA::Proc0", "SysC::Proc0"].iter().map(|s|s.to_string()).collect::<HashSet<String>>());

        // B and B (different instances, but B has no eval demands)
        let sys_b2 = create_test_system("SysB2", 1,0,0, vec![(0,0,1.0)], vec![],vec![], None, vec![],vec![],vec![]);
        let combined_bb2 = LcaSystem::combine(vec![sys_b.clone(), sys_b2.clone()]).unwrap();
        assert!(combined_bb2.evaluation_demand_process_names.is_none());
    }

    #[test]
    fn test_combine_empty_list_error() {
        let systems: Vec<LcaSystem> = Vec::new();
        let result = LcaSystem::combine(systems);
        match result {
            Err(LcaError::Generic(msg)) => assert_eq!(msg, "Cannot combine an empty list of systems"),
            _ => panic!("Expected LcaError::Generic for empty list"),
        }
    }

    #[test]
    fn test_combine_duplicate_system_names_error() {
        let sys_a1 = create_test_system("SysA", 1,0,0, vec![(0,0,1.0)], vec![],vec![], None, vec![],vec![],vec![]);
        let sys_a2 = create_test_system("SysA", 1,0,0, vec![(0,0,1.0)], vec![],vec![], None, vec![],vec![],vec![]); // Same name
        let result = LcaSystem::combine(vec![sys_a1, sys_a2]);
        match result {
            Err(LcaError::Generic(msg)) => assert_eq!(msg, "Duplicate system name found: SysA. System names must be unique for combination."),
            _ => panic!("Expected LcaError::Generic for duplicate names, got {:?}", result),
        }
    }

    #[test]
    fn test_combine_link_to_non_existent_target_id_error() {
        let sys_a = create_test_system("SysA", 1,0,0, vec![(0,0,1.0)], vec![],vec![], None,
            vec![InterSystemLink::new("Proc0".to_string(), "SysB".to_string(), "NonExistentProc".to_string(), -0.5)],
            vec![], vec![]);
        let sys_b = create_test_system("SysB", 1,0,0, vec![(0,0,1.0)], vec![],vec![], None, vec![],vec![],vec![]); // SysB only has Proc0

        let result = LcaSystem::combine(vec![sys_a, sys_b]);
        match result {
            Err(LcaError::LinkError(msg)) => {
                assert!(msg.contains("Link target row ID 'SysB::NonExistentProc'"));
                assert!(msg.contains("not found in global map for A matrix"));
            }
            _ => panic!("Expected LcaError::LinkError for link to non-existent ID, got {:?}", result),
        }
    }

    #[test]
    fn test_combine_link_to_non_existent_source_id_error() {
        // Link from SysA::NonExistentProc
        let sys_a = create_test_system("SysA", 1,0,0, vec![(0,0,1.0)], vec![],vec![], None,
            vec![InterSystemLink::new("NonExistentProc".to_string(), "SysB".to_string(), "Proc0".to_string(), -0.5)],
            vec![], vec![]);
        let sys_b = create_test_system("SysB", 1,0,0, vec![(0,0,1.0)], vec![],vec![], None, vec![],vec![],vec![]);

        let result = LcaSystem::combine(vec![sys_a, sys_b]);
        match result {
            Err(LcaError::LinkError(msg)) => {
                assert!(msg.contains("Link source row ID 'SysA::NonExistentProc'"));
                assert!(msg.contains("not found in global map for A matrix"));
            }
            _ => panic!("Expected LcaError::LinkError for link from non-existent ID, got {:?}", result),
        }
    }

    #[test]
    fn test_combine_link_to_non_existent_target_system_error() {
        let sys_a = create_test_system("SysA", 1,0,0, vec![(0,0,1.0)], vec![],vec![], None,
            vec![InterSystemLink::new("Proc0".to_string(), "SysC".to_string(), "Proc0".to_string(), -0.5)], // SysC not in combination
            vec![], vec![]);
        let sys_b = create_test_system("SysB", 1,0,0, vec![(0,0,1.0)], vec![],vec![], None, vec![],vec![],vec![]);

        let result = LcaSystem::combine(vec![sys_a, sys_b]); // SysC is not included
        match result {
            Err(LcaError::LinkError(msg)) => {
                assert!(msg.contains("Link target row ID 'SysC::Proc0'"));
                assert!(msg.contains("not found in global map for A matrix"));
            }
            _ => panic!("Expected LcaError::LinkError for link to non-existent system, got {:?}", result),
        }
    }
    #[test]
    fn test_calculate_lca_system() {
        // A[0,0] = 1.0 (Bike output)
        // A[1,1] = 1.0 (Carbon Fibre output)
        // A[2,2] = 1.0 (Natural Gas output)
        // A[1,0] = -2.5 (Carbon Fibre input to Bike)
        // A[2,1] = -237.0 (Natural Gas input to Carbon Fibre)
        // B[0,1] = 26.6 (CO2 emission from Carbon Fibre)
        // C[0,0] = 1.0 (GWP100 impact from CO2)
      let a = SparseMatrix::from_triplets(3, 3, vec![
          Triplete::new(0, 0, 1.0),
          Triplete::new(1, 1, 1.0),
          Triplete::new(2, 2, 1.0),
          Triplete::new(1, 0, -2.5),
          Triplete::new(2, 1, -237.0),
      ]).unwrap();
      let b = SparseMatrix::from_triplets(1, 3, vec![
        Triplete::new(0, 1, 26.6),
      ]).unwrap();  

      let c = SparseMatrix::from_triplets(1, 1, vec![
        Triplete::new(0, 0, 1.0),
      ]).unwrap();

      let f = vec![1.0, 0.0, 0.0];
      pollster::block_on(async {
        let device = GpuDevice::new().await.unwrap();
        let result = calculate_lca(&device, &a, &b, &c, f, 1000, 1e-12)
          .await
          .expect("LCA calculation failed");
        assert_eq!(result.len(), 1);
        let expected_co2 = 66.5;
        assert!((result[0] - expected_co2).abs() < 1e-3, "LCA result for GWP100 is incorrect. Expected: {}, Got: {}", expected_co2, result[0]);
      })
    }
}
