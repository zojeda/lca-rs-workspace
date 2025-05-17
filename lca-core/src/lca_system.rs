use std::{collections::{HashMap, HashSet}, hash::Hash};

use serde::{Deserialize, Serialize};
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

use crate::{error::Result, sparse_matrix::Triplete, LcaCoreError, LcaMatrix, SparseMatrix};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InterSystemLink {
    pub local_source_row_id: String,
    pub target_system_name: String,
    pub target_system_row_id: String,
    pub value: f64,
}

impl InterSystemLink {
    pub fn new(
        local_source_row_id: String,
        target_system_name: String,
        target_system_col_id: String,
        value: f64,
    ) -> Self {
        Self {
            local_source_row_id,
            target_system_name,
            target_system_row_id: target_system_col_id,
            value,
        }
    }
}

#[cfg_attr(feature = "wasm", wasm_bindgen)]
#[derive(Debug, Clone)]
pub struct LcaSystem {
    pub name: String,
    pub a_matrix: LcaMatrix,
    pub b_matrix: LcaMatrix,
    pub c_matrix: LcaMatrix,
    pub evaluation_demand: Option<Vec<DemandItem>>,
    pub evaluation_methods: Option<Vec<String>>,
    pub a_links: Vec<InterSystemLink>,
    pub b_links: Vec<InterSystemLink>,
    pub c_links: Vec<InterSystemLink>,
}

#[cfg_attr(feature = "wasm", wasm_bindgen)]
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct DemandItem {
    pub product: String, // Made pub
    pub amount: f64,     // Made pub
}
impl DemandItem {
    pub fn new(product: String, amount: f64) -> Self {
        // new is already pub
        Self { product, amount }
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
impl LcaSystem {
    pub fn new(
        name: String,
        a_matrix: LcaMatrix,
        b_matrix: LcaMatrix,
        c_matrix: LcaMatrix,
        evaluation_demand: Option<Vec<DemandItem>>,
        evaluation_methods: Option<Vec<String>>,
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
            evaluation_demand,
            evaluation_methods,
            a_links,
            b_links,
            c_links,
        })
    }

    pub fn combine(systems: Vec<LcaSystem>) -> Result<Self> {
        if systems.is_empty() {
            return Err(LcaCoreError::Generic(
                "Cannot combine an empty list of systems".to_string(),
            )
            .into());
        }

        // Check for unique system names
        let mut system_names = HashSet::new();
        for sys in &systems {
            if !system_names.insert(sys.name.clone()) {
                return Err(LcaCoreError::Generic(format!(
                    "Duplicate system name found: {}. System names must be unique for combination.",
                    sys.name
                ))
                .into());
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

        // Combine evaluation demands
        let mut combined_eval_demand_set = Vec::new();
        for sys in &systems {
            if let Some(demand_items) = &sys.evaluation_demand {
                for item in demand_items {
                    combined_eval_demand_set.push(DemandItem {
                        product: format!("{}::{}", sys.name, item.product),
                        amount: item.amount,
                    });
                }
            }
        }
        let final_eval_demand_names = if combined_eval_demand_set.is_empty() {
            None
        } else {
            Some(combined_eval_demand_set.into_iter().collect())
        };



        // Combine evaluation methods
        let mut combined_eval_methods_set = HashSet::new();
        for sys in &systems {
            if let Some(methods) = &sys.evaluation_methods {
                for method in methods {
                    combined_eval_methods_set.insert(method.clone());
                }
            }
        }
        let final_eval_methods = if combined_eval_methods_set.is_empty() {
            None
        } else {
            Some(combined_eval_methods_set.into_iter().collect())
        };

        // Create a meaningful name for the combined system
        let combined_system_name = systems
            .iter()
            .map(|s| s.name.as_str())
            .collect::<Vec<&str>>()
            .join("_plus_");

        Ok(LcaSystem {
            name: combined_system_name,
            a_matrix: combined_a_matrix,
            b_matrix: combined_b_matrix,
            c_matrix: combined_c_matrix,
            evaluation_demand: final_eval_demand_names,
            evaluation_methods: final_eval_methods,
            a_links: Vec::new(), // Links are resolved into the new matrices
            b_links: Vec::new(),
            c_links: Vec::new(),
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
    pub fn demands(&self) -> &Option<Vec<DemandItem>> {
        &self.evaluation_demand
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
        log::debug!(
            "Phase 1: Collecting and mapping global IDs for {} matrix.",
            matrix_type_for_log
        );
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

        log::debug!(
            "Collected {} unique global row IDs and {} unique global col IDs for {} matrix.",
            final_row_ids.len(),
            final_col_ids.len(),
            matrix_type_for_log
        );

        let mut combined_triplets: Vec<Triplete> = Vec::new();

        // Phase 2: Process diagonal blocks (original matrices' triplets)
        log::debug!(
            "Phase 2: Processing diagonal blocks for {} matrix.",
            matrix_type_for_log
        );
        for sys in systems {
            let matrix = get_matrix_fn(sys);
            for triplet in matrix.matrix.iter() {
                let local_row_id_str = matrix.row_ids.get(triplet.row()).ok_or_else(|| LcaCoreError::Generic(format!("Invalid local row index {} for system '{}' in {} matrix during diagonal processing.", triplet.row(), sys.name, matrix_type_for_log)))?;
                let global_row_id_str = format!("{}::{}", sys.name, local_row_id_str);
                let final_r = *global_row_id_to_idx.get(&global_row_id_str)
                  .ok_or_else(|| LcaCoreError::Generic(format!("Global row ID '{}' (from local '{}' in system '{}') not found in map for {} matrix.", global_row_id_str, local_row_id_str, sys.name, matrix_type_for_log)))?;

                let local_col_id_str = matrix.col_ids.get(triplet.col()).ok_or_else(|| LcaCoreError::Generic(format!("Invalid local col index {} for system '{}' in {} matrix during diagonal processing.", triplet.col(), sys.name, matrix_type_for_log)))?;
                let global_col_id_str = format!("{}::{}", sys.name, local_col_id_str);
                let final_c = *global_col_id_to_idx.get(&global_col_id_str)
                  .ok_or_else(|| LcaCoreError::Generic(format!("Global col ID '{}' (from local '{}' in system '{}') not found in map for {} matrix.", global_col_id_str, local_col_id_str, sys.name, matrix_type_for_log)))?;

                combined_triplets.push(Triplete::new(final_r, final_c, triplet.value()));
            }
        }

        // Phase 3: Process off-diagonal blocks (InterSystemLinks)
        // Activities/Processes (or Substances for B) are always columns, and Products are always columns.
        log::debug!(
            "Phase 3: Processing off-diagonal links for {} matrix.",
            matrix_type_for_log
        );
        for sys in systems {
            // Iterate through systems providing the links
            let links = get_links_fn(sys);
            for link in links {
                // Source row comes from the current 'sys' (activity/process/substance)
                let global_source_row_id = format!("{}::{}", sys.name, link.local_source_row_id);
                // Target row comes from 'link.target_system_name' (required product input)
                let global_target_row_id =
                    format!("{}::{}", link.target_system_name, link.target_system_row_id);

                let final_c = *global_row_id_to_idx.get(&global_source_row_id)
                  .ok_or_else(|| LcaCoreError::LinkError(format!("Link source row ID '{}' (local '{}' from system '{}') not found in global map for {} matrix.", global_source_row_id, link.local_source_row_id, sys.name, matrix_type_for_log)))?;

                let final_r = *global_col_id_to_idx.get(&global_target_row_id)
                  .ok_or_else(|| LcaCoreError::LinkError(format!("Link target row ID '{}' (local '{}' in system '{}') not found in global map for {} matrix.", global_target_row_id, link.target_system_row_id, link.target_system_name, matrix_type_for_log)))?;

                // The current design assumes links are specific to the matrix type being combined.

                let link_resolution = Triplete::new(final_r, final_c, link.value);
                combined_triplets.push(link_resolution);
            }
        }

        let num_rows = final_row_ids.len();
        let num_cols = final_col_ids.len();
        log::debug!(
            "Creating final {} matrix with {} rows and {} columns from {} triplets.",
            matrix_type_for_log,
            num_rows,
            num_cols,
            combined_triplets.len()
        );

        let sparse_matrix = SparseMatrix::from_triplets(num_rows, num_cols, combined_triplets)?;
        LcaMatrix::new(sparse_matrix, final_col_ids, final_row_ids)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_combine_two_simple_systems_no_links() {
        let sys_a = create_test_system(
            "SysA",
            2,
            1,
            1,                              // P, S, I counts
            vec![(0, 0, 1.0), (1, 1, 1.0)], // A
            vec![(0, 0, 0.5)],              // B (Sub0-Proc0)
            vec![(0, 0, 2.0)],              // C (Impact0-Sub0)
            Some(vec![DemandItem::new("Proc0".to_string(), 1.0)]),
            Some(vec!["Impact0".to_string()]), // Evaluation methods
            vec![],
            vec![],
            vec![],
        );
        let sys_b = create_test_system(
            "SysB",
            1,
            1,
            1,
            vec![(0, 0, 1.0)],
            vec![(0, 0, 0.2)],
            vec![(0, 0, 3.0)],
            Some(vec![DemandItem::new("Proc0".to_string(), 1.0)]),
            Some(vec!["Impact0".to_string()]), // Evaluation methods
            vec![],
            vec![],
            vec![],
        );

        let combined = LcaSystem::combine(vec![sys_a, sys_b]).unwrap();

        assert_eq!(combined.name, "SysA_plus_SysB");

        // A Matrix
        assert_eq!(combined.a_matrix.row_ids.len(), 3); // SysA::Proc0, SysA::Proc1, SysB::Proc0
        assert_eq!(
            get_matrix_value(&combined.a_matrix, "SysA::Proc0", "SysA::Proc0"),
            Some(1.0)
        );
        assert_eq!(
            get_matrix_value(&combined.a_matrix, "SysA::Proc1", "SysA::Proc1"),
            Some(1.0)
        );
        assert_eq!(
            get_matrix_value(&combined.a_matrix, "SysB::Proc0", "SysB::Proc0"),
            Some(1.0)
        );
        assert_eq!(
            get_matrix_value(&combined.a_matrix, "SysA::Proc0", "SysB::Proc0"),
            None
        ); // No link

        // B Matrix
        assert_eq!(combined.b_matrix.row_ids.len(), 2); // SysA::Sub0, SysB::Sub0
        assert_eq!(combined.b_matrix.col_ids.len(), 3); // SysA::Proc0, SysA::Proc1, SysB::Proc0
        assert_eq!(
            get_matrix_value(&combined.b_matrix, "SysA::Sub0", "SysA::Proc0"),
            Some(0.5)
        );
        assert_eq!(
            get_matrix_value(&combined.b_matrix, "SysB::Sub0", "SysB::Proc0"),
            Some(0.2)
        );

        // C Matrix
        assert_eq!(combined.c_matrix.row_ids.len(), 2); // SysA::Impact0, SysB::Impact0
        assert_eq!(combined.c_matrix.col_ids.len(), 2); // SysA::Sub0, SysB::Sub0
        assert_eq!(
            get_matrix_value(&combined.c_matrix, "SysA::Impact0", "SysA::Sub0"),
            Some(2.0)
        );
        assert_eq!(
            get_matrix_value(&combined.c_matrix, "SysB::Impact0", "SysB::Sub0"),
            Some(3.0)
        );

        // Eval demands
        assert_eq!(
            combined.evaluation_demand,
            Some(vec![DemandItem::new("SysA::Proc0".to_string(), 1.0), DemandItem::new("SysB::Proc0".to_string(), 1.0)])
        );
        // Evaluation methods
        assert_eq!(
            combined.evaluation_methods,
            Some(vec!["Impact0".to_string()])
        );
        // Links
        assert!(
            combined.a_links.is_empty()
                && combined.b_links.is_empty()
                && combined.c_links.is_empty()
        );
    }

    #[test]
    fn test_combine_with_a_links() {
        let sys_a = create_test_system(
            "SysA",
            1,
            0,
            0,
            vec![(0, 0, 1.0)],
            vec![],
            vec![],
            None,
            None,
            vec![InterSystemLink::new(
                "Proc0".to_string(),
                "SysB".to_string(),
                "Proc0".to_string(),
                -0.5,
            )],
            vec![],
            vec![],
        );
        let sys_b = create_test_system(
            "SysB",
            1,
            0,
            0,
            vec![(0, 0, 1.0)],
            vec![],
            vec![],
            None,
            Some(vec!["Impact0".to_string()]), // Evaluation methods,
            vec![],
            vec![],
            vec![],
        );
        let combined = LcaSystem::combine(vec![sys_a, sys_b]).unwrap();

        assert_eq!(
            get_matrix_value(&combined.a_matrix, "SysA::Proc0", "SysA::Proc0"),
            Some(1.0)
        );
        assert_eq!(
            get_matrix_value(&combined.a_matrix, "SysB::Proc0", "SysB::Proc0"),
            Some(1.0)
        );
        assert_eq!(
            get_matrix_value(&combined.a_matrix, "SysB::Proc0", "SysA::Proc0"),
            Some(-0.5)
        ); // Link
    }

    #[test]
    fn test_combine_with_b_links() {
        let sys_a = create_test_system(
            "SysA",
            1,
            1,
            0,
            vec![(0, 0, 1.0)],
            vec![(0, 0, 0.1)],
            vec![],
            None,
            None,
            vec![],
            vec![InterSystemLink::new(
                "Sub0".to_string(),
                "SysB".to_string(),
                "Proc0".to_string(),
                0.7,
            )],
            vec![],
        );
        let sys_b = create_test_system(
            "SysB",
            1,
            1,
            0,
            vec![(0, 0, 1.0)],
            vec![(0, 0, 0.2)],
            vec![],
            None,
            None,
            vec![],
            vec![],
            vec![],
        );
        let combined = LcaSystem::combine(vec![sys_a, sys_b]).unwrap();

        assert_eq!(
            get_matrix_value(&combined.b_matrix, "SysA::Sub0", "SysA::Proc0"),
            Some(0.1)
        );
        assert_eq!(
            get_matrix_value(&combined.b_matrix, "SysB::Sub0", "SysB::Proc0"),
            Some(0.2)
        );
        assert_eq!(
            get_matrix_value(&combined.b_matrix, "SysB::Sub0", "SysA::Proc0"),
            Some(0.7)
        ); // Link
    }

    #[test]
    fn test_combine_with_c_links() {
        let sys_a = create_test_system(
            "SysA",
            0,
            1,
            1,
            vec![],
            vec![],
            vec![(0, 0, 10.0)],
            None,
            None,
            vec![],
            vec![],
            vec![InterSystemLink::new(
                "Impact0".to_string(),
                "SysB".to_string(),
                "Sub0".to_string(),
                1.5,
            )],
        );
        let sys_b = create_test_system(
            "SysB",
            0,
            1,
            1,
            vec![],
            vec![],
            vec![(0, 0, 20.0)],
            None,
            None,
            vec![],
            vec![],
            vec![],
        );
        let combined = LcaSystem::combine(vec![sys_a, sys_b]).unwrap();

        assert_eq!(
            get_matrix_value(&combined.c_matrix, "SysA::Impact0", "SysA::Sub0"),
            Some(10.0)
        );
        assert_eq!(
            get_matrix_value(&combined.c_matrix, "SysB::Impact0", "SysB::Sub0"),
            Some(20.0)
        );
        assert_eq!(
            get_matrix_value(&combined.c_matrix, "SysB::Impact0", "SysA::Sub0"),
            Some(1.5)
        ); // Link
    }

    #[test]
    fn test_combine_evaluation_demands_logic() {
        let sys_a = create_test_system(
            "SysA",
            1,
            0,
            0,
            vec![(0, 0, 1.0)],
            vec![],
            vec![],
            Some(vec![DemandItem::new("Proc0".to_string(), 1.0)]),
            None,
            vec![],
            vec![],
            vec![],
        );
        let sys_b = create_test_system(
            "SysB",
            1,
            0,
            0,
            vec![(0, 0, 1.0)],
            vec![],
            vec![],
            None,
            None,
            vec![],
            vec![],
            vec![],
        ); // No eval demands
        let sys_c = create_test_system(
            "SysC",
            1,
            0,
            0,
            vec![(0, 0, 1.0)],
            vec![],
            vec![],
            Some(vec![DemandItem::new("Proc0".to_string(), 1.0)]),
            None,
            vec![],
            vec![],
            vec![],
        );

        // A and B
        let combined_ab = LcaSystem::combine(vec![sys_a.clone(), sys_b.clone()]).unwrap();
        let eval_ab_set: Vec<DemandItem> =
            combined_ab.evaluation_demand.unwrap().into_iter().collect();
        assert_eq!(eval_ab_set.len(), 1);
        assert_eq!(
            eval_ab_set,
            vec![DemandItem::new("SysA::Proc0".to_string(), 1.0)]
        );

        // A and C
        let combined_ac = LcaSystem::combine(vec![sys_a.clone(), sys_c.clone()]).unwrap();

        let eval_ac_set: Vec<DemandItem> =
            combined_ac.evaluation_demand.unwrap().into_iter().collect();
        assert_eq!(eval_ac_set.len(), 2);
        assert_eq!(
            eval_ac_set,
            vec![DemandItem::new("SysA::Proc0".to_string(), 1.0), DemandItem::new("SysC::Proc0".to_string(), 1.0)]
        );

        // B and B (different instances, but B has no eval demands)
        let sys_b2 = create_test_system(
            "SysB2",
            1,
            0,
            0,
            vec![(0, 0, 1.0)],
            vec![],
            vec![],
            None,
            None,
            vec![],
            vec![],
            vec![],
        );
        let combined_bb2 = LcaSystem::combine(vec![sys_b.clone(), sys_b2.clone()]).unwrap();
        assert!(combined_bb2.evaluation_demand.is_none());
    }

    #[test]
    fn test_combine_empty_list_error() {
        let systems: Vec<LcaSystem> = Vec::new();
        let result = LcaSystem::combine(systems);
        match result {
            Err(LcaCoreError::Generic(msg)) => {
                assert_eq!(msg, "Cannot combine an empty list of systems")
            }
            _ => panic!("Expected LcaError::Generic for empty list"),
        }
    }

    #[test]
    fn test_combine_duplicate_system_names_error() {
        let sys_a1 = create_test_system(
            "SysA",
            1,
            0,
            0,
            vec![(0, 0, 1.0)],
            vec![],
            vec![],
            None,
            None,
            vec![],
            vec![],
            vec![],
        );
        let sys_a2 = create_test_system(
            "SysA",
            1,
            0,
            0,
            vec![(0, 0, 1.0)],
            vec![],
            vec![],
            None,
            None,
            vec![],
            vec![],
            vec![],
        ); // Same name
        let result = LcaSystem::combine(vec![sys_a1, sys_a2]);
        match result {
            Err(LcaCoreError::Generic(msg)) => assert_eq!(
                msg,
                "Duplicate system name found: SysA. System names must be unique for combination."
            ),
            _ => panic!(
                "Expected LcaCoreError::Generic for duplicate names, got {:?}",
                result
            ),
        }
    }

    #[test]
    fn test_combine_link_to_non_existent_target_id_error() {
        let sys_a = create_test_system(
            "SysA",
            1,
            0,
            0,
            vec![(0, 0, 1.0)],
            vec![],
            vec![],
            None,
            None,
            vec![InterSystemLink::new(
                "Proc0".to_string(),
                "SysB".to_string(),
                "NonExistentProc".to_string(),
                -0.5,
            )],
            vec![],
            vec![],
        );
        let sys_b = create_test_system(
            "SysB",
            1,
            0,
            0,
            vec![(0, 0, 1.0)],
            vec![],
            vec![],
            None,
            None,
            vec![],
            vec![],
            vec![],
        ); // SysB only has Proc0

        let result = LcaSystem::combine(vec![sys_a, sys_b]);
        match result {
            Err(LcaCoreError::LinkError(msg)) => {
                assert!(msg.contains("Link target row ID 'SysB::NonExistentProc'"));
                assert!(msg.contains("not found in global map for A matrix"));
            }
            _ => panic!(
                "Expected LcaCoreError::LinkError for link to non-existent ID, got {:?}",
                result
            ),
        }
    }

    #[test]
    fn test_combine_link_to_non_existent_source_id_error() {
        // Link from SysA::NonExistentProc
        let sys_a = create_test_system(
            "SysA",
            1,
            0,
            0,
            vec![(0, 0, 1.0)],
            vec![],
            vec![],
            None,
            None,
            vec![InterSystemLink::new(
                "NonExistentProc".to_string(),
                "SysB".to_string(),
                "Proc0".to_string(),
                -0.5,
            )],
            vec![],
            vec![],
        );
        let sys_b = create_test_system(
            "SysB",
            1,
            0,
            0,
            vec![(0, 0, 1.0)],
            vec![],
            vec![],
            None,
            None,
            vec![],
            vec![],
            vec![],
        );

        let result = LcaSystem::combine(vec![sys_a, sys_b]);
        match result {
            Err(LcaCoreError::LinkError(msg)) => {
                assert!(msg.contains("Link source row ID 'SysA::NonExistentProc'"));
                assert!(msg.contains("not found in global map for A matrix"));
            }
            _ => panic!(
                "Expected LcaCoreError::LinkError for link from non-existent ID, got {:?}",
                result
            ),
        }
    }

    #[test]
    fn test_combine_link_to_non_existent_target_system_error() {
        let sys_a = create_test_system(
            "SysA",
            1,
            0,
            0,
            vec![(0, 0, 1.0)],
            vec![],
            vec![],
            None,
            None,
            vec![InterSystemLink::new(
                "Proc0".to_string(),
                "SysC".to_string(),
                "Proc0".to_string(),
                -0.5,
            )], // SysC not in combination
            vec![],
            vec![],
        );
        let sys_b = create_test_system(
            "SysB",
            1,
            0,
            0,
            vec![(0, 0, 1.0)],
            vec![],
            vec![],
            None,
            None,
            vec![],
            vec![],
            vec![],
        );

        let result = LcaSystem::combine(vec![sys_a, sys_b]); // SysC is not included
        match result {
            Err(LcaCoreError::LinkError(msg)) => {
                assert!(msg.contains("Link target row ID 'SysC::Proc0'"));
                assert!(msg.contains("not found in global map for A matrix"));
            }
            _ => panic!(
                "Expected LcaCoreError::LinkError for link to non-existent system, got {:?}",
                result
            ),
        }
    }

    // Helper function to create an LcaSystem for testing
    fn create_test_system(
      name: &str,
      n_procs: usize,
      n_subs: usize,
      n_impacts: usize,
      a_triplets: Vec<(usize, usize, f64)>, // (proc_idx, proc_idx, val)
      b_triplets: Vec<(usize, usize, f64)>, // (sub_idx, proc_idx, val)
      c_triplets: Vec<(usize, usize, f64)>, // (impact_idx, sub_idx, val)
      eval_demands: Option<Vec<DemandItem>>, // Local process names like "Proc0"
      eval_methods: Option<Vec<String>>, // Local demand names like "Impact0"
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
          eval_methods,
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


}
