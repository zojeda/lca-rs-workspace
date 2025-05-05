use lca_core::{SparseMatrix, sparse_matrix::Triplete};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::{DemandItem, LcaMatrix, LcaSystem, error::LcaModelCompilationError};
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LcaModel {
    pub database_name: String,
    pub imported_dbs: Vec<DbImport>,
    pub processes: Vec<Process>,
    pub substances: Vec<Substance>,
    pub evaluation_demands: Vec<Vec<DemandItem>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DbImport {
    pub name: String,
    pub alias: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Process {
    pub name: String,
    pub products: Vec<OutputItem>,
    pub inputs: Vec<InputItem>,
    pub emissions: Vec<BiosphereItem>,
    pub resources: Vec<BiosphereItem>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OutputItem {
    pub product: Product,
    pub amount: Amount,
    pub unit: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BiosphereItem {
    pub substance: SubstanceRef,
    pub amount: Amount,
    pub unit: String,
    pub compartment: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InputItem {
    pub product: ProductRef,
    pub amount: Amount,
    pub unit: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ProductRef {
    Product(String),
    External(ExternalRef),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum SubstanceRef {
    Substance(String),
    External(ExternalRef),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExternalRef {
    pub alias: String,
    pub name: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Amount {
    Literal(f32),
}

impl Amount {
    pub fn evaluate(&self) -> core::result::Result<f32, LcaModelCompilationError> {
        match self {
            Amount::Literal(val) => Ok(*val),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Product {
    pub name: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Substance {
    pub name: String,
    pub r#type: SubstanceType,
    pub compartment: Option<String>,
    pub sub_compartment: Option<String>,
    pub reference_unit: String,
    pub impacts: Vec<Impact>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Impact {
    pub name: String,
    pub amount: Amount,
    pub unit: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum SubstanceType {
    Emission,
    Resource,
}

/// Compiles the LCA (Life Cycle Assessment) model into a matrix-based representation suitable for calculations.
///
/// This process involves several key steps:
/// 1. **Internal Validation:** Ensures the model's structural integrity, including valid references between processes, products, and substances.
/// 2. **Amount Evaluation:** Computes numerical values from any expressions used for amounts within the model.
/// 3. **Unit Consistency (TODO):**  Currently a placeholder, this will eventually handle unit conversions and checks for consistency.
/// 4. **Matrix Construction:** Creates the A (technosphere), B (characterization), and C (impact assessment) matrices, which represent the core of the LCA model.
/// 5. **Unlinked Item Detection:** Identifies any inputs or biosphere items that reference external databases not included in the current model.
///
/// If the model is invalid, a `LcaModelCompilationError` is returned, detailing the specific issue.
/// Otherwise, a `CompiledLcaModel` is returned, containing the matrix representation,
/// as well as lists of any unlinked inputs or biosphere items (represented as `ExternalRef`).
impl LcaModel {
    pub fn compile(self) -> core::result::Result<CompiledLcaModel, LcaModelCompilationError> {
        let num_processes = self.processes.len();
        let num_substances = self.substances.len();

        // --- Create Mappings and ID Lists ---
        let mut process_map: HashMap<String, usize> = HashMap::with_capacity(num_processes);
        let mut process_ids: Vec<String> = Vec::with_capacity(num_processes);
        for (i, process) in self.processes.iter().enumerate() {
            if process_map.insert(process.name.clone(), i).is_some() {
                return Err(LcaModelCompilationError::DuplicateProcessName(
                    process.name.clone(),
                ));
            }
            process_ids.push(process.name.clone());
        }

        // --- Initialize for Evaluation Demands ---
        let mut evaluation_demand_process_names_for_system: Vec<String> = Vec::new();
        let mut current_eval_demand_idx_offset = 0;

        let mut substance_map: HashMap<String, usize> = HashMap::with_capacity(num_substances);
        let mut substance_ids: Vec<String> = Vec::with_capacity(num_substances);
        let mut impact_map: HashMap<String, usize> = HashMap::new(); // Map impact names to C matrix rows
        let mut impact_ids: Vec<String> = Vec::new(); // Ordered list of impact names for C matrix rows
        let mut c_triplets: Vec<Triplete> = Vec::new(); // Triplets for C matrix

        for (substance_idx, substance) in self.substances.iter().enumerate() {
            if substance_map
                .insert(substance.name.clone(), substance_idx)
                .is_some()
            {
                return Err(LcaModelCompilationError::DuplicateSubstanceName(
                    substance.name.clone(),
                ));
            }
            substance_ids.push(substance.name.clone());

            // Process impacts for C matrix
            for impact in &substance.impacts {
                let impact_idx = *impact_map.entry(impact.name.clone()).or_insert_with(|| {
                    let next_idx = impact_ids.len();
                    impact_ids.push(impact.name.clone());
                    next_idx
                });
                let amount = impact.amount.evaluate()?;
                // TODO: Add unit checks/conversions for impacts if necessary
                c_triplets.push(Triplete::new(impact_idx, substance_idx, amount));
            }
        }
        let num_impacts = impact_ids.len();

        // --- Initialize A/B Matrix Triplets and Unlinked Lists ---
        let mut a_triplets: Vec<Triplete> = Vec::new();
        let mut b_triplets: Vec<Triplete> = Vec::new();
        let mut unlinked_inputs: Vec<ExternalRef> = Vec::new();
        let mut unlinked_biosphere_items: Vec<ExternalRef> = Vec::new();

        // --- Iterate Through Processes ---
        for (process_idx, process) in self.processes.iter().enumerate() {
            // Products -> Diagonal of A matrix
            if process.products.len() != 1 {
                return Err(LcaModelCompilationError::ProcessProductCardinalityError(
                    process.name.clone(),
                    process.products.len(),
                ));
            }
            let output = &process.products[0];
            if output.product.name != process.name {
                return Err(LcaModelCompilationError::ProcessProductMismatch(
                    process.name.clone(),
                    output.product.name.clone(),
                ));
            }
            let amount = output.amount.evaluate()?;
            if amount <= 0.0 {
                return Err(LcaModelCompilationError::NonPositiveProductAmount(
                    process.name.clone(),
                    output.product.name.clone(),
                    amount,
                ));
            }
            // Use Triplete::new constructor
            a_triplets.push(Triplete::new(process_idx, process_idx, amount));

            // Inputs -> Off-diagonal of A matrix or unlinked
            for input in &process.inputs {
                let amount = input.amount.evaluate()?;
                if amount < 0.0 {
                    return Err(LcaModelCompilationError::NegativeInputAmount(
                        process.name.clone(),
                        format!("{:?}", input.product),
                        amount,
                    ));
                }
                match &input.product {
                    ProductRef::Product(product_name) => {
                        if let Some(&producer_idx) = process_map.get(product_name) {
                            // Use Triplete::new constructor
                            a_triplets.push(Triplete::new(producer_idx, process_idx, -amount));
                        } else {
                            return Err(LcaModelCompilationError::UnresolvedInternalProductRef(
                                product_name.clone(),
                            ));
                        }
                    }
                    ProductRef::External(ext_ref) => {
                        unlinked_inputs.push(ext_ref.clone());
                    }
                }
            }

            // Emissions & Resources -> B matrix or unlinked
            let process_biosphere = process.emissions.iter().chain(process.resources.iter());
            for bio_item in process_biosphere {
                let amount = bio_item.amount.evaluate()?;
                match &bio_item.substance {
                    SubstanceRef::Substance(substance_name) => {
                        if let Some(&substance_idx) = substance_map.get(substance_name) {
                            // Use Triplete::new constructor
                            b_triplets.push(Triplete::new(substance_idx, process_idx, amount));
                        } else {
                            return Err(LcaModelCompilationError::UnresolvedInternalSubstanceRef(
                                substance_name.clone(),
                            ));
                        }
                    }
                    SubstanceRef::External(ext_ref) => {
                        unlinked_biosphere_items.push(ext_ref.clone());
                    }
                }
            }
        }

        // --- Process Evaluation Demands ---
        if !self.evaluation_demands.is_empty()
            && !self.evaluation_demands.iter().all(|v| v.is_empty())
        {
            for demand_item in self.evaluation_demands.iter().flatten() {
                let original_product_name = &demand_item.product; // Assuming DemandItem has product: String
                let eval_demand_process_name = format!("eval_demand_{}", original_product_name);

                if process_map.contains_key(&eval_demand_process_name)
                    || evaluation_demand_process_names_for_system
                        .contains(&eval_demand_process_name)
                {
                    return Err(LcaModelCompilationError::EvaluationDemandNameClash(
                        eval_demand_process_name.clone(),
                    ));
                }

                let original_product_idx = match process_map.get(original_product_name) {
                    Some(&idx) => idx,
                    None => {
                        return Err(LcaModelCompilationError::EvaluationDemandTargetNotFound(
                            original_product_name.clone(),
                        ));
                    }
                };

                let eval_process_actual_idx = num_processes + current_eval_demand_idx_offset;
                process_ids.push(eval_demand_process_name.clone());
                process_map.insert(eval_demand_process_name.clone(), eval_process_actual_idx);
                evaluation_demand_process_names_for_system.push(eval_demand_process_name.clone());

                // Output of the new eval demand process (produces 1 unit of "demand satisfaction")
                a_triplets.push(Triplete::new(
                    eval_process_actual_idx,
                    eval_process_actual_idx,
                    demand_item.amount,
                ));

                // Input to the new eval demand process (consumes original_product_amount of the original_product_name)
                // Assuming DemandItem in LcaModel has an Amount field similar to other items.
                // If DemandItem directly stores f32, adjust this.
                // For now, assuming DemandItem has `amount: Amount` like other model items.
                // If `crate::DemandItem` is used here, it has `amount: f32`.
                // The `evaluation_demands: Vec<Vec<DemandItem>>` in `LcaModel` uses `crate::DemandItem`.
                let evaluated_amount = demand_item.amount; // Direct f32 from crate::DemandItem
                if evaluated_amount < 0.0 {
                    // Optional: Add a specific error for negative evaluation demand amounts if desired
                    return Err(LcaModelCompilationError::NegativeInputAmount(
                        eval_demand_process_name.clone(), // "process" is the eval demand
                        original_product_name.clone(),    // "product" is what it consumes
                        evaluated_amount,
                    ));
                }
                a_triplets.push(Triplete::new(
                    original_product_idx,
                    eval_process_actual_idx,
                    -evaluated_amount,
                ));

                current_eval_demand_idx_offset += 1;
            }
        }

        let total_num_processes = num_processes + current_eval_demand_idx_offset;

        // --- Create Sparse Matrices ---
        // Use ? to propagate potential errors from from_triplets
        let a_sparse_matrix =
            SparseMatrix::from_triplets(total_num_processes, total_num_processes, a_triplets)?;
        let b_sparse_matrix =
            SparseMatrix::from_triplets(num_substances, total_num_processes, b_triplets)?;
        let c_sparse_matrix = SparseMatrix::from_triplets(num_impacts, num_substances, c_triplets)?;

        // --- Create LcaMatrix wrappers ---
        // Use ? to propagate potential errors from LcaMatrix::new
        // Note: process_ids has been augmented with evaluation demand process names
        let a_lca_matrix =
            LcaMatrix::new(a_sparse_matrix, process_ids.clone(), process_ids.clone()) // A: processes x processes
                .map_err(|e| {
                    LcaModelCompilationError::UnableToCreateSystemMatrix(format!("A: {}", e))
                })?;
        let b_lca_matrix =
            LcaMatrix::new(b_sparse_matrix, process_ids.clone(), substance_ids.clone()) // B: substances x processes
                .map_err(|e| {
                    LcaModelCompilationError::UnableToCreateSystemMatrix(format!("B: {}", e))
                })?;
        let c_lca_matrix = LcaMatrix::new(c_sparse_matrix, substance_ids, impact_ids) // C: impacts x substances
            .map_err(|e| {
                LcaModelCompilationError::UnableToCreateSystemMatrix(format!("C: {}", e))
            })?;

        // --- Create LcaSystem ---
        // Use ? to propagate potential errors from LcaSystem::new
        let compiled_system = LcaSystem::new(
            a_lca_matrix,
            b_lca_matrix,
            c_lca_matrix,
            Some(evaluation_demand_process_names_for_system),
        )
        .map_err(|e| LcaModelCompilationError::UnableToLcaSystem(format!("System: {}", e)))?;

        Ok(CompiledLcaModel {
            lca_model: self, // Move the original model into the result
            compiled_system,
            unlinked_inputs,
            unlinked_biosphere_items,
        })
    }
}

pub struct CompiledLcaModel {
    pub lca_model: LcaModel,
    pub compiled_system: LcaSystem,
    pub unlinked_inputs: Vec<ExternalRef>,
    pub unlinked_biosphere_items: Vec<ExternalRef>,
}

#[cfg(test)]
mod tests {
    use crate::DemandItem;

    use super::*; // Import items from parent module (model.rs)

    #[test]
    fn test_compile_bike_example() {
        // example taken from https://github.com/brightway-lca/from-the-ground-up/blob/main/1%20-%20The%20supply%20chain%20graph.ipynb
        let demand = vec![DemandItem::new("Bike production DK".to_string(), 1.0)];
        let model = LcaModel {
            database_name: "Bike Example DB".to_string(),
            imported_dbs: vec![],
            substances: vec![
                Substance {
                    name: "CO2".to_string(),
                    r#type: SubstanceType::Emission,
                    compartment: None, // Assuming default compartment for simplicity
                    sub_compartment: None,
                    reference_unit: "kg".to_string(),
                    impacts: vec![Impact {
                        name: "GWP100".to_string(),
                        amount: Amount::Literal(1.0),
                        unit: "kg CO2 eq".to_string(),
                    }],
                },
                // Note: Natural Gas is treated as a product/process, not a raw resource substance here
            ],
            processes: vec![
                Process {
                    name: "Bike production DK".to_string(),
                    products: vec![OutputItem {
                        product: Product {
                            name: "Bike production DK".to_string(),
                        },
                        amount: Amount::Literal(1.0),
                        unit: "bike".to_string(), // Unit consistency check is not implemented yet
                    }],
                    inputs: vec![InputItem {
                        product: ProductRef::Product("Carbon fibre DE".to_string()),
                        amount: Amount::Literal(2.5),
                        unit: "kg".to_string(),
                    }],
                    emissions: vec![],
                    resources: vec![],
                },
                Process {
                    name: "Carbon fibre DE".to_string(),
                    products: vec![OutputItem {
                        product: Product {
                            name: "Carbon fibre DE".to_string(),
                        },
                        amount: Amount::Literal(1.0),
                        unit: "kg".to_string(),
                    }],
                    inputs: vec![InputItem {
                        product: ProductRef::Product("Natural gas NO".to_string()),
                        amount: Amount::Literal(237.0),
                        unit: "MJ".to_string(),
                    }],
                    emissions: vec![BiosphereItem {
                        substance: SubstanceRef::Substance("CO2".to_string()),
                        amount: Amount::Literal(26.6),
                        unit: "kg".to_string(),
                        compartment: None,
                    }],
                    resources: vec![],
                },
                Process {
                    name: "Natural gas NO".to_string(),
                    products: vec![OutputItem {
                        product: Product {
                            name: "Natural gas NO".to_string(),
                        },
                        amount: Amount::Literal(1.0), // Reference output is 1 MJ
                        unit: "MJ".to_string(),
                    }],
                    inputs: vec![], // Assuming this is an elementary flow source process
                    emissions: vec![],
                    resources: vec![], // Or maybe this should be a resource input? Depends on system boundary. Treating as process for now.
                },
            ],
            evaluation_demands: vec![demand.clone()],
        };

        let compile_result = model.compile();

        assert!(
            compile_result.is_ok(),
            "Compilation failed: {:?}",
            compile_result.err()
        );

        let compiled_model = compile_result.unwrap();

        assert!(
            compiled_model.unlinked_inputs.is_empty(),
            "Found unlinked inputs: {:?}",
            compiled_model.unlinked_inputs
        );
        assert!(
            compiled_model.unlinked_biosphere_items.is_empty(),
            "Found unlinked biosphere items: {:?}",
            compiled_model.unlinked_biosphere_items
        );

        // Optional: Check matrix dimensions or content if needed
        let system = compiled_model.compiled_system;
        // Use accessor methods
        let (a_rows, a_cols) = system.a_matrix().matrix().dims();
        let (b_rows, b_cols) = system.b_matrix().matrix().dims();
        let (c_rows, c_cols) = system.c_matrix().matrix().dims();

        assert_eq!(a_rows, 4, "A matrix should have 4 rows (processes)");
        assert_eq!(a_cols, 4, "A matrix should have 4 columns (processes)");
        assert_eq!(b_rows, 1, "B matrix should have 1 row (substances)");
        assert_eq!(b_cols, 4, "B matrix should have 4 columns (processes)");
        assert_eq!(c_rows, 1, "C matrix should have 0 rows (impacts)");
        assert_eq!(c_cols, 1, "C matrix should have 1 column (substances)");

        // Check some specific values (optional, adjust indices based on process/substance order)
        // Process order: Bike(0), CarbonFibre(1), NaturalGas(2)
        // Substance order: CO2(0)
        // A[0,0] = 1.0 (Bike output)
        // A[1,1] = 1.0 (Carbon Fibre output)
        // A[2,2] = 1.0 (Natural Gas output)
        // A[1,0] = -2.5 (Carbon Fibre input to Bike)
        // A[2,1] = -237.0 (Natural Gas input to Carbon Fibre)
        // A[3,3] = 1 (eval_demand_Bike production DK)
        // B[0,1] = 26.6 (CO2 emission from Carbon Fibre)
        // C[0,0] = 1.0 (GWP100 impact from CO2)

        // Use accessor methods
        assert_eq!(system.a_matrix().matrix().get(0, 0), Some(1.0));
        assert_eq!(system.a_matrix().matrix().get(1, 1), Some(1.0));
        assert_eq!(system.a_matrix().matrix().get(2, 2), Some(1.0));
        assert_eq!(system.a_matrix().matrix().get(1, 0), Some(-2.5));
        assert_eq!(system.a_matrix().matrix().get(2, 1), Some(-237.0));
        assert_eq!(system.a_matrix().matrix().get(3, 3), Some(1.0));
        assert_eq!(system.b_matrix().matrix().get(0, 1), Some(26.6));

        // Check evaluation demand process names
        assert_eq!(
            system
                .evaluation_demand_process_names()
                .as_ref()
                .unwrap()
                .len(),
            1
        );
        assert_eq!(
            system.evaluation_demand_process_names().as_ref().unwrap()[0],
            "eval_demand_Bike production DK"
        );

        // Check IDs using accessor methods
        assert_eq!(
            system.a_matrix().col_ids(),
            &[
                "Bike production DK",
                "Carbon fibre DE",
                "Natural gas NO",
                "eval_demand_Bike production DK"
            ]
        );
        assert_eq!(
            system.a_matrix().row_ids(),
            &[
                "Bike production DK",
                "Carbon fibre DE",
                "Natural gas NO",
                "eval_demand_Bike production DK"
            ]
        );
        assert_eq!(
            system.b_matrix().col_ids(),
            &[
                "Bike production DK",
                "Carbon fibre DE",
                "Natural gas NO",
                "eval_demand_Bike production DK"
            ]
        );
        assert_eq!(system.b_matrix().row_ids(), &["CO2"]);
        assert_eq!(system.c_matrix().col_ids(), &["CO2"]);
        assert_eq!(system.c_matrix().row_ids(), &["GWP100"]);

        pollster::block_on(async {
            let device = lca_core::GpuDevice::new()
                .await
                .expect("Failed to create GPU device");

            let result = system
                .evaluate(
                    &device,
                    Some(demand.clone()),
                    Some(vec!["GWP100".to_string()]),
                )
                .await
                .expect("LCA evaluation failed");
            assert_eq!(result.len(), 1, "LCA result should have 1 element");
            assert_eq!(
                result[0], 66.5,
                "LCA result should be 1.0 for the bike production demand"
            );
        })
    }
}
