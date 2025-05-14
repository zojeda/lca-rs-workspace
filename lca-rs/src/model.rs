use lca_core::{SparseMatrix, sparse_matrix::Triplete};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::{
    DemandItem, LcaMatrix, LcaSystem, error::LcaModelCompilationError, lca_system::InterSystemLink,
};
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LcaModel {
    pub database_name: String,
    pub case_name: String,
    pub case_description: String,
    pub imported_dbs: Vec<DbImport>,
    pub processes: Vec<Process>,
    pub substances: Vec<Substance>,
    pub evaluation_demands: Vec<Vec<DemandItem>>,
    pub evaluation_impacts: Vec<ImpactRef>,
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
    Literal(f64),
}

impl Amount {
    pub fn evaluate(&self) -> core::result::Result<f64, LcaModelCompilationError> {
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

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ImpactRef {
    Impact(String),
    External(ExternalRef),
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
    pub fn compile(self) -> core::result::Result<LcaSystem, LcaModelCompilationError> {
        let num_processes = self.processes.len();
        let num_substances = self.substances.len();

        // --- Create Mappings and ID Lists ---
        let mut process_map: HashMap<String, usize> = HashMap::with_capacity(num_processes);
        let mut process_ids: Vec<String> = Vec::with_capacity(num_processes);
        for (i, process) in self.processes.iter().enumerate() {
            let process_product_id = format!(
                "{}|{}",
                process.name.clone(),
                process.products[0].product.name
            );
            if process_map.insert(process_product_id.clone(), i).is_some() {
                return Err(LcaModelCompilationError::DuplicateProcessName(
                    process.name.clone(),
                ));
            }
            process_ids.push(process_product_id.clone());
        }

        // --- Initialize for Evaluation Demands ---
        let mut evaluation_demand: Vec<String> = Vec::new();
        let mut evaluation_methods: Vec<String> = Vec::new();

        // --- Populate Evaluation Demands ---
        for demand_group in &self.evaluation_demands {
            for demand_item in demand_group {
                if !process_map.contains_key(&demand_item.product) {
                    return Err(LcaModelCompilationError::UnresolvedInternalProductRef(
                        demand_item.product.clone(),
                    ));
                }
                evaluation_demand.push(demand_item.product.clone());
            }
        }

        // --- Populate Evaluation Methods ---
        for impact_ref in &self.evaluation_impacts {
            match impact_ref {
                ImpactRef::Impact(impact_name) => {
                    if !evaluation_methods.contains(impact_name) {
                        evaluation_methods.push(impact_name.clone());
                    }
                }
                ImpactRef::External(ext_ref) => {
                    let db = self
                        .imported_dbs
                        .iter()
                        .find(|db| db.alias == ext_ref.alias)
                        .ok_or_else(|| {
                            LcaModelCompilationError::UnresolvedExternalImpactRef(
                                ext_ref.alias.clone(),
                            )
                        })?;
                    let complete_external_name = format!("{}:{}", db.name, ext_ref.name);
                    if !evaluation_methods.contains(&complete_external_name) {
                        evaluation_methods.push(complete_external_name);
                    }
                }
            }
        }

        // --- Initialize Substance and Impact Mappings ---



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
        let mut a_links_data: Vec<InterSystemLink> = Vec::new();
        let mut b_links_data: Vec<InterSystemLink> = Vec::new();

        // --- Iterate Through Processes ---
        for (process_idx, process) in self.processes.iter().enumerate() {
            // Products -> Diagonal of A matrix
            if process.products.len() != 1 {
                return Err(LcaModelCompilationError::ProcessProductCardinalityError(
                    process.name.clone(),
                    process.products.len(),
                ));
            }
            if process.products.len() > 1 {
                return Err(LcaModelCompilationError::MultiProductUnsupported(
                    process.name.clone(),
                ));
            }
            let output = &process.products[0];

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
                        let amount_val = amount; // Already evaluated amount for the input
                        unlinked_inputs.push(ext_ref.clone());
                        a_links_data.push(InterSystemLink {
                            local_source_row_id: format!(
                                "{}|{}",
                                process.name.clone(),
                                output.product.name
                            ),
                            target_system_name: self
                                .imported_dbs
                                .iter()
                                .find(|db| db.alias == ext_ref.alias)
                                .map(|db| db.name.clone())
                                .unwrap_or_else(|| ext_ref.alias.clone()),
                            target_system_row_id: ext_ref.name.clone(),
                            value: -amount_val, // Inputs to A matrix are typically negative
                        });
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
                        let amount_val = amount; // Already evaluated amount for the biosphere item
                        unlinked_biosphere_items.push(ext_ref.clone());
                        b_links_data.push(InterSystemLink {
                            local_source_row_id: process.name.clone(),
                            target_system_name: ext_ref.alias.clone(),
                            target_system_row_id: ext_ref.name.clone(),
                            value: amount_val, // Biosphere flows in B matrix
                        });
                    }
                }
            }
        }

        let total_num_processes = num_processes; // current_eval_demand_idx_offset is now always 0

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
        let c_links_data: Vec<InterSystemLink> = Vec::new(); // As per "ignore for now"



        let compiled_system = LcaSystem::new(
            self.database_name.clone(),
            a_lca_matrix,
            b_lca_matrix,
            c_lca_matrix,
            Some(evaluation_demand),
            Some(evaluation_methods),
            a_links_data,
            b_links_data,
            c_links_data,
        )
        .map_err(|e| LcaModelCompilationError::UnableToLcaSystem(format!("System: {}", e)))?;

        Ok(compiled_system)
    }
}

#[cfg(test)]
mod tests {

    use crate::DemandItem;

    use super::*; // Import items from parent module (model.rs)

    #[test]
    fn test_compile_bike_example() {
        // example taken from https://github.com/brightway-lca/from-the-ground-up/blob/main/1%20-%20The%20supply%20chain%20graph.ipynb
        let demand = vec![DemandItem::new("Bike production DK|Bike".to_string(), 1.0)];
        let model = LcaModel {
            case_name: "Bike Example".to_string(),
            case_description: "Bike Example Description".to_string(),
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
                            name: "Bike".to_string(),
                        },
                        amount: Amount::Literal(1.0),
                        unit: "bike".to_string(), // Unit consistency check is not implemented yet
                    }],
                    inputs: vec![InputItem {
                        product: ProductRef::Product("Carbon fibre DE|Carbon fibre DE".to_string()),
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
                        product: ProductRef::Product("Natural gas NO|Natural gas NO".to_string()),
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
            evaluation_impacts: vec![ImpactRef::Impact("GWP100".to_string())],
        };

        println!("model: {}", serde_json::to_string_pretty(&model).unwrap());
        let compile_result = model.compile();

        assert!(
            compile_result.is_ok(),
            "Compilation failed: {:?}",
            compile_result.err()
        );

        let system = compile_result.unwrap();

        let (a_rows, a_cols) = system.a_matrix().matrix().dims();
        let (b_rows, b_cols) = system.b_matrix().matrix().dims();
        let (c_rows, c_cols) = system.c_matrix().matrix().dims();

        assert_eq!(a_rows, 3, "A matrix should have 3 rows (processes)");
        assert_eq!(a_cols, 3, "A matrix should have 3 columns (processes)");
        assert_eq!(b_rows, 1, "B matrix should have 1 row (substances)");
        assert_eq!(b_cols, 3, "B matrix should have 3 columns (processes)");
        assert_eq!(c_rows, 1, "C matrix should have 1 rows (impacts)"); // Corrected from 0 to 1 as per original test logic for C
        assert_eq!(c_cols, 1, "C matrix should have 1 column (substances)");

        // Process order: Bike(0), CarbonFibre(1), NaturalGas(2)
        // Substance order: CO2(0)
        // A[0,0] = 1.0 (Bike output)
        // A[1,1] = 1.0 (Carbon Fibre output)
        // A[2,2] = 1.0 (Natural Gas output)
        // A[1,0] = -2.5 (Carbon Fibre input to Bike)
        // A[2,1] = -237.0 (Natural Gas input to Carbon Fibre)
        // B[0,1] = 26.6 (CO2 emission from Carbon Fibre)
        // C[0,0] = 1.0 (GWP100 impact from CO2)

        // Use accessor methods
        assert_eq!(system.a_matrix().matrix().get(0, 0), Some(1.0));
        assert_eq!(system.a_matrix().matrix().get(1, 0), Some(-2.5));
        assert_eq!(system.a_matrix().matrix().get(1, 1), Some(1.0));
        assert_eq!(system.a_matrix().matrix().get(2, 2), Some(1.0));
        assert_eq!(system.a_matrix().matrix().get(2, 1), Some(-237.0));

        assert_eq!(system.b_matrix().matrix().get(0, 1), Some(26.6));

        // Check evaluation demand process names
        assert_eq!(
            system
                .evaluation_demand_process_names()
                .as_ref()
                .unwrap()
                .len(),
            1 // Expect 1 evaluation demand processes
        );

        // Check IDs using accessor methods
        assert_eq!(
            system.a_matrix().col_ids(),
            &[
                "Bike production DK|Bike",
                "Carbon fibre DE|Carbon fibre DE",
                "Natural gas NO|Natural gas NO",
            ]
        );
        assert_eq!(
            system.a_matrix().row_ids(),
            &[
                "Bike production DK|Bike",
                "Carbon fibre DE|Carbon fibre DE",
                "Natural gas NO|Natural gas NO",
            ]
        );
        assert_eq!(
            system.b_matrix().col_ids(),
            &[
                "Bike production DK|Bike",
                "Carbon fibre DE|Carbon fibre DE",
                "Natural gas NO|Natural gas NO",
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
                    None,
                    None,
                )
                .await
                .expect("Failed to evaluate LCA model");

            assert_eq!(result.len(), 1, "LCA result should have 1 element");
            let expected_co2 = 66.5; // CO2 emissions from Carbon Fibre production
            assert!(
                (result[0] - expected_co2).abs() < 1e-3,
                "LCA result for GWP100 is incorrect. Expected: {}, Got: {}",
                expected_co2,
                result[0]
            );
        })
    }

    #[test]
    fn test_compile_toaster_example() {
        // Based on https://youtu.be/hjjRjrmljn4?t=588
        // Demand is for 1 unit of "Disposed Toaster" to capture the full lifecycle.
        let demand = vec![DemandItem::new(
            "Toaster disposal|Disposed Toaster".to_string(),
            1.0,
        )];

        let model = LcaModel {
            database_name: "Toaster Example DB".to_string(),
            case_name: "Toaster Lifecycle CO2".to_string(),
            case_description: "Calculates the total CO2 emissions for a toaster's lifecycle."
                .to_string(),
            imported_dbs: vec![],
            substances: vec![Substance {
                name: "CO2".to_string(),
                r#type: SubstanceType::Emission,
                compartment: Some("air".to_string()), // Example compartment
                sub_compartment: None,
                reference_unit: "kg".to_string(),
                impacts: vec![Impact {
                    name: "GWP100".to_string(),
                    amount: Amount::Literal(1.0), // 1 kg CO2 = 1 kg CO2-eq
                    unit: "kg CO2-eq".to_string(),
                }],
            }],
            processes: vec![
                // Process 0: Steel Production
                Process {
                    name: "Steel production".to_string(),
                    products: vec![OutputItem {
                        product: Product {
                            name: "Steel".to_string(),
                        },
                        amount: Amount::Literal(1.0),
                        unit: "kg".to_string(),
                    }],
                    inputs: vec![InputItem {
                        product: ProductRef::Product("Steam production|Steam".to_string()),
                        amount: Amount::Literal(0.5), // 0.5 MJ Steam per kg Steel
                        unit: "MJ".to_string(),
                    }],
                    emissions: vec![BiosphereItem {
                        substance: SubstanceRef::Substance("CO2".to_string()),
                        amount: Amount::Literal(1.0), // 1 kg CO2 per kg Steel
                        unit: "kg".to_string(),
                        compartment: Some("air".to_string()),
                    }],
                    resources: vec![],
                },
                // Process 1: Steam Production
                Process {
                    name: "Steam production".to_string(),
                    products: vec![OutputItem {
                        product: Product {
                            name: "Steam".to_string(),
                        },
                        amount: Amount::Literal(1.0), // Produces 1 MJ Steam (reference)
                        unit: "MJ".to_string(),
                    }],
                    inputs: vec![InputItem {
                        product: ProductRef::Product("Steel production|Steel".to_string()),
                        amount: Amount::Literal(0.25), // 0.25 kg Steel per MJ Steam
                        unit: "kg".to_string(),
                    }],
                    emissions: vec![BiosphereItem {
                        substance: SubstanceRef::Substance("CO2".to_string()),
                        amount: Amount::Literal(4.0), // 4 kg CO2 per MJ Steam
                        unit: "kg".to_string(),
                        compartment: Some("air".to_string()),
                    }],
                    resources: vec![],
                },
                // Process 2: Toaster Production
                Process {
                    name: "Toaster production".to_string(),
                    products: vec![OutputItem {
                        product: Product {
                            name: "Toaster".to_string(),
                        },
                        amount: Amount::Literal(1.0), // Produces 1 unit Toaster
                        unit: "unit".to_string(),
                    }],
                    inputs: vec![
                        InputItem {
                            product: ProductRef::Product("Steel production|Steel".to_string()),
                            amount: Amount::Literal(1.0), // 1 kg Steel per Toaster
                            unit: "kg".to_string(),
                        },
                        InputItem {
                            product: ProductRef::Product("Steam production|Steam".to_string()),
                            amount: Amount::Literal(0.5), // 0.5 MJ Steam per Toaster
                            unit: "MJ".to_string(),
                        },
                    ],
                    emissions: vec![BiosphereItem {
                        substance: SubstanceRef::Substance("CO2".to_string()),
                        amount: Amount::Literal(2.0), // 2 kg CO2 per unit Toaster production
                        unit: "kg".to_string(),
                        compartment: Some("air".to_string()),
                    }],
                    resources: vec![],
                },
                // Process 3: Toaster Use
                Process {
                    name: "Toaster use".to_string(),
                    products: vec![OutputItem {
                        product: Product {
                            name: "Used Toaster".to_string(),
                        }, // Represents the service of a used toaster
                        amount: Amount::Literal(1.0),
                        unit: "unit".to_string(),
                    }],
                    inputs: vec![InputItem {
                        // Consumes a new toaster
                        product: ProductRef::Product("Toaster production|Toaster".to_string()),
                        amount: Amount::Literal(1.0),
                        unit: "unit".to_string(),
                    }],
                    emissions: vec![BiosphereItem {
                        // 0.001 kg CO2/piece of bread, assume 1000 pieces
                        substance: SubstanceRef::Substance("CO2".to_string()),
                        amount: Amount::Literal(1.0), // 0.001 kg/piece * 1000 pieces
                        unit: "kg".to_string(),
                        compartment: Some("air".to_string()),
                    }],
                    resources: vec![],
                },
                // Process 4: Toaster Disposal
                Process {
                    name: "Toaster disposal".to_string(),
                    products: vec![OutputItem {
                        product: Product {
                            name: "Disposed Toaster".to_string(),
                        }, // Final product of the lifecycle
                        amount: Amount::Literal(1.0),
                        unit: "unit".to_string(),
                    }],
                    inputs: vec![InputItem {
                        // Consumes a used toaster
                        product: ProductRef::Product("Toaster use|Used Toaster".to_string()),
                        amount: Amount::Literal(1.0),
                        unit: "unit".to_string(),
                    }],
                    emissions: vec![BiosphereItem {
                        substance: SubstanceRef::Substance("CO2".to_string()),
                        amount: Amount::Literal(0.5), // 0.5 kg CO2 per unit Toaster disposal
                        unit: "kg".to_string(),
                        compartment: Some("air".to_string()),
                    }],
                    resources: vec![],
                },
            ],
            evaluation_demands: vec![demand.clone()],
            evaluation_impacts: vec![
                ImpactRef::Impact("GWP100".to_string()), // CO2 emissions
            ],
        };

        let compile_result = model.compile();
        assert!(
            compile_result.is_ok(),
            "Compilation failed: {:?}",
            compile_result.err()
        );

        let system = compile_result.unwrap();

        let (a_rows, a_cols) = system.a_matrix().matrix().dims();
        let (b_rows, b_cols) = system.b_matrix().matrix().dims();
        let (c_rows, c_cols) = system.c_matrix().matrix().dims();

        assert_eq!(a_rows, 5, "A matrix should have 5 rows (processes)");
        assert_eq!(a_cols, 5, "A matrix should have 5 columns (processes)");
        assert_eq!(b_rows, 1, "B matrix should have 1 row (substances)");
        assert_eq!(b_cols, 5, "B matrix should have 5 columns (processes)");
        assert_eq!(c_rows, 1, "C matrix should have 1 row (impacts)");
        assert_eq!(c_cols, 1, "C matrix should have 1 column (substances)");

        // Expected process IDs (name|product_name)
        let expected_process_ids = vec![
            "Steel production|Steel".to_string(),
            "Steam production|Steam".to_string(),
            "Toaster production|Toaster".to_string(),
            "Toaster use|Used Toaster".to_string(),
            "Toaster disposal|Disposed Toaster".to_string(),
        ];
        assert_eq!(system.a_matrix().col_ids(), &expected_process_ids);
        assert_eq!(system.a_matrix().row_ids(), &expected_process_ids);
        assert_eq!(system.b_matrix().col_ids(), &expected_process_ids);
        assert_eq!(system.b_matrix().row_ids(), &["CO2".to_string()]);
        assert_eq!(system.c_matrix().col_ids(), &["CO2".to_string()]);
        assert_eq!(system.c_matrix().row_ids(), &["GWP100".to_string()]);

        // Check some key matrix values
        // A Matrix:
        // Steel production (idx 0): Output Steel (A[0,0]=1), Input Steam (A[1,0]=-0.5)
        assert_eq!(system.a_matrix().matrix().get(0, 0), Some(1.0)); // Steel output
        assert_eq!(system.a_matrix().matrix().get(1, 0), Some(-0.5)); // Steam input to Steel

        // Steam production (idx 1): Output Steam (A[1,1]=1), Input Steel (A[0,1]=-0.25)
        assert_eq!(system.a_matrix().matrix().get(1, 1), Some(1.0)); // Steam output
        assert_eq!(system.a_matrix().matrix().get(0, 1), Some(-0.25)); // Steel input to Steam

        // Toaster production (idx 2): Output Toaster (A[2,2]=1), Input Steel (A[0,2]=-1), Input Steam (A[1,2]=-0.5)
        assert_eq!(system.a_matrix().matrix().get(2, 2), Some(1.0)); // Toaster output
        assert_eq!(system.a_matrix().matrix().get(0, 2), Some(-1.0)); // Steel input to Toaster
        assert_eq!(system.a_matrix().matrix().get(1, 2), Some(-0.5)); // Steam input to Toaster

        // Toaster use (idx 3): Output Used Toaster (A[3,3]=1), Input Toaster (A[2,3]=-1)
        assert_eq!(system.a_matrix().matrix().get(3, 3), Some(1.0));
        assert_eq!(system.a_matrix().matrix().get(2, 3), Some(-1.0));

        // Toaster disposal (idx 4): Output Disposed Toaster (A[4,4]=1), Input Used Toaster (A[3,4]=-1)
        assert_eq!(system.a_matrix().matrix().get(4, 4), Some(1.0));
        assert_eq!(system.a_matrix().matrix().get(3, 4), Some(-1.0));

        // B Matrix (CO2 emissions):
        // Steel production (col 0): 1 kg CO2 (B[0,0]=1)
        assert_eq!(system.b_matrix().matrix().get(0, 0), Some(1.0));
        // Steam production (col 1): 4 kg CO2 (B[0,1]=4)
        assert_eq!(system.b_matrix().matrix().get(0, 1), Some(4.0));
        // Toaster production (col 2): 2 kg CO2 (B[0,2]=2)
        assert_eq!(system.b_matrix().matrix().get(0, 2), Some(2.0));
        // Toaster use (col 3): 1 kg CO2 (B[0,3]=1)
        assert_eq!(system.b_matrix().matrix().get(0, 3), Some(1.0));
        // Toaster disposal (col 4): 0.5 kg CO2 (B[0,4]=0.5)
        assert_eq!(system.b_matrix().matrix().get(0, 4), Some(0.5));

        // C Matrix:
        // GWP100 from CO2 (C[0,0]=1)
        assert_eq!(system.c_matrix().matrix().get(0, 0), Some(1.0));

        pollster::block_on(async {
            let device = lca_core::GpuDevice::new()
                .await
                .expect("Failed to create GPU device");

            let result = system
                .evaluate(
                    &device,
                    None,
                    None,
                )
                .await
                .expect("LCA evaluation failed");

            assert_eq!(
                result.len(),
                1,
                "LCA result should have 1 element for the GWP100 impact"
            );
            let expected_co2 = 65.5 / 7.0;
            assert!(
                (result[0] - expected_co2).abs() < 1e-9,
                "LCA result for GWP100 is incorrect. Expected: {}, Got: {}",
                expected_co2,
                result[0]
            );
        })
    }
}
