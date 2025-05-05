use std::{error::Error, vec}; // Added exit for handling breakdown

use lca_core::{GpuDevice, SparseMatrix, sparse_matrix::Triplete}; // Added LsolverError
use lca_rs::{
    human_size, model::{Amount, DbImport, ExternalRef, InputItem, LcaModel, OutputItem, Process, Product, ProductRef}, DemandItem, LcaMatrix, LcaSystem
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .filter_module("wgpu", log::LevelFilter::Off)
        .filter_module("naga", log::LevelFilter::Off)
        .init();

    // Load the LCA system
    let water_bottle_lca_system = create_water_botle_lca_system()?;
    water_bottle_lca_system.a_links().iter().for_each(|link| {
        println!("A link: {:?}", link);
    });
    println!("Water bottle LCA system loaded");

    
    let ecoinvent_system = load_ecoinvent_lca_system()?;
    println!("EcoInvent system loaded");

    let lca_system = LcaSystem::combine(vec![ecoinvent_system, water_bottle_lca_system])?;
    
    println!("Combined A matrix: {:?}", lca_system.a_matrix().matrix().dims());
    println!("Combined B matrix: {:?}", lca_system.b_matrix().matrix().dims());
    println!("Combined C matrix: {:?}", lca_system.c_matrix().matrix().dims());

    for triplete in lca_system.a_matrix().matrix().iter() {
        if triplete.row() >=  25412 {
            println!("A matrix triplet: {:?}", triplete);
        }
    }
    println!("B matrix row id 562: {:?}", lca_system.b_matrix().row_id(562));



    let gpu = GpuDevice::new().await?;
    let demand = vec![DemandItem::new(
        "Water bottle LCA::Drinking water from a bottle|Drinking 1 liter from a water bottle".to_string(),
        1.0,
    )];
    let start_time = std::time::Instant::now();
    let lca_result = lca_system
        .evaluate(
            &gpu,
            Some(demand),
            Some(vec![
                // Restore original requested methods
                "ecoinvent-3.11::EF v3.1|climate change|global warming potential (GWP100)".to_string(),
                "ecoinvent-3.11::EF v3.1|climate change: biogenic|global warming potential (GWP100)".to_string(),
                "ecoinvent-3.11::EF v3.1|climate change: fossil|global warming potential (GWP100)".to_string(),
                "ecoinvent-3.11::EF v3.1|climate change: land use and land use change|global warming potential (GWP100)".to_string(),

            ]),
        )
        .await?;
    let elapsed_time = start_time.elapsed();
    println!("Elapsed time: {:?}", elapsed_time);
    println!("result length: {}", lca_result.len());
    println!("Result: {:?}", lca_result);
    let ts = gpu.get_transfer_stats();
    log::info!("Bytes transferred To GPU: {}", human_size(ts.bytes_to_gpu));
    log::info!(
        "Bytes transferred From GPU: {}",
        human_size(ts.bytes_from_gpu)
    );

    Ok(())
}

fn create_water_botle_lca_system() -> Result<lca_rs::LcaSystem, Box<dyn Error>> {
    let model = LcaModel {
        case_name: "water bottle production".to_string(),
        case_description: "Water bottle LCA Description".to_string(),
        database_name: "Water bottle LCA".to_string(),
        imported_dbs: vec![DbImport {
            name: "ecoinvent-3.11".to_string(),
            alias: "ei".to_string(),
        }],
        substances: vec![], // Remove dummy substance
        evaluation_demands: vec![],
        processes: vec![
          Process {
            name: "water bottle production".to_string(),
            products: vec![OutputItem {
                product: Product {
                    name: "water bottle".to_string(), //FIXME allow different name from process
                },
                amount: Amount::Literal(1.0),
                unit: "unit".to_string(), // Unit consistency check is not implemented yet
            }],
            inputs: vec![
              InputItem {
                product: ProductRef::External(ExternalRef {
                  alias: "ei".to_string(),
                  name: "treatment of aluminium scrap, post-consumer, prepared for recycling, at remelter|RoW|aluminium, wrought alloy".to_string(),
                }),
                amount: Amount::Literal(0.33),
                unit: "kg".to_string(),
              },
              InputItem {
                product: ProductRef::External(ExternalRef {
                  alias: "ei".to_string(),
                  name: "impact extrusion of aluminium, deformation stroke|RoW|impact extrusion of aluminium, deformation stroke".to_string(),
                }),
                amount: Amount::Literal(0.33),
                unit: "kg".to_string(),
              }              
            ],
            emissions: vec![
            ],
            resources: vec![],
          },
          Process {
            name: "Drinking water from a bottle".to_string(),
            products: vec![OutputItem {
                product: Product {
                    name: "Drinking 1 liter from a water bottle".to_string(), //FIXME allow different name from process
                },
                amount: Amount::Literal(1.0),
                unit: "lt".to_string(), // Unit consistency check is not implemented yet
            }],
            inputs: vec![
              InputItem {
                product: ProductRef::Product("water bottle production|water bottle".to_string()),
                amount: Amount::Literal(0.005),
                unit: "unit".to_string(),
              },
            ],
            emissions: vec![
              // Remove dummy emission
            ],
            resources: vec![],
          },          
          ],
    };
    let system = model.compile()?;
    Ok(system)

}

fn load_ecoinvent_lca_system() -> Result<lca_rs::LcaSystem, Box<dyn Error>> {
    // Read matrices without regularization
    let a_matrix = read_sparse_matrix("universal_matrix_export/A_public.csv", None)?;
    println!("Matrix A: {:?}", a_matrix.dims());
    let col_ids = read_dim_ids("universal_matrix_export/ie_index.csv", &[0, 1, 2])?;
    let row_ids = read_dim_ids("universal_matrix_export/ie_index.csv", &[0, 1, 2])?;

    let a_matrix = LcaMatrix::new(a_matrix, col_ids, row_ids)?;

    let b_matrix = read_sparse_matrix("universal_matrix_export/B_public.csv", None)?;
    println!("Matrix B: {:?}", b_matrix.dims());
    let col_ids = read_dim_ids("universal_matrix_export/ie_index.csv", &[0, 1, 2])?;
    let row_ids = read_dim_ids("universal_matrix_export/ee_index.csv", &[0, 1, 2])?;
    let b_matrix = LcaMatrix::new(b_matrix, col_ids, row_ids)?;

    let c_matrix = read_sparse_matrix("universal_matrix_export/C.csv", None)?;
    println!("Matrix C: {:?}", c_matrix.dims());
    let col_ids = read_dim_ids("universal_matrix_export/ee_index.csv", &[0, 1, 2])?;
    let row_ids = read_dim_ids("universal_matrix_export/LCIA_index.csv", &[0, 1, 2])?;
    let c_matrix = LcaMatrix::new(c_matrix, col_ids, row_ids)?;

    let lca_system = lca_rs::LcaSystem::new(
        "ecoinvent-3.11".to_string(),
        a_matrix,
        b_matrix,
        c_matrix,
        None,
        vec![],
        vec![],
        vec![],
    )?;
    Ok(lca_system)
}

fn read_dim_ids(filename: &str, concat_columns: &[usize]) -> Result<Vec<String>, Box<dyn Error>> {
    let mut rdr = csv::ReaderBuilder::new()
        .delimiter(b';')
        .has_headers(true)
        .from_path(filename)?;

    let mut dim_ids = Vec::new();

    for result in rdr.records() {
        let record = result?;
        let mut concat_value = vec![];
        for col_index in concat_columns {
            let col_value: String = record[*col_index].parse()?;
            concat_value.push(col_value);
        }
        dim_ids.push(concat_value.join("|"));
    }
    Ok(dim_ids)
}

fn read_sparse_matrix(
    filename: &str,
    epsilon: Option<f64>, // Keep signature for now, but ignore epsilon
) -> Result<SparseMatrix, Box<dyn Error>> {
    let mut rdr = csv::ReaderBuilder::new()
        .delimiter(b';')
        .has_headers(true)
        .from_path(filename)?;
    let mut triplets = Vec::new();
    let mut max_row: usize = 0;
    let mut max_col: usize = 0;
    // Read the CSV file and parse the triplets
    // into a vector of (row_index, col_index, value) tuples
    for result in rdr.records() {
        let record = result?;
        let row_index: usize = record[0].parse()?;
        let col_index: usize = record[1].parse()?;
        let eps = epsilon.unwrap_or(0.0);
        let value: f64 = record[2].parse()?;
        if row_index == col_index {
            // Add epsilon to diagonal elements
            triplets.push(Triplete::new(row_index, col_index, value + eps));
            continue;
        } else {
            triplets.push(Triplete::new(row_index, col_index, value));
        }
        if row_index > max_row {
            max_row = row_index;
        }
        if col_index > max_col {
            max_col = col_index;
        }
    }

    // Determine matrix dimensions
    let num_rows = max_row + 1;
    let num_cols = max_col + 1;

    // Regularization logic removed

    // Create a sparse matrix from the triplets.
    // `from_coo` should handle summing values for duplicate (row, col) entries,
    // effectively adding epsilon to existing diagonal elements or creating them if absent.
    let m = SparseMatrix::from_triplets(num_rows, num_cols, triplets)?;
    Ok(m)
}

fn read_c_matrix(
    filename: &str,
    keep_rows: &[usize],
    num_cols: usize,
) -> Result<SparseMatrix, Box<dyn Error>> {
    let mut rdr = csv::ReaderBuilder::new()
        .delimiter(b';')
        .has_headers(true)
        .from_path(filename)?;
    let mut triplets = Vec::new();
    let mut max_row: usize = 0;
    let mut max_col: usize = 0;
    // Read the CSV file and parse the triplets
    // into a vector of (row_index, col_index, value) tuples
    for result in rdr.records() {
        let record = result?;

        let row_index_in_matrix: usize = record[0].parse()?;
        let col_index: usize = record[1].parse()?;
        if !keep_rows.contains(&row_index_in_matrix) {
            continue; // Skip rows not in keep_rows
        }
        let row_index = keep_rows
            .iter()
            .position(|&x| x == row_index_in_matrix)
            .unwrap();
        let value: f64 = record[2].parse()?;
        triplets.push(Triplete::new(row_index, col_index, value));

        if row_index > max_row {
            max_row = row_index;
        }
        if col_index > max_col {
            max_col = col_index;
        }
    }

    // Determine matrix dimensions
    let num_rows = max_row + 1;

    // Regularization logic removed

    // Create a sparse matrix from the triplets.
    // `from_coo` should handle summing values for duplicate (row, col) entries,
    // effectively adding epsilon to existing diagonal elements or creating them if absent.
    let m = SparseMatrix::from_triplets(num_rows, num_cols, triplets)?;
    Ok(m)
}
