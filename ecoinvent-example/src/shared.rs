use lca_rs::core::sparse_matrix::Triplete;
use lca_rs::core::{LcaMatrix, LcaSystem, SparseMatrix};

use lca_rs::model::{
    Amount, DbImport, ExternalRef, InputItem, LcaModel, OutputItem, Process, Product, ProductRef,
};
use std::error::Error;

pub fn create_water_botle_lca_system() -> Result<LcaSystem, Box<dyn Error>> {
    let model = LcaModel {
        case_name: "water bottle production".to_string(),
        case_description: "Water bottle LCA Description".to_string(),
        database_name: "Water bottle LCA".to_string(),
        imported_dbs: vec![DbImport { name: "ecoinvent-3.11".to_string(), alias: "ei".to_string() }],
        substances: vec![],
        evaluation_demands: vec![],
        evaluation_impacts: vec![],
        processes: vec![
            Process {
                name: "water bottle production".to_string(),
                products: vec![OutputItem { product: Product { name: "water bottle".to_string() }, amount: Amount::Literal(1.0), unit: "unit".to_string() }],
                inputs: vec![
                    InputItem { product: ProductRef::External(ExternalRef { alias: "ei".to_string(), name: "treatment of aluminium scrap, post-consumer, prepared for recycling, at remelter|RoW|aluminium, wrought alloy".to_string() }), amount: Amount::Literal(0.33), unit: "kg".to_string() },
                    InputItem { product: ProductRef::External(ExternalRef { alias: "ei".to_string(), name: "impact extrusion of aluminium, deformation stroke|RoW|impact extrusion of aluminium, deformation stroke".to_string() }), amount: Amount::Literal(0.33), unit: "kg".to_string() },
                ],
                emissions: vec![],
                resources: vec![],
            },
            Process {
                name: "Drinking water from a bottle".to_string(),
                products: vec![OutputItem { product: Product { name: "Drinking 1 liter from a water bottle".to_string() }, amount: Amount::Literal(1.0), unit: "lt".to_string() }],
                inputs: vec![InputItem { product: ProductRef::Product("water bottle production|water bottle".to_string()), amount: Amount::Literal(0.005), unit: "unit".to_string() }],
                emissions: vec![],
                resources: vec![],
            },
        ],
    };
    let system = model.compile()?;
    Ok(system)
}

pub fn load_ecoinvent_lca_system() -> Result<LcaSystem, Box<dyn Error>> {
    let a_matrix = read_sparse_matrix("universal_matrix_export/A_public.csv", None)?;
    let col_ids = read_dim_ids("universal_matrix_export/ie_index.csv", &[0, 1, 2])?;
    let row_ids = read_dim_ids("universal_matrix_export/ie_index.csv", &[0, 1, 2])?;
    let a_matrix = LcaMatrix::new(a_matrix, col_ids, row_ids)?;

    let b_matrix = read_sparse_matrix("universal_matrix_export/B_public.csv", None)?;
    let col_ids = read_dim_ids("universal_matrix_export/ie_index.csv", &[0, 1, 2])?;
    let row_ids = read_dim_ids("universal_matrix_export/ee_index.csv", &[0, 1, 2])?;
    let b_matrix = LcaMatrix::new(b_matrix, col_ids, row_ids)?;

    let c_matrix = read_sparse_matrix("universal_matrix_export/C.csv", None)?;
    let col_ids = read_dim_ids("universal_matrix_export/ee_index.csv", &[0, 1, 2])?;
    let row_ids = read_dim_ids("universal_matrix_export/LCIA_index.csv", &[0, 1, 2])?;
    let c_matrix = LcaMatrix::new(c_matrix, col_ids, row_ids)?;

    let lca_system = LcaSystem::new(
        "ecoinvent-3.11".to_string(),
        a_matrix,
        b_matrix,
        c_matrix,
        None,
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
    epsilon: Option<f64>,
) -> Result<SparseMatrix, Box<dyn Error>> {
    let mut rdr = csv::ReaderBuilder::new()
        .delimiter(b';')
        .has_headers(true)
        .from_path(filename)?;
    let mut triplets = Vec::new();
    let mut max_row: usize = 0;
    let mut max_col: usize = 0;
    for result in rdr.records() {
        let record = result?;
        let row_index: usize = record[0].parse()?;
        let col_index: usize = record[1].parse()?;
        let eps = epsilon.unwrap_or(0.0);
        let value: f64 = record[2].parse()?;
        if row_index == col_index {
            triplets.push(Triplete::new(row_index, col_index, value + eps));
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
    let num_rows = max_row + 1;
    let num_cols = max_col + 1;
    let m = SparseMatrix::from_triplets(num_rows, num_cols, triplets)?;
    Ok(m)
}

#[allow(dead_code)]
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
    for result in rdr.records() {
        let record = result?;
        let row_index_in_matrix: usize = record[0].parse()?;
        let col_index: usize = record[1].parse()?;
        if !keep_rows.contains(&row_index_in_matrix) {
            continue;
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
    let num_rows = max_row + 1;
    let m = SparseMatrix::from_triplets(num_rows, num_cols, triplets)?;
    Ok(m)
}
