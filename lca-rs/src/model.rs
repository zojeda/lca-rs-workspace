use lca_core::{SparseMatrix, sparse_matrix::Triplete};

use std::collections::HashMap;

use crate::error::{LcaModelCompilationError, Result};

pub struct LcaModel {
    pub imported_dbs: Vec<DbImport>,
    pub processes: Vec<Process>,
}

pub struct DbImport {
    pub name: String,
    pub alias: String,
}
pub struct Process {
    pub name: String,
    pub products: Vec<ProductItem>,
}

pub struct ProductItem {
    pub product: Product,
    pub amount: Amount,
    pub unit: String,
}

pub struct InputItem {
    pub product: ProductRef,
    pub amount: Amount,
    pub unit: String,
}

pub enum ProductRef {
    Product(String),
}

pub enum Amount {
    Literal(f32),
}

impl Amount {
    pub fn evaluate(&self) -> Result<f32> {
        match self {
            Amount::Literal(val) => Ok(*val),
        }
    }
}

pub struct Product {
    pub name: String,
}

/// Compiles the current LCA (Life Cycle Assessment) model into a compiled version.
///
/// * check model internally valid, references, etc
/// * evaluate amount expressions
/// * check units
/// * createa a compiled model with matrix representation that can be calculated
impl LcaModel {
    pub fn compile(self) -> core::result::Result<CompiledLcaModel, LcaModelCompilationError> {
        // Build process index for internal products (by process name)
        let n = self.processes.len();
        let mut triplets: Vec<Triplete> = Vec::with_capacity(n * n);
        let process_index: HashMap<&str, usize> = self
            .processes
            .iter()
            .enumerate()
            .map(|(i, p)| (p.name.as_str(), i))
            .collect();

        // Iterate through processes and accumulate intermediate exchanges
        for (i, process) in self.processes.iter().enumerate() {
            for item in &process.products {
                // Get the literal amount value
                let amount_value = item.amount.evaluate().unwrap();

                // If the product is produced by an internal process, record the exchange.
                // External products are ignored.
                if let Some(&j) = process_index.get(item.product.name.as_str()) {
                    // Avoid self-referential exchange if desired.
                    if i != j {
                        let current = triplets
                            .iter()
                            .find_map(|triplete| {
                                if triplete.row() == i && triplete.col() == j {
                                    Some(triplete.value())
                                } else {
                                    None
                                }
                            })
                            .unwrap_or(0.0);
                        triplets.push(Triplete::new(i, j, current + amount_value));
                    }
                } else {
                    // External referenced product; ignore it.
                }
            }
        }
        let matrix = SparseMatrix::from_triplets(n, n, triplets)?;
        Ok(CompiledLcaModel {
            lca_model: self,
            a_matrix: matrix,
        })
    }
}

pub struct CompiledLcaModel {
    pub lca_model: LcaModel,
    pub a_matrix: SparseMatrix,
}
