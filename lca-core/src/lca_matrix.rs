use std::collections::HashSet;


#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

use crate::{error::Result, sparse_matrix::Triplete, SparseMatrix};

#[cfg_attr(feature = "wasm", wasm_bindgen)]
#[derive(Debug, Clone)]
pub struct LcaMatrix {
    pub matrix: SparseMatrix,
    pub col_ids: Vec<String>,
    pub row_ids: Vec<String>,
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
impl LcaMatrix {
    pub fn new(matrix: SparseMatrix, col_ids: Vec<String>, row_ids: Vec<String>) -> Result<Self> {
        if matrix.cols() != col_ids.len() {
            return Err(crate::error::LcaCoreError::DimensionError(format!(
                "Matrix columns ({}) must match column IDs length ({})",
                matrix.cols(),
                col_ids.len()
            ))
            .into());
        }
        if matrix.rows() != row_ids.len() {
            return Err(crate::error::LcaCoreError::DimensionError(format!(
                "Matrix rows ({}) must match row IDs length ({})",
                matrix.rows(),
                row_ids.len()
            ))
            .into());
        }
        Ok(Self {
            matrix,
            col_ids: col_ids.to_vec(),
            row_ids: row_ids.to_vec(),
        })
    }
    #[cfg(not(target_arch = "wasm32"))]
    pub fn matrix(&self) -> &SparseMatrix {
        &self.matrix
    }

    pub fn filter_rows(&self, keep_rows: &[String]) -> Result<Self> {
        let mut new_row_ids = HashSet::new();
        let mut triplets = Vec::new();
        // Iterate over the rows and keep only the specified ones, creating a triplets list for the new matrix
        for triplete in self.matrix.iter() {
            let row_id = &self.row_ids[triplete.row()];
            if keep_rows.contains(row_id) {
                new_row_ids.insert(row_id.clone());
                let row_id = new_row_ids.iter().position(|id| id == row_id).unwrap();
                triplets.push(Triplete::new(row_id, triplete.col(), triplete.value()));
            };
        }
        let new_row_ids: Vec<String> = new_row_ids.into_iter().collect();
        let new_matrix =
            SparseMatrix::from_triplets(new_row_ids.len(), self.matrix.cols(), triplets)?;

        Ok(Self {
            matrix: new_matrix,
            col_ids: self.col_ids.clone(),
            row_ids: new_row_ids,
        })
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn col_ids(&self) -> &[String] {
        &self.col_ids
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn row_ids(&self) -> &[String] {
        &self.row_ids
    }
    #[cfg(not(target_arch = "wasm32"))]
    pub fn col_id(&self, index: usize) -> Option<&String> {
        self.col_ids.get(index)
    }
    #[cfg(not(target_arch = "wasm32"))]
    pub fn row_id(&self, index: usize) -> Option<&String> {
        self.row_ids.get(index)
    }
    #[cfg(not(target_arch = "wasm32"))]
    pub fn col_idx_by_name(&self, name: &str) -> Option<usize> {
        self.col_ids.iter().position(|id| id == name)
    }
    #[cfg(not(target_arch = "wasm32"))]
    pub fn row_idx_by_name(&self, name: &str) -> Option<usize> {
        self.row_ids.iter().position(|id| id == name)
    }

}
