use std::collections::HashMap;

use crate::{SparseMatrix, error::Result, math::sparse_matrix::Triplete};

#[derive(Debug, Clone)]
pub struct LcaMatrix {
    pub matrix: SparseMatrix,
    pub col_ids: Vec<String>,
    pub row_ids: Vec<String>,
}

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
        // Build a stable mapping from row name to new row index preserving the keep_rows order
        let mut index_map: HashMap<&str, usize> = HashMap::with_capacity(keep_rows.len());
        let mut new_row_ids: Vec<String> = Vec::with_capacity(keep_rows.len());
        for name in keep_rows.iter() {
            // Only include rows that actually exist in this matrix and avoid duplicates
            if self.row_ids.contains(name) && !index_map.contains_key(name.as_str()) {
                let next_index = new_row_ids.len();
                index_map.insert(name.as_str(), next_index);
                new_row_ids.push(name.clone());
            }
        }

        let mut triplets = Vec::new();
        for triplete in self.matrix.iter() {
            let row_name = &self.row_ids[triplete.row()];
            if let Some(&new_row) = index_map.get(row_name.as_str()) {
                triplets.push(Triplete::new(new_row, triplete.col(), triplete.value()));
            }
        }

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
