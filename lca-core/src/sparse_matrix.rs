use crate::context::GpuContext; // Use context from this crate
use crate::error::LcaCoreError; // Use error from this crate
use crate::traits::Matrix; // Use Matrix trait from this crate (will be created)
use std::sync::Arc; // Need bytemuck for casting in SparseMatrixGpu
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;
/// Represents a sparse matrix in Compressed Sparse Row (CSR) format on the CPU.
#[cfg_attr(feature = "wasm", wasm_bindgen)]
#[derive(Debug, Clone, PartialEq)]
pub struct SparseMatrix {
    /// Number of rows.
    rows: usize,
    /// Number of columns.
    cols: usize,
    /// Vector containing the non-zero values of the matrix.
    pub(crate) values: Vec<f32>, // Made pub(crate) for internal access
    /// Vector containing the column indices corresponding to the values.
    pub(crate) col_indices: Vec<usize>,
    /// Vector containing the pointers to the start of each row in `values` and `col_indices`.
    /// The length of this vector is `rows + 1`. `row_ptr[i]` gives the index in `values`
    /// where row `i` starts, and `row_ptr[rows]` gives the total number of non-zero elements (nnz).
    pub(crate) row_ptr: Vec<usize>,
}

#[cfg_attr(feature = "wasm", wasm_bindgen)]
impl SparseMatrix {
    /// Creates a new empty SparseMatrix with given dimensions.
    #[cfg_attr(feature = "wasm", wasm_bindgen(constructor))]
    pub fn new(rows: usize, cols: usize) -> Self {
        SparseMatrix {
            rows,
            cols,
            values: Vec::new(),
            col_indices: Vec::new(),
            row_ptr: vec![0; rows + 1], // Initialize row_ptr with zeros
        }
    }
    pub fn from_triplets(
        rows: usize,
        cols: usize,
        triplets: Vec<Triplete>,
    ) -> Result<Self, LcaCoreError> {
        // Sort the triplets by row and then by column
        let mut coords = triplets.to_vec();
        coords.sort_unstable_by_key(|&Triplete { row, col, .. }| (row, col));

        // Count non-zeros per row
        let mut row_ptr = vec![0usize; rows + 1];
        for &Triplete { row, .. } in &coords {
            if row >= rows {
                return Err(LcaCoreError::InvalidDimensions(
                    "Row index out of bounds".to_string(),
                ));
            }
            row_ptr[row + 1] += 1;
        }
        // Convert counts to cumulative row pointers
        for i in 1..=rows {
            row_ptr[i] += row_ptr[i - 1];
        }

        let nnz = coords.len();
        let mut values = vec![f32::default(); nnz];
        let mut col_indices = vec![0usize; nnz];
        // Temporary copy of row_ptr to keep track of positions while filling
        let mut next = row_ptr.clone();
        for &Triplete { row, col, value } in &coords {
            if col >= cols {
                return Err(LcaCoreError::InvalidDimensions(
                    "Column index out of bounds".to_string(),
                ));
            }
            let pos = next[row];
            values[pos] = value;
            col_indices[pos] = col;
            next[row] += 1;
        }

        Ok(SparseMatrix {
            rows,
            cols,
            values,
            col_indices,
            row_ptr,
        })
    }

    /// Creates a SparseMatrix from CSR components.
    /// Performs basic validation.
    pub fn from_csr(
        rows: usize,
        cols: usize,
        values: Vec<f32>,
        col_indices: Vec<usize>,
        row_ptr: Vec<usize>,
    ) -> Result<Self, LcaCoreError> {
        // Basic validation
        if row_ptr.len() != rows + 1 {
            return Err(LcaCoreError::InvalidDimensions(
                "row_ptr length must be rows + 1".to_string(),
            ));
        }
        if values.len() != col_indices.len() {
            return Err(LcaCoreError::InvalidDimensions(
                // Using InvalidDimensions as the closest fit
                "values and col_indices must have the same length".to_string(),
            ));
        }
        if let Some(&last_ptr) = row_ptr.last() {
            if last_ptr != values.len() {
                return Err(LcaCoreError::InvalidDimensions(
                    // Using InvalidDimensions
                    "Last element of row_ptr must equal the number of non-zero values".to_string(),
                ));
            }
        }
        if col_indices.iter().any(|&c| c >= cols) {
            return Err(LcaCoreError::InvalidDimensions(
                // Using InvalidDimensions
                "Column index out of bounds".to_string(),
            ));
        }
        // Could add more validation (e.g., row_ptr is non-decreasing)

        Ok(SparseMatrix {
            rows,
            cols,
            values,
            col_indices,
            row_ptr,
        })
    }

    /// Returns the dimensions of the matrix (rows, cols).
    #[cfg(not(feature = "wasm"))]
    pub fn dims(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn is_square(&self) -> bool {
        self.rows == self.cols
    }

    /// Returns the number of non-zero elements.
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Gets the value at a specific row and column.
    /// This is inefficient for sparse matrices, primarily for testing/debugging.
    pub fn get(&self, row: usize, col: usize) -> Option<f32> {
        if row >= self.rows || col >= self.cols {
            return None; // Out of bounds
        }

        let row_start = self.row_ptr[row];
        let row_end = self.row_ptr[row + 1];

        // Search within the specific row's non-zero elements
        for i in row_start..row_end {
            if self.col_indices[i] == col {
                return Some(self.values[i]);
            }
            // Optimization: Since col_indices within a row are usually sorted
            // if self.col_indices[i] > col {
            //     break;
            // }
        }

        // If not found among non-zeros, it's implicitly zero
        // However, returning Option<T> is more idiomatic than returning T::default()
        None
    }

    #[cfg(not(feature = "wasm"))]
    /// Returns a slice containing the non-zero values.
    pub fn values(&self) -> &[f32] {
        &self.values
    }

    #[cfg(not(feature = "wasm"))]
    /// Returns a mutable slice containing the non-zero values.
    pub fn values_mut(&mut self) -> &mut [f32] {
        &mut self.values
    }

    #[cfg(not(feature = "wasm"))]
    /// Returns a slice containing the column indices.
    pub fn col_indices(&self) -> &[usize] {
        &self.col_indices
    }

    #[cfg(not(feature = "wasm"))]
    /// Returns a slice containing the row pointers.
    pub fn row_ptr(&self) -> &[usize] {
        &self.row_ptr
    }

    /// Creates a SparseMatrix from a dense 2D vector representation.
    /// Requires `T` to implement `PartialEq` and `Default`.
    #[cfg(not(feature = "wasm"))]
    pub fn from_dense(dense: &[Vec<f32>]) -> Self {
        let rows = dense.len();
        if rows == 0 {
            return SparseMatrix::new(0, 0);
        }
        // Assume rectangular matrix, get cols from first row if it exists
        let cols = dense.get(0).map_or(0, |row| row.len());
        let mut values = Vec::new();
        let mut col_indices = Vec::new();
        let mut row_ptr = vec![0; rows + 1];
        let mut nnz = 0;

        for (r, row_vec) in dense.iter().enumerate() {
            // Optional: Add assertion for rectangular matrix if needed
            // assert_eq!(row_vec.len(), cols, "All rows must have the same length");
            if row_vec.len() != cols {
                // Handle non-rectangular input? For now, assume it's rectangular based on first row.
                // Or return an error:
                // return Err(LsolverError::InvalidDimensions("Input dense matrix must be rectangular".to_string()));
                // For simplicity matching the original helper, we proceed assuming rectangular based on row 0.
            }
            for (c, &val) in row_vec.iter().enumerate() {
                if val != f32::default() {
                    // Compare against default value
                    values.push(val);
                    col_indices.push(c);
                    nnz += 1;
                }
            }
            row_ptr[r + 1] = nnz;
        }
        // Use from_csr which handles validation, although in this case,
        // the construction guarantees validity if input is rectangular.
        // Unwrap is safe here assuming correct construction logic.
        SparseMatrix::from_csr(rows, cols, values, col_indices, row_ptr).unwrap()
    }
}

#[cfg_attr(feature = "wasm", wasm_bindgen)]
#[derive(Debug, Clone, PartialEq)]
pub struct Triplete {
    row: usize,
    col: usize,
    value: f32,
}
impl Triplete {
    pub fn new(row: usize, col: usize, value: f32) -> Self {
        Triplete { row, col, value }
    }

    pub fn row(&self) -> usize {
        self.row
    }

    pub fn col(&self) -> usize {
        self.col
    }

    pub fn value(&self) -> f32 {
        self.value
    }
}

pub struct SparseMatrixIter {
    matrix: SparseMatrix,
    row: usize,
    col: usize,
}

impl Iterator for SparseMatrixIter {
    type Item = Triplete;

    fn next(&mut self) -> Option<Self::Item> {
        if self.row >= self.matrix.rows {
            return None;
        }

        let row_start = self.matrix.row_ptr[self.row];
        let row_end = self.matrix.row_ptr[self.row + 1];

        if self.col >= row_end - row_start {
            self.row += 1;
            self.col = 0;
            return self.next();
        }

        let index = row_start + self.col;
        let col = self.matrix.col_indices[index];
        let value = self.matrix.values[index];

        self.col += 1;

        Some(Triplete {
            row: self.row,
            col,
            value,
        })
    }
}

impl SparseMatrix {
    pub fn iter(&self) -> SparseMatrixIter {
        SparseMatrixIter {
            matrix: self.clone(),
            row: 0,
            col: 0,
        }
    }
}

// Implement the generic Matrix trait for the CPU version
impl Matrix for SparseMatrix {
    type Value = f32;

    fn dims(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    // rows(), cols(), is_square() are provided by default impls in the trait
}

#[cfg(test)]
mod tests {
    // Use types from this crate for tests
    use crate::{LcaCoreError, SparseMatrix};

    #[test]
    fn test_sparse_matrix_new() {
        let matrix: SparseMatrix = SparseMatrix::new(3, 4);
        assert_eq!(matrix.dims(), (3, 4));
        assert_eq!(matrix.nnz(), 0);
        assert_eq!(matrix.row_ptr, vec![0, 0, 0, 0]); // rows + 1 entries
        assert!(matrix.values.is_empty());
        assert!(matrix.col_indices.is_empty());
    }

    #[test]
    fn test_sparse_matrix_from_csr_valid() {
        let rows = 3;
        let cols = 4;
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let col_indices = vec![0, 2, 1, 3, 2];
        let row_ptr = vec![0, 2, 4, 5]; // nnz per row: 2, 2, 1

        let matrix = SparseMatrix::from_csr(
            rows,
            cols,
            values.clone(),
            col_indices.clone(),
            row_ptr.clone(),
        );

        assert!(matrix.is_ok());
        let matrix = matrix.unwrap();

        assert_eq!(matrix.dims(), (rows, cols));
        assert_eq!(matrix.nnz(), 5);
        assert_eq!(matrix.values, values);
        assert_eq!(matrix.col_indices, col_indices);
        assert_eq!(matrix.row_ptr, row_ptr);
    }

    #[test]
    fn test_sparse_matrix_from_csr_invalid_row_ptr_len() {
        let matrix = SparseMatrix::from_csr(3, 4, vec![1.0], vec![0], vec![0, 1]); // row_ptr too short
        assert!(matrix.is_err());
        match matrix.err().unwrap() {
            LcaCoreError::InvalidDimensions(msg) => assert!(msg.contains("row_ptr length")),
            _ => panic!("Expected InvalidDimensions error"),
        }
    }

    #[test]
    fn test_sparse_matrix_from_csr_invalid_last_row_ptr() {
        let matrix = SparseMatrix::from_csr(3, 4, vec![1.0, 2.0], vec![0, 1], vec![0, 1, 1, 1]); // last ptr != nnz
        assert!(matrix.is_err());
        match matrix.err().unwrap() {
            LcaCoreError::InvalidDimensions(msg) => {
                assert!(msg.contains("Last element of row_ptr"))
            }
            _ => panic!("Expected InvalidDimensions error"),
        }
    }

    #[test]
    fn test_sparse_matrix_from_csr_invalid_col_index() {
        let matrix = SparseMatrix::from_csr(2, 3, vec![1.0, 2.0], vec![0, 3], vec![0, 1, 2]); // col index 3 >= cols 3
        assert!(matrix.is_err());
        match matrix.err().unwrap() {
            LcaCoreError::InvalidDimensions(msg) => {
                assert!(msg.contains("Column index out of bounds"))
            }
            _ => panic!("Expected InvalidDimensions error"),
        }
    }

    #[test]
    fn test_sparse_matrix_from_csr_mismatch_values_indices() {
        let matrix = SparseMatrix::from_csr(2, 3, vec![1.0, 2.0], vec![0], vec![0, 1, 1]); // values len != indices len
        assert!(matrix.is_err());
        match matrix.err().unwrap() {
            LcaCoreError::InvalidDimensions(msg) => assert!(msg.contains("values and col_indices")),
            _ => panic!("Expected InvalidDimensions error"),
        }
    }

    #[test]
    fn test_sparse_matrix_get() {
        // Example matrix:
        // [ 1.0, 0.0, 2.0, 0.0 ]
        // [ 0.0, 3.0, 0.0, 4.0 ]
        // [ 0.0, 0.0, 5.0, 0.0 ]
        let rows = 3;
        let cols = 4;
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let col_indices = vec![0, 2, 1, 3, 2];
        let row_ptr = vec![0, 2, 4, 5];
        let matrix = SparseMatrix::from_csr(rows, cols, values, col_indices, row_ptr).unwrap();

        // Test existing values
        assert_eq!(matrix.get(0, 0), Some(1.0));
        assert_eq!(matrix.get(0, 2), Some(2.0));
        assert_eq!(matrix.get(1, 1), Some(3.0));
        assert_eq!(matrix.get(1, 3), Some(4.0));
        assert_eq!(matrix.get(2, 2), Some(5.0));

        // Test zero values
        assert_eq!(matrix.get(0, 1), None);
        assert_eq!(matrix.get(0, 3), None);
        assert_eq!(matrix.get(1, 0), None);
        assert_eq!(matrix.get(1, 2), None);
        assert_eq!(matrix.get(2, 0), None);
        assert_eq!(matrix.get(2, 1), None);
        assert_eq!(matrix.get(2, 3), None);

        // Test out of bounds
        assert_eq!(matrix.get(3, 0), None);
        assert_eq!(matrix.get(0, 4), None);
        assert_eq!(matrix.get(3, 4), None);
    }
}

/// Represents a sparse matrix in CSR format stored on the GPU.
#[cfg_attr(feature = "wasm", wasm_bindgen)]
#[derive(Debug)] // Removed Clone as buffers are not easily cloneable
pub struct SparseMatrixGpu {
    /// Number of rows.
    rows: usize,
    /// Number of columns.
    cols: usize,
    /// GPU buffer containing the non-zero values (f32).
    values_buffer: wgpu::Buffer,
    /// GPU buffer containing the column indices (u32).
    col_indices_buffer: wgpu::Buffer,
    /// GPU buffer containing the row pointers (u32).
    row_pointers_buffer: wgpu::Buffer,
    /// Number of non-zero elements.
    nnz: usize,
    /// Reference to the GPU context (internal).
    pub(crate) context: Arc<GpuContext>,
}

// Implement the Matrix trait for the GPU version
impl Matrix for SparseMatrixGpu {
    type Value = f32; // GPU version likely specialized to f32 for now

    fn dims(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    // rows(), cols(), is_square() are provided by default impls in the trait
}

impl SparseMatrixGpu {
    /// Internal constructor used by GpuDevice.
    pub(crate) fn new_internal(
        rows: usize,
        cols: usize,
        nnz: usize,
        values_buffer: wgpu::Buffer,
        col_indices_buffer: wgpu::Buffer,
        row_pointers_buffer: wgpu::Buffer,
        context: Arc<GpuContext>,
    ) -> Self {
        Self {
            rows,
            cols,
            nnz,
            values_buffer,
            col_indices_buffer,
            row_pointers_buffer,
            context,
        }
    }

    // Public `from_sparse_matrix` is removed. Creation happens via `GpuDevice::create_sparse_matrix`.
    /*
    // Old public constructor - removed
    pub fn from_sparse_matrix(...) { ... }
    */

    // Accessors
    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn cols(&self) -> usize {
        self.cols
    }
    pub fn nnz(&self) -> usize {
        self.nnz
    }
    // Provide accessors for buffers if needed internally or by ops module
    pub(crate) fn values_buffer(&self) -> &wgpu::Buffer {
        &self.values_buffer
    }
    pub(crate) fn col_indices_buffer(&self) -> &wgpu::Buffer {
        &self.col_indices_buffer
    }
    pub(crate) fn row_pointers_buffer(&self) -> &wgpu::Buffer {
        &self.row_pointers_buffer
    }
}
