use crate::context::GpuContext;
use crate::error::LcaCoreError;
use crate::math::traits::Matrix;
use std::sync::Arc;

/// Represents a sparse matrix in Compressed Sparse Row (CSR) format on the CPU.
#[derive(Debug, Clone, PartialEq)]
pub struct SparseMatrix {
    rows: usize,
    cols: usize,
    pub(crate) values: Vec<f64>,
    pub(crate) col_indices: Vec<usize>,
    pub(crate) row_ptr: Vec<usize>,
}

impl SparseMatrix {
    pub fn new(rows: usize, cols: usize) -> Self {
        SparseMatrix {
            rows,
            cols,
            values: Vec::new(),
            col_indices: Vec::new(),
            row_ptr: vec![0; rows + 1],
        }
    }
    pub fn from_triplets(
        rows: usize,
        cols: usize,
        triplets: Vec<Triplete>,
    ) -> Result<Self, LcaCoreError> {
        let mut coords = triplets.to_vec();
        coords.sort_unstable_by_key(|&Triplete { row, col, .. }| (row, col));

        let mut row_ptr = vec![0usize; rows + 1];
        for &Triplete { row, .. } in &coords {
            if row >= rows {
                return Err(LcaCoreError::InvalidDimensions(
                    "Row index out of bounds".to_string(),
                ));
            }
            row_ptr[row + 1] += 1;
        }
        for i in 1..=rows {
            row_ptr[i] += row_ptr[i - 1];
        }

        let nnz = coords.len();
        let mut values = vec![f64::default(); nnz];
        let mut col_indices = vec![0usize; nnz];
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

    pub fn from_csr(
        rows: usize,
        cols: usize,
        values: Vec<f64>,
        col_indices: Vec<usize>,
        row_ptr: Vec<usize>,
    ) -> Result<Self, LcaCoreError> {
        if row_ptr.len() != rows + 1 {
            return Err(LcaCoreError::InvalidDimensions(
                "row_ptr length must be rows + 1".to_string(),
            ));
        }
        if values.len() != col_indices.len() {
            return Err(LcaCoreError::InvalidDimensions(
                "values and col_indices must have the same length".to_string(),
            ));
        }
        if let Some(&last_ptr) = row_ptr.last() {
            if last_ptr != values.len() {
                return Err(LcaCoreError::InvalidDimensions(
                    "Last element of row_ptr must equal the number of non-zero values".to_string(),
                ));
            }
        }
        if col_indices.iter().any(|&c| c >= cols) {
            return Err(LcaCoreError::InvalidDimensions(
                "Column index out of bounds".to_string(),
            ));
        }
        Ok(SparseMatrix {
            rows,
            cols,
            values,
            col_indices,
            row_ptr,
        })
    }

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
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    pub fn get(&self, row: usize, col: usize) -> Option<f64> {
        if row >= self.rows || col >= self.cols {
            return None;
        }
        let row_start = self.row_ptr[row];
        let row_end = self.row_ptr[row + 1];
        for i in row_start..row_end {
            if self.col_indices[i] == col {
                return Some(self.values[i]);
            }
        }
        None
    }

    pub fn values(&self) -> &[f64] {
        &self.values
    }
    pub fn values_mut(&mut self) -> &mut [f64] {
        &mut self.values
    }
    pub fn col_indices(&self) -> &[usize] {
        &self.col_indices
    }
    pub fn row_ptr(&self) -> &[usize] {
        &self.row_ptr
    }

    pub fn from_dense(dense: &[Vec<f64>]) -> Self {
        let rows = dense.len();
        if rows == 0 {
            return SparseMatrix::new(0, 0);
        }
        let cols = dense.get(0).map_or(0, |row| row.len());
        let mut values = Vec::new();
        let mut col_indices = Vec::new();
        let mut row_ptr = vec![0; rows + 1];
        let mut nnz = 0;
        for (r, row_vec) in dense.iter().enumerate() {
            for (c, &val) in row_vec.iter().enumerate() {
                if val != f64::default() {
                    values.push(val);
                    col_indices.push(c);
                    nnz += 1;
                }
            }
            row_ptr[r + 1] = nnz;
        }
        SparseMatrix::from_csr(rows, cols, values, col_indices, row_ptr).unwrap()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Triplete {
    row: usize,
    col: usize,
    value: f64,
}
impl Triplete {
    pub fn new(row: usize, col: usize, value: f64) -> Self {
        Triplete { row, col, value }
    }
    pub fn row(&self) -> usize {
        self.row
    }
    pub fn col(&self) -> usize {
        self.col
    }
    pub fn value(&self) -> f64 {
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

impl Matrix for SparseMatrix {
    type Value = f64;
    fn dims(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }
}

/// Represents a sparse matrix in CSR format stored on the GPU.
#[derive(Debug)]
pub struct SparseMatrixGpu {
    rows: usize,
    cols: usize,
    values_buffer: wgpu::Buffer,
    col_indices_buffer: wgpu::Buffer,
    row_pointers_buffer: wgpu::Buffer,
    nnz: usize,
    pub(crate) context: Arc<GpuContext>,
}

impl Matrix for SparseMatrixGpu {
    type Value = f64;
    fn dims(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }
}

impl SparseMatrixGpu {
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
    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn cols(&self) -> usize {
        self.cols
    }
    pub fn nnz(&self) -> usize {
        self.nnz
    }
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
