use lca_core::{LcaCoreError, Matrix}; // Import from lca_core

/// Represents a dense matrix stored in row-major order on the CPU.
#[derive(Debug, Clone)] // Added Clone
pub struct DenseMatrix<T: Copy + Send + Sync + std::fmt::Debug + Default + bytemuck::Pod> {
    rows: usize,
    cols: usize,
    data: Vec<T>, // Data stored row-major: data[row * cols + col]
}

impl<T: Copy + Send + Sync + std::fmt::Debug + Default + bytemuck::Pod> DenseMatrix<T> {
    /// Creates a new DenseMatrix from raw data, dimensions, assuming row-major order.
    pub fn new(rows: usize, cols: usize, data: Vec<T>) -> Result<Self, LcaCoreError> {
        if data.len() != rows * cols {
            return Err(LcaCoreError::InvalidDimensions(format!(
                "Data length ({}) does not match dimensions ({}x{})",
                data.len(),
                rows,
                cols
            )));
        }
        Ok(Self { rows, cols, data })
    }

    /// Creates a new DenseMatrix filled with zeros.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![T::default(); rows * cols],
        }
    }

    /// Returns a slice view of the underlying data vector.
    pub fn data(&self) -> &[T] {
        &self.data
    }

    /// Returns a mutable slice view of the underlying data vector.
    pub fn data_mut(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Gets the element at the specified row and column (immutable).
    /// Returns None if indices are out of bounds.
    pub fn get(&self, row: usize, col: usize) -> Option<&T> {
        if row < self.rows && col < self.cols {
            self.data.get(row * self.cols + col)
        } else {
            None
        }
    }

    /// Gets the element at the specified row and column (mutable).
    /// Returns None if indices are out of bounds.
    pub fn get_mut(&mut self, row: usize, col: usize) -> Option<&mut T> {
        if row < self.rows && col < self.cols {
            self.data.get_mut(row * self.cols + col)
        } else {
            None
        }
    }
}

// Implement the generic Matrix trait
impl<T: Copy + Send + Sync + std::fmt::Debug + Default + bytemuck::Pod> Matrix for DenseMatrix<T> {
    type Value = T;

    fn dims(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    // rows(), cols(), is_square() are provided by default impls in the trait
}
