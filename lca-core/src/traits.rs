use bytemuck::{Pod, Zeroable}; // Need Pod and Zeroable for trait bounds
use std::fmt::Debug;

/// Generic trait representing a matrix.
/// Implementations can be sparse, dense, CPU-based, GPU-based.
/// Needs Send + Sync + Debug.
pub trait Matrix: Debug {
    /// The underlying numeric type of the matrix elements (e.g., f32, f32).
    type Value: Copy + Debug + Default + Pod + Zeroable; // Added Zeroable

    /// Returns the dimensions of the matrix as (rows, columns).
    fn dims(&self) -> (usize, usize);

    /// Returns the number of rows.
    fn rows(&self) -> usize {
        self.dims().0
    }

    /// Returns the number of columns.
    fn cols(&self) -> usize {
        self.dims().1
    }

    /// Checks if the matrix is square.
    fn is_square(&self) -> bool {
        let (rows, cols) = self.dims();
        rows == cols
    }
}

/// Generic trait representing a vector.
/// Implementations can be CPU-based or GPU-based.
/// Needs Send + Sync + Debug.
pub trait Vector: Debug {
    /// The underlying numeric type of the vector elements (e.g., f32, f32).
    type Value: Copy + Debug + Default + Pod + Zeroable; // Added Zeroable

    /// Returns the number of elements in the vector.
    fn len(&self) -> usize;

    /// Checks if the vector is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
