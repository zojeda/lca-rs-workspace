use bytemuck::{Pod, Zeroable};
use std::fmt::Debug;

/// Generic trait representing a matrix.
pub trait Matrix: Debug {
    type Value: Copy + Debug + Default + Pod + Zeroable;
    fn dims(&self) -> (usize, usize);
    fn rows(&self) -> usize { self.dims().0 }
    fn cols(&self) -> usize { self.dims().1 }
    fn is_square(&self) -> bool { let (r,c) = self.dims(); r == c }
}

/// Generic trait representing a vector.
pub trait Vector: Debug {
    type Value: Copy + Debug + Default + Pod + Zeroable;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool { self.len() == 0 }
}
