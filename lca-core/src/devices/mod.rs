use std::fmt::Debug;

pub trait Device: Debug {}

#[derive(Debug, Clone, Default)]
pub struct CpuDevice {}
impl Device for CpuDevice {}


pub mod gpu;
pub use gpu::{GpuDevice, TransferStats};
// Removed the placeholder traits module
// pub mod traits {
//     use super::Device;
//     pub trait ComputeDevice: Device {}
// }
