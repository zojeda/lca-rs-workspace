// Consolidated module that re-exports the GPU device and keeps the trait/types aligned
use std::fmt::Debug;

pub trait Device: Debug {}

#[derive(Debug, Clone, Default)]
pub struct CpuDevice {}
impl Device for CpuDevice {}

pub mod gpu_device;
pub use gpu_device::{GpuDevice, TransferStats};
