// Include the legacy implementation file from the previous module layout.
#[path = "../../device/gpu_device.rs"]
mod gpu_device;
pub use gpu_device::{GpuDevice, TransferStats};
