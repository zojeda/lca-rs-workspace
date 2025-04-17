use thiserror::Error;

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

#[derive(Error, Debug)]
pub enum LcaCoreError {
    #[error("WGPU initialization failed: {0}")]
    WgpuInitError(String),

    #[error("WGPU error: {0}")]
    WgpuError(String), // Placeholder for more specific wgpu errors later

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Shader compilation error: {0}")]
    ShaderError(String),

    #[error("Invalid matrix dimensions: {0}")]
    InvalidDimensions(String),

    #[error("Matrix is singular")]
    SingularMatrix,

    #[error("Algorithm did not converge")]
    NonConvergence,

    #[error("Unsupported operation: {0}")]
    UnsupportedOperation(String),

    #[error("Matrix is not symmetric")]
    NonSymmetricMatrix,

    #[error("Matrix has non-positive diagonal elements")]
    NonPositiveDiagonal,

    #[error("Internal error: {0}")]
    Internal(String),

    #[error("BiCGSTAB breakdown at iteration {iteration}: {value_name} ({value}) is near zero.")]
    BiCGSTABBreakdown {
        iteration: usize,
        value_name: String, // e.g., "rho", "omega"
        value: f32,
    },
    // Add more specific errors as needed
}

// Optional: Implement conversion from wgpu specific errors if needed later
// impl From<wgpu::RequestDeviceError> for LsolverError { ... }
#[cfg(feature = "wasm")]
impl From<LcaCoreError> for wasm_bindgen::JsValue {
    fn from(err: LcaCoreError) -> Self {
        Self::from_str(&err.to_string())
    }
}
