mod compilation_error;

pub use compilation_error::LcaModelCompilationError;
use derive_more::From;

use lca_core::error::LcaCoreError;

pub type Result<T> = core::result::Result<T, LcaError>;

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

#[derive(Debug, From)]
pub enum LcaError {
    // -- Externals
    #[from]
    Io(std::io::Error),

    #[from]
    LcaModelCompilationError(LcaModelCompilationError),

    #[from]
    LcaCoreError(LcaCoreError),

    DimensionError(String),

    JsError(String),

    ConversionError(String),
}

// region:    --- Error Boilerplate

impl core::fmt::Display for LcaError {
    fn fmt(&self, fmt: &mut core::fmt::Formatter) -> core::result::Result<(), core::fmt::Error> {
        write!(fmt, "{self:?}")
    }
}

impl std::error::Error for LcaError {}

// endregion: --- Error Boilerplate

// Convert custom error to JsValue for WASM boundary
#[cfg(feature = "wasm")]
impl From<LcaError> for JsValue {
    fn from(err: LcaError) -> JsValue {
        JsValue::from_str(&err.to_string())
    }
}
