use lca_core::LcaCoreError;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum LcaModelCompilationError {
    #[error("Lca Core error: {0}")]
    LcaCoreError(#[from] LcaCoreError),

    #[error("Duplicated Product Name {0}")]
    DuplicatedProductDeclaration(String),
}
