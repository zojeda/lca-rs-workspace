use lca_core::LcaCoreError;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum LcaModelCompilationError {
    #[error("Lca Core error: {0}")]
    LcaCoreError(#[from] LcaCoreError), // Error from sparse matrix creation

    #[error("Duplicate process name found: {0}")]
    DuplicateProcessName(String),

    #[error("Duplicate substance name found: {0}")]
    DuplicateSubstanceName(String),

    #[error("Process '{0}' must have exactly one product output, found {1}")]
    ProcessProductCardinalityError(String, usize),

    #[error("Process '{0}' has more than one product output, but only one is allowed for now")]
    MultiProductUnsupported(String),

    #[error("Process '{0}' product output '{1}' has non-positive amount: {2}")]
    NonPositiveProductAmount(String, String, f64),

    #[error("Process '{0}' input '{1}' has negative amount: {2}")]
    NegativeInputAmount(String, String, f64), // Second String will be formatted ProductRef

    #[error("Unresolved internal product reference: {0}")]
    UnresolvedInternalProductRef(String),

    #[error("Unresolved external impact reference: {0}")]
    UnresolvedExternalImpactRef(String),

    #[error("Unresolved internal substance reference: {0}")]
    UnresolvedInternalSubstanceRef(String),

    #[error("Error creating system matrix: {0}")]
    UnableToCreateSystemMatrix(String),

    #[error("Error system matrix: {0}")]
    UnableToLcaSystem(String),

    #[error("Evaluation demand name clashes with an existing process or evaluation demand: {0}")]
    EvaluationDemandNameClash(String),

    #[error("Evaluation demand targets a product not found in processes: {0}")]
    EvaluationDemandTargetNotFound(String),
}
