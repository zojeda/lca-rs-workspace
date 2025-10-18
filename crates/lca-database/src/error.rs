use std::path::PathBuf;

use derive_more::{Display, From};
pub type Result<T> = core::result::Result<T, Error>;

#[derive(Debug, From, Display)]
#[display("{self:?}")]
pub enum Error {
    #[from]
    IoError(std::io::Error),
    #[from]
    SurrealDbError(surrealdb::Error),
    #[from]
    UuidError(uuid::Error),

    #[from]
    EcospoldParserError(ecospold_parser::error::Error),

    NoActivityDatasetFound(PathBuf),

    DatabaseImportError(String),

    FailedToCreateEntity(String),
}

// region:    --- Error Boilerplate

impl std::error::Error for Error {}

// endregion: --- Error Boilerplate
