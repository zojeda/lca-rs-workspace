use derive_more::{Display, From};

pub type Result<T> = core::result::Result<T, Error>;

#[derive(Debug, Display, From)]
#[display("{self:?}")]
pub enum Error {
  // -- Externals
  #[from]
  Io(std::io::Error), // as example

  #[from]
  ParseError(quick_xml::de::DeError),
}

// region:    --- Error Boilerplate

impl std::error::Error for Error {}

// endregion: --- Error Boilerplate