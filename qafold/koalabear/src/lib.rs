//! Public-metadata ParComm over KoalaBear/KoalaBear^4, with complete QAFold openings.
//! No hiding or zero-knowledge claim is made by this implementation.
pub mod eo;
pub mod field;
pub mod layout;
pub mod protocol;
pub mod tbv;
pub mod transcript;
pub use protocol::*;
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;
pub(crate) fn check(ok: bool, message: &'static str) -> Result<()> {
    if ok { Ok(()) } else { Err(message.into()) }
}
