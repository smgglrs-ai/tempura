pub mod hub;
pub mod inspect;
pub mod store;

#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("io error: {0}")]
    Io(String),
    #[error("download error: {0}")]
    Download(String),
    #[error("format error: {0}")]
    Format(String),
}
