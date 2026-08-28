use halo2_proofs::poly::commitment::CommitmentScheme;
use halo2_proofs::poly::commitment::Params;
use log::debug;
use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;

#[derive(thiserror::Error, Debug)]
#[allow(missing_docs)]
pub enum SrsError {
    #[error("failed to load srs from {0}")]
    LoadError(PathBuf),
    #[error("failed to read srs {0}")]
    ReadError(String),
}

/// Loads the [CommitmentScheme::ParamsVerifier] at `path`.
pub fn load_srs_verifier<Scheme: CommitmentScheme>(
    path: PathBuf,
) -> Result<Scheme::ParamsVerifier, SrsError> {
    debug!("loading srs from {:?}", path);
    let f = File::open(path.clone()).map_err(|_| SrsError::LoadError(path))?;
    let mut reader = BufReader::new(f);
    Params::<'_, Scheme::Curve>::read(&mut reader).map_err(|e| SrsError::ReadError(e.to_string()))
}

/// Loads the [CommitmentScheme::ParamsVerifier] at `path`.
pub fn load_srs_prover<Scheme: CommitmentScheme>(
    path: PathBuf,
) -> Result<Scheme::ParamsProver, SrsError> {
    debug!("loading srs from {:?}", path);
    let f = File::open(path.clone()).map_err(|_| SrsError::LoadError(path.clone()))?;
    let mut reader = BufReader::new(f);
    Params::<'_, Scheme::Curve>::read(&mut reader).map_err(|e| SrsError::ReadError(e.to_string()))
}
