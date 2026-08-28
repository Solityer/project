use crate::curve::{ScalarCfg, ScalarField};
use icicle_core::impl_poseidon;

impl_poseidon!("bn254", bn254, ScalarField, ScalarCfg);
