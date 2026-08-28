use crate::curve::{ScalarCfg, ScalarField};
use icicle_core::impl_poseidon2;

impl_poseidon2!("bn254", bn254, ScalarField, ScalarCfg);
