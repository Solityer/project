use crate::curve::{ScalarCfg, ScalarField};
use icicle_core::{
    impl_vec_ops_field,
    vec_ops::{/*BitReverseConfig,*/ VecOps, VecOpsConfig},
};
use icicle_runtime::{errors::eIcicleError, memory::HostOrDeviceSlice};

use icicle_core::program::Program;
use icicle_core::traits::FieldImpl;

impl_vec_ops_field!("bn254", bn254, ScalarField, ScalarCfg);
