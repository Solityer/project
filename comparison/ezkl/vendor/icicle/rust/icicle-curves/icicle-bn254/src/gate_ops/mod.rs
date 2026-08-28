use crate::curve::{ScalarCfg, ScalarField};
use icicle_core::{
    gate_ops::{
        CalculationData, GateData, GateOps, GateOpsConfig, HornerData, LookupConfig, LookupData,
    },
    impl_gate_ops_field,
};
use icicle_runtime::{errors::eIcicleError, memory::HostOrDeviceSlice};

impl_gate_ops_field!("bn254", bn254, ScalarField, ScalarCfg);
