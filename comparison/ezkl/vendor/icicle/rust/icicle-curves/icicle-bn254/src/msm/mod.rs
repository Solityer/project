use crate::curve::CurveCfg;
#[cfg(not(feature = "no_g2"))]
use crate::curve::G2CurveCfg;
use icicle_core::{
    curve::{Affine, Curve, Projective},
    impl_msm,
    msm::{MSMConfig, MSM},
};
use icicle_runtime::{
    errors::eIcicleError,
    memory::{DeviceSlice, HostOrDeviceSlice},
};

impl_msm!("bn254", bn254, CurveCfg);
#[cfg(not(feature = "no_g2"))]
impl_msm!("bn254_g2", bn254_g2, G2CurveCfg);
