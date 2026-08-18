use crate::algebra::{inner_product, msm, Point, Scalar};
use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GpuTelemetry {
    pub provider_id: String,
    pub device_name: String,
    pub compute_capability: String,
    pub cross_backend_kat_passed: bool,
    pub kat_vectors_digest: String,
    pub probe_kernel_invocations: u64,
    pub msm_kernel_invocations: u64,
    pub field_inner_product_invocations: u64,
    pub scalar_fold_invocations: u64,
    pub point_fold_invocations: u64,
    pub sumcheck_kernel_invocations: u64,
}

pub enum ProverBackend {
    Cpu,
    #[cfg(feature = "gpu")]
    Gpu(crate::gpu::GpuProvider),
}

impl ProverBackend {
    pub fn open(device: &str) -> Result<Self> {
        match device {
            "cpu" => Ok(Self::Cpu),
            "gpu" => {
                #[cfg(feature = "gpu")]
                { Ok(Self::Gpu(crate::gpu::GpuProvider::open()?)) }
                #[cfg(not(feature = "gpu"))]
                { bail!("GPU proof requested from CPU-only binary; build the formal GPU binary with --features gpu") }
            }
            _ => bail!("device must be cpu|gpu"),
        }
    }

    pub fn msm(&self, points: &[Point], scalars: &[Scalar]) -> Result<Point> {
        match self {
            Self::Cpu => msm(points, scalars),
            #[cfg(feature = "gpu")]
            Self::Gpu(gpu) => gpu.msm(points, scalars),
        }
    }

    pub fn inner_product(&self, lhs: &[Scalar], rhs: &[Scalar]) -> Result<Scalar> {
        match self {
            Self::Cpu => inner_product(lhs, rhs),
            #[cfg(feature = "gpu")]
            Self::Gpu(gpu) => gpu.inner_product(lhs, rhs),
        }
    }

    pub fn fold_scalars(&self, values: &[Scalar], left: Scalar, right: Scalar) -> Result<Vec<Scalar>> {
        match self {
            Self::Cpu => {
                let half = values.len() / 2;
                Ok((0..half).map(|i| values[i] * left + values[i + half] * right).collect())
            }
            #[cfg(feature = "gpu")]
            Self::Gpu(gpu) => gpu.fold_scalars(values, left, right),
        }
    }

    pub fn fold_points(&self, values: &[Point], left: Scalar, right: Scalar) -> Result<Vec<Point>> {
        match self {
            Self::Cpu => {
                let half = values.len() / 2;
                (0..half).map(|i| values[i].mul(left)?.add(values[i + half].mul(right)?)).collect()
            }
            #[cfg(feature = "gpu")]
            Self::Gpu(gpu) => gpu.fold_points(values, left, right),
        }
    }

    #[cfg(feature = "gpu")]
    pub fn gpu(&self) -> Option<&crate::gpu::GpuProvider> {
        match self { Self::Gpu(gpu) => Some(gpu), Self::Cpu => None }
    }

    pub fn finish_gpu(&self) -> Result<Option<GpuTelemetry>> {
        match self {
            Self::Cpu => Ok(None),
            #[cfg(feature = "gpu")]
            Self::Gpu(gpu) => { gpu.ensure_formal_prover_coverage()?; Ok(Some(gpu.telemetry())) }
        }
    }
}
