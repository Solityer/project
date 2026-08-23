use crate::algebra::{Point, Scalar};
use crate::prover_backend::GpuTelemetry;
use crate::registry::Expr;
use anyhow::{ensure, Result};
use sha2::{Digest, Sha256};
use std::ffi::CStr;
use std::os::raw::{c_char, c_int};
use std::sync::atomic::{AtomicU64, Ordering};

pub struct GpuProvider {
    device_name: String,
    compute_capability: String,
    kat_vectors_digest: String,
    msm: AtomicU64,
    inner_product: AtomicU64,
    scalar_fold: AtomicU64,
    point_fold: AtomicU64,
    sumcheck: AtomicU64,
}

#[link(name = "zkdl_gpu_backend", kind = "static")]
extern "C" {
    fn zkdl_gpu_last_error() -> *const c_char;
    fn zkdl_gpu_probe(
        name: *mut c_char,
        capacity: usize,
        major: *mut c_int,
        minor: *mut c_int,
    ) -> c_int;
    fn zkdl_gpu_inner_product(
        lhs: *const u8,
        rhs: *const u8,
        count: usize,
        output: *mut u8,
    ) -> c_int;
    fn zkdl_gpu_fold_scalars_split(
        input: *const u8,
        count: usize,
        left: *const u8,
        right: *const u8,
        output: *mut u8,
    ) -> c_int;
    fn zkdl_gpu_fold_points_split(
        points: *const u8,
        count: usize,
        left: *const u8,
        right: *const u8,
        output: *mut u8,
    ) -> c_int;
    fn zkdl_gpu_msm(points: *const u8, scalars: *const u8, count: usize, output: *mut u8) -> c_int;
    fn zkdl_gpu_sumcheck_evaluations(
        columns: *const u8,
        column_count: usize,
        row_count: usize,
        beta: *const u8,
        mask: *const u8,
        mask_count: usize,
        first_round: c_int,
        factor: *const u8,
        lambdas: *const u8,
        residual_count: usize,
        opcodes: *const u32,
        arguments: *const u32,
        constants: *const u8,
        token_count: usize,
        residual_offsets: *const u32,
        degree: usize,
        lookup_values: *const u8,
        lookup_value_count: usize,
        lookup_value_offsets: *const u32,
        lookup_bit_columns: *const u32,
        lookup_bit_count: usize,
        lookup_bit_offsets: *const u32,
        lookup_count: usize,
        output: *mut u8,
    ) -> c_int;
}

fn native_error(operation: &str) -> anyhow::Error {
    let detail = unsafe {
        let pointer = zkdl_gpu_last_error();
        if pointer.is_null() {
            "unknown CUDA error".to_owned()
        } else {
            CStr::from_ptr(pointer).to_string_lossy().into_owned()
        }
    };
    anyhow::anyhow!("formal GPU provider {operation} failed: {detail}")
}

fn scalar_bytes(values: &[Scalar]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 32);
    for value in values {
        out.extend_from_slice(&value.0);
    }
    out
}

fn parse_scalars(bytes: &[u8]) -> Result<Vec<Scalar>> {
    ensure!(
        bytes.len() % 32 == 0,
        "GPU returned misaligned scalar bytes"
    );
    bytes
        .chunks_exact(32)
        .map(|chunk| {
            let mut canonical = [0u8; 32];
            canonical.copy_from_slice(chunk);
            Scalar::from_canonical_bytes(canonical)
        })
        .collect()
}

fn flatten_expression(
    expression: &Expr,
    opcodes: &mut Vec<u32>,
    arguments: &mut Vec<u32>,
    constants: &mut Vec<Scalar>,
    lookup_values: &mut Vec<Scalar>,
    lookup_value_offsets: &mut Vec<u32>,
    lookup_bits: &mut Vec<u32>,
    lookup_bit_offsets: &mut Vec<u32>,
) -> Result<()> {
    match expression {
        Expr::Const(value) => {
            opcodes.push(0);
            arguments.push(0);
            constants.push(*value);
        }
        Expr::Column(index) => {
            opcodes.push(1);
            arguments.push(*index);
            constants.push(Scalar::zero());
        }
        Expr::Add(lhs, rhs) => {
            flatten_expression(
                lhs,
                opcodes,
                arguments,
                constants,
                lookup_values,
                lookup_value_offsets,
                lookup_bits,
                lookup_bit_offsets,
            )?;
            flatten_expression(
                rhs,
                opcodes,
                arguments,
                constants,
                lookup_values,
                lookup_value_offsets,
                lookup_bits,
                lookup_bit_offsets,
            )?;
            opcodes.push(2);
            arguments.push(0);
            constants.push(Scalar::zero());
        }
        Expr::Sub(lhs, rhs) => {
            flatten_expression(
                lhs,
                opcodes,
                arguments,
                constants,
                lookup_values,
                lookup_value_offsets,
                lookup_bits,
                lookup_bit_offsets,
            )?;
            flatten_expression(
                rhs,
                opcodes,
                arguments,
                constants,
                lookup_values,
                lookup_value_offsets,
                lookup_bits,
                lookup_bit_offsets,
            )?;
            opcodes.push(3);
            arguments.push(0);
            constants.push(Scalar::zero());
        }
        Expr::Mul(lhs, rhs) => {
            flatten_expression(
                lhs,
                opcodes,
                arguments,
                constants,
                lookup_values,
                lookup_value_offsets,
                lookup_bits,
                lookup_bit_offsets,
            )?;
            flatten_expression(
                rhs,
                opcodes,
                arguments,
                constants,
                lookup_values,
                lookup_value_offsets,
                lookup_bits,
                lookup_bit_offsets,
            )?;
            opcodes.push(4);
            arguments.push(0);
            constants.push(Scalar::zero());
        }
        Expr::PublicTableLookup {
            values,
            bit_columns,
        } => {
            let lookup_index = u32::try_from(lookup_value_offsets.len() - 1)?;
            lookup_values.extend_from_slice(values);
            lookup_value_offsets.push(u32::try_from(lookup_values.len())?);
            lookup_bits.extend_from_slice(bit_columns);
            lookup_bit_offsets.push(u32::try_from(lookup_bits.len())?);
            opcodes.push(5);
            arguments.push(lookup_index);
            constants.push(Scalar::zero());
        }
    }
    Ok(())
}

impl GpuProvider {
    pub fn open() -> Result<Self> {
        let mut name = [0 as c_char; 256];
        let mut major = 0;
        let mut minor = 0;
        let status =
            unsafe { zkdl_gpu_probe(name.as_mut_ptr(), name.len(), &mut major, &mut minor) };
        if status != 0 {
            return Err(native_error("probe"));
        }
        let device_name = unsafe { CStr::from_ptr(name.as_ptr()) }
            .to_string_lossy()
            .into_owned();
        ensure!(
            !device_name.is_empty(),
            "CUDA provider returned empty device identity"
        );
        let mut provider = Self {
            device_name,
            compute_capability: format!("{major}.{minor}"),
            kat_vectors_digest: String::new(),
            msm: 0.into(),
            inner_product: 0.into(),
            scalar_fold: 0.into(),
            point_fold: 0.into(),
            sumcheck: 0.into(),
        };
        provider.kat_vectors_digest = provider.cross_backend_known_answer_test()?;
        // KAT calls are a route gate, not proof work.  Reset proof telemetry so
        // a GPU envelope cannot pass coverage using probe/KAT kernels alone.
        provider.msm.store(0, Ordering::Relaxed);
        provider.inner_product.store(0, Ordering::Relaxed);
        provider.scalar_fold.store(0, Ordering::Relaxed);
        provider.point_fold.store(0, Ordering::Relaxed);
        provider.sumcheck.store(0, Ordering::Relaxed);
        Ok(provider)
    }

    pub fn telemetry(&self) -> GpuTelemetry {
        GpuTelemetry {
            provider_id: "zkDL-native-CUDA-BLS12-381-v1".into(),
            device_name: self.device_name.clone(),
            compute_capability: self.compute_capability.clone(),
            cross_backend_kat_passed: true,
            kat_vectors_digest: self.kat_vectors_digest.clone(),
            probe_kernel_invocations: 1,
            msm_kernel_invocations: self.msm.load(Ordering::Relaxed),
            field_inner_product_invocations: self.inner_product.load(Ordering::Relaxed),
            scalar_fold_invocations: self.scalar_fold.load(Ordering::Relaxed),
            point_fold_invocations: self.point_fold.load(Ordering::Relaxed),
            sumcheck_kernel_invocations: self.sumcheck.load(Ordering::Relaxed),
        }
    }

    fn cross_backend_known_answer_test(&self) -> Result<String> {
        let points = (0u64..4)
            .map(|i| Point::hash_to_curve(b"zkDL-GPU-cross-backend-KAT-v1", &i.to_be_bytes()))
            .collect::<Vec<_>>();
        let large = Scalar::hash_to_field(
            b"zkDL-GPU-KAT-large-scalar-v1",
            b"canonical-large-nontrivial",
        );
        ensure!(
            large != Scalar::zero() && large != Scalar::one(),
            "GPU KAT scalar derivation degenerated"
        );
        let scalars = [
            Scalar::zero(),
            Scalar::one(),
            large,
            Scalar::from_u64(u64::MAX),
        ];
        let gpu_msm = self.msm(&points, &scalars)?;
        let cpu_msm = crate::algebra::msm(&points, &scalars)?;
        ensure!(
            gpu_msm.eq_checked(cpu_msm)?,
            "CUDA/blst G1 ABI or scalar order KAT failed for MSM"
        );
        let rhs = [
            large,
            Scalar::from_u64(17),
            Scalar::one(),
            Scalar::from_u64(u64::MAX - 1),
        ];
        ensure!(
            self.inner_product(&scalars, &rhs)? == crate::algebra::inner_product(&scalars, &rhs)?,
            "CUDA/blst Fr encoding KAT failed for inner product"
        );
        let left = Scalar::hash_to_field(b"zkDL-GPU-KAT-fold-left-v1", b"left");
        let right = left.inverse()?;
        let scalar_fold = self.fold_scalars(&scalars, left, right)?;
        let expected_scalar = (0..2)
            .map(|i| scalars[i] * left + scalars[i + 2] * right)
            .collect::<Vec<_>>();
        ensure!(
            scalar_fold == expected_scalar,
            "CUDA/blst Fr fold KAT failed"
        );
        let point_fold = self.fold_points(&points, left, right)?;
        for i in 0..2 {
            let expected = points[i].mul(left)?.add(points[i + 2].mul(right)?)?;
            ensure!(
                point_fold[i].eq_checked(expected)?,
                "CUDA/blst G1 fold KAT failed at {i}"
            );
        }
        let mut h = Sha256::new();
        h.update(b"zkDL-GPU-cross-backend-KAT-v1\0");
        for point in &points {
            h.update(point.0);
        }
        for scalar in scalars.iter().chain(&rhs) {
            h.update(scalar.0);
        }
        h.update(left.0);
        h.update(right.0);
        h.update(gpu_msm.0);
        for value in scalar_fold {
            h.update(value.0);
        }
        for value in point_fold {
            h.update(value.0);
        }
        Ok(hex::encode(h.finalize()))
    }

    pub fn ensure_formal_prover_coverage(&self) -> Result<()> {
        let telemetry = self.telemetry();
        ensure!(
            telemetry.msm_kernel_invocations > 0,
            "GPU proof did not execute commitment/IPA MSM kernels"
        );
        ensure!(
            telemetry.field_inner_product_invocations > 0,
            "GPU proof did not execute IPA inner-product kernels"
        );
        ensure!(
            telemetry.scalar_fold_invocations > 0 && telemetry.point_fold_invocations > 0,
            "GPU proof did not execute IPA fold kernels"
        );
        ensure!(
            telemetry.sumcheck_kernel_invocations > 0,
            "GPU proof did not execute Sumcheck residual/fold kernels"
        );
        Ok(())
    }

    pub fn msm(&self, points: &[Point], scalars: &[Scalar]) -> Result<Point> {
        ensure!(
            !points.is_empty() && points.len() == scalars.len(),
            "GPU MSM shape mismatch"
        );
        let raw = points
            .iter()
            .map(|point| point.to_raw_p1())
            .collect::<Result<Vec<_>>>()?;
        let mut output = [0u8; 144];
        let scalar_data = scalar_bytes(scalars);
        let status = unsafe {
            zkdl_gpu_msm(
                raw.as_ptr().cast(),
                scalar_data.as_ptr(),
                points.len(),
                output.as_mut_ptr(),
            )
        };
        if status != 0 {
            return Err(native_error("MSM"));
        }
        self.msm.fetch_add(1, Ordering::Relaxed);
        Point::from_raw_p1(&output)
    }

    pub fn inner_product(&self, lhs: &[Scalar], rhs: &[Scalar]) -> Result<Scalar> {
        ensure!(
            !lhs.is_empty() && lhs.len() == rhs.len(),
            "GPU inner-product shape mismatch"
        );
        let lhs_bytes = scalar_bytes(lhs);
        let rhs_bytes = scalar_bytes(rhs);
        let mut output = [0u8; 32];
        let status = unsafe {
            zkdl_gpu_inner_product(
                lhs_bytes.as_ptr(),
                rhs_bytes.as_ptr(),
                lhs.len(),
                output.as_mut_ptr(),
            )
        };
        if status != 0 {
            return Err(native_error("inner product"));
        }
        self.inner_product.fetch_add(1, Ordering::Relaxed);
        Scalar::from_canonical_bytes(output)
    }

    pub fn fold_scalars(
        &self,
        values: &[Scalar],
        left: Scalar,
        right: Scalar,
    ) -> Result<Vec<Scalar>> {
        ensure!(
            values.len() >= 2 && values.len().is_power_of_two(),
            "GPU scalar fold needs power-of-two input"
        );
        let input = scalar_bytes(values);
        let mut output = vec![0u8; values.len() / 2 * 32];
        let status = unsafe {
            zkdl_gpu_fold_scalars_split(
                input.as_ptr(),
                values.len(),
                left.0.as_ptr(),
                right.0.as_ptr(),
                output.as_mut_ptr(),
            )
        };
        if status != 0 {
            return Err(native_error("scalar fold"));
        }
        self.scalar_fold.fetch_add(1, Ordering::Relaxed);
        parse_scalars(&output)
    }

    pub fn fold_points(&self, values: &[Point], left: Scalar, right: Scalar) -> Result<Vec<Point>> {
        ensure!(
            values.len() >= 2 && values.len().is_power_of_two(),
            "GPU point fold needs power-of-two input"
        );
        let raw = values
            .iter()
            .map(|point| point.to_raw_p1())
            .collect::<Result<Vec<_>>>()?;
        let mut output = vec![[0u8; 144]; values.len() / 2];
        let status = unsafe {
            zkdl_gpu_fold_points_split(
                raw.as_ptr().cast(),
                values.len(),
                left.0.as_ptr(),
                right.0.as_ptr(),
                output.as_mut_ptr().cast(),
            )
        };
        if status != 0 {
            return Err(native_error("point fold"));
        }
        self.point_fold.fetch_add(1, Ordering::Relaxed);
        output.iter().map(Point::from_raw_p1).collect()
    }

    pub fn sumcheck_evaluations(
        &self,
        columns: &[Vec<Scalar>],
        beta: &[Scalar],
        mask: &[Scalar],
        first_round: bool,
        factor: Scalar,
        lambdas: &[Scalar],
        residuals: &[Expr],
        degree: usize,
    ) -> Result<Vec<Scalar>> {
        ensure!(
            !columns.is_empty() && !beta.is_empty() && beta.len().is_power_of_two(),
            "GPU Sumcheck shape is empty or non-binary"
        );
        ensure!(
            columns.iter().all(|column| column.len() == beta.len()),
            "GPU Sumcheck column shape mismatch"
        );
        ensure!(
            lambdas.len() == residuals.len() && !residuals.is_empty(),
            "GPU Sumcheck residual shape mismatch"
        );
        let mut opcodes = Vec::new();
        let mut arguments = Vec::new();
        let mut constants = Vec::new();
        let mut offsets = vec![0u32];
        let mut lookup_values = Vec::new();
        let mut lookup_value_offsets = vec![0u32];
        let mut lookup_bits = Vec::new();
        let mut lookup_bit_offsets = vec![0u32];
        for residual in residuals {
            flatten_expression(
                residual,
                &mut opcodes,
                &mut arguments,
                &mut constants,
                &mut lookup_values,
                &mut lookup_value_offsets,
                &mut lookup_bits,
                &mut lookup_bit_offsets,
            )?;
            offsets.push(u32::try_from(opcodes.len())?);
        }
        ensure!(
            opcodes.len() <= u32::MAX as usize,
            "GPU expression bytecode too large"
        );
        let flattened_columns = columns
            .iter()
            .flat_map(|column| column.iter().copied())
            .collect::<Vec<_>>();
        let column_bytes = scalar_bytes(&flattened_columns);
        let beta_bytes = scalar_bytes(beta);
        let mask_bytes = scalar_bytes(mask);
        let lambda_bytes = scalar_bytes(lambdas);
        let constant_bytes = scalar_bytes(&constants);
        let lookup_value_bytes = scalar_bytes(&lookup_values);
        let mut output = vec![0u8; (degree + 1) * 32];
        let status = unsafe {
            zkdl_gpu_sumcheck_evaluations(
                column_bytes.as_ptr(),
                columns.len(),
                beta.len(),
                beta_bytes.as_ptr(),
                mask_bytes.as_ptr(),
                mask.len(),
                i32::from(first_round),
                factor.0.as_ptr(),
                lambda_bytes.as_ptr(),
                residuals.len(),
                opcodes.as_ptr(),
                arguments.as_ptr(),
                constant_bytes.as_ptr(),
                opcodes.len(),
                offsets.as_ptr(),
                degree,
                lookup_value_bytes.as_ptr(),
                lookup_values.len(),
                lookup_value_offsets.as_ptr(),
                lookup_bits.as_ptr(),
                lookup_bits.len(),
                lookup_bit_offsets.as_ptr(),
                lookup_value_offsets.len() - 1,
                output.as_mut_ptr(),
            )
        };
        if status != 0 {
            return Err(native_error("Sumcheck residual evaluation"));
        }
        self.sumcheck.fetch_add(1, Ordering::Relaxed);
        parse_scalars(&output)
    }
}
