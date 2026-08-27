//! Authenticated, scoped CUDA kernel telemetry for formal GPU cells.
//!
//! The implementation is backed by CUPTI activity records compiled into a C
//! shim.  A device marker alone is never accepted as evidence of GPU
//! execution: every formal prove/verify scope must return at least one kernel
//! record with a nonce and a non-empty name/category inventory.

use serde::{Deserialize, Serialize};

const OUTPUT_BYTES: usize = 256 * 1024;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct KernelName {
    pub(crate) name: String,
    pub(crate) category: String,
    pub(crate) count: u64,
    pub(crate) device_id: u32,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct GpuTelemetry {
    pub(crate) schema: String,
    pub(crate) scope: String,
    pub(crate) scope_nonce: String,
    pub(crate) scope_start_ns: u64,
    pub(crate) scope_end_ns: u64,
    pub(crate) kernel_count: u64,
    pub(crate) dropped_records: u64,
    pub(crate) active_device_id: u32,
    pub(crate) kernel_names: Vec<KernelName>,
}

#[allow(unsafe_code)]
unsafe extern "C" {
    fn ezkl_gpu_telemetry_init() -> i32;
    fn ezkl_gpu_scope_begin(scope_name: *const std::ffi::c_char, nonce: *mut u64) -> i32;
    fn ezkl_gpu_scope_end(nonce: u64, output: *mut std::ffi::c_char, output_size: usize) -> i32;
}

fn status_error(operation: &str, status: i32) -> String {
    format!(
        "CUPTI {operation} failed with status {status}; refusing unauthenticated CUDA execution"
    )
}

pub(crate) fn ensure_available() -> Result<(), String> {
    #[allow(unsafe_code)]
    let status = unsafe { ezkl_gpu_telemetry_init() };
    if status == 0 {
        Ok(())
    } else {
        Err(status_error("initialization", status))
    }
}

pub(crate) struct Scope {
    nonce: u64,
}

impl Scope {
    pub(crate) fn begin(name: &str) -> Result<Self, String> {
        ensure_available()?;
        let name =
            std::ffi::CString::new(name).map_err(|_| "CUPTI scope name contains NUL".to_owned())?;
        let mut nonce = 0u64;
        #[allow(unsafe_code)]
        let status = unsafe { ezkl_gpu_scope_begin(name.as_ptr(), &mut nonce) };
        if status != 0 {
            return Err(status_error("scope begin", status));
        }
        if nonce == 0 {
            return Err("CUPTI scope returned an empty nonce".into());
        }
        Ok(Self { nonce })
    }

    pub(crate) fn finish(self) -> Result<GpuTelemetry, String> {
        let mut output = vec![0i8; OUTPUT_BYTES];
        #[allow(unsafe_code)]
        let status = unsafe { ezkl_gpu_scope_end(self.nonce, output.as_mut_ptr(), output.len()) };
        if status != 0 {
            return Err(status_error("scope end", status));
        }
        #[allow(unsafe_code)]
        let bytes =
            unsafe { std::slice::from_raw_parts(output.as_ptr().cast::<u8>(), output.len()) };
        let end = bytes
            .iter()
            .position(|byte| *byte == 0)
            .ok_or_else(|| "CUPTI scope output was not NUL terminated".to_owned())?;
        let telemetry: GpuTelemetry = serde_json::from_slice(&bytes[..end])
            .map_err(|error| format!("invalid CUPTI scope output: {error}"))?;
        if telemetry.schema != "ezkl_cuda_kernel_telemetry_v1"
            || telemetry.scope_nonce.is_empty()
            || telemetry.scope.is_empty()
            || telemetry.scope_start_ns == 0
            || telemetry.scope_end_ns < telemetry.scope_start_ns
            || telemetry.kernel_count == 0
            || telemetry.kernel_names.is_empty()
            || telemetry.active_device_id != 0
            || telemetry.dropped_records != 0
        {
            return Err(
                "CUPTI formal scope did not provide complete non-zero kernel evidence".into(),
            );
        }
        if telemetry.kernel_names.iter().any(|kernel| {
            kernel.name.is_empty()
                || kernel.category.is_empty()
                || kernel.count == 0
                || kernel.device_id != telemetry.active_device_id
        }) {
            return Err("CUPTI formal scope contains incomplete kernel identity evidence".into());
        }
        let inventory_count: u64 = telemetry
            .kernel_names
            .iter()
            .map(|kernel| kernel.count)
            .sum();
        if inventory_count != telemetry.kernel_count {
            return Err("CUPTI formal scope kernel inventory count mismatch".into());
        }
        Ok(telemetry)
    }
}
