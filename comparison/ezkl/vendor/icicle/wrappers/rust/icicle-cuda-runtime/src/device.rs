use crate::{
    bindings::{
        cudaFreeAsync, cudaGetDevice, cudaGetDeviceCount, cudaMallocAsync, cudaMemGetInfo, cudaPointerAttributes,
        cudaPointerGetAttributes, cudaSetDevice,
    },
    error::{CudaResult, CudaResultWrap},
    stream::CudaStream,
};
use std::mem::MaybeUninit;

pub fn set_device(device_id: usize) -> CudaResult<()> {
    unsafe { cudaSetDevice(device_id as i32) }.wrap()
}

pub fn get_device_count() -> CudaResult<usize> {
    let mut count = 0;
    unsafe { cudaGetDeviceCount(&mut count) }.wrap_value(count as usize)
}

pub fn get_device() -> CudaResult<usize> {
    let mut device_id = 0;
    unsafe { cudaGetDevice(&mut device_id) }.wrap_value(device_id as usize)
}

/// Returns the CUDA runtime's current free and total device memory in bytes.
pub fn get_memory_info() -> CudaResult<(usize, usize)> {
    let mut free_memory = 0usize;
    let mut total_memory = 0usize;
    unsafe {
        cudaMemGetInfo(&mut free_memory as *mut usize, &mut total_memory as *mut usize).wrap()?;
    }
    Ok((free_memory, total_memory))
}

/// Returns CUDA's stable UUID for a runtime device ordinal.
pub fn get_device_uuid(device_id: usize) -> CudaResult<String> {
    let mut properties = MaybeUninit::<crate::bindings::cudaDeviceProp>::uninit();
    unsafe {
        crate::bindings::cudaGetDeviceProperties(properties.as_mut_ptr(), device_id as i32).wrap()?;
        let uuid = properties.assume_init().uuid;
        let bytes = uuid.bytes;
        let raw = bytes.iter().map(|byte| format!("{:02x}", *byte as u8)).collect::<String>();
        Ok(format!(
            "GPU-{}-{}-{}-{}-{}",
            &raw[0..8], &raw[8..12], &raw[12..16], &raw[16..20], &raw[20..32],
        ))
    }
}

pub fn get_device_from_pointer(ptr: *const ::std::os::raw::c_void) -> CudaResult<usize> {
    let mut ptr_attributes = MaybeUninit::<cudaPointerAttributes>::uninit();
    unsafe {
        cudaPointerGetAttributes(ptr_attributes.as_mut_ptr(), ptr).wrap()?;
        Ok(ptr_attributes
            .assume_init()
            .device as usize)
    }
}

pub fn check_device(device_id: usize) {
    match device_id == get_device().unwrap() {
        true => (),
        false => panic!("Attempt to use on a different device"),
    }
}

// This function pre-allocates default memory pool and warms the GPU up
// so that subsequent memory allocations and other calls are not slowed down
pub fn warmup(stream: &CudaStream) -> CudaResult<()> {
    let mut device_ptr = MaybeUninit::<*mut std::ffi::c_void>::uninit();
    let mut free_memory: usize = 0;
    let mut _total_memory: usize = 0;
    unsafe {
        cudaMemGetInfo(&mut free_memory as *mut usize, &mut _total_memory as *mut usize).wrap()?;
        cudaMallocAsync(device_ptr.as_mut_ptr(), free_memory >> 1, stream.handle).wrap()?;
        cudaFreeAsync(device_ptr.assume_init(), stream.handle).wrap()
    }
}
