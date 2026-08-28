use std::ffi::{CStr, CString};
use std::fs;
use std::io::Read;
use std::os::raw::{c_char, c_void};
use std::os::unix::fs::MetadataExt;
use std::path::{Component, Path, PathBuf};

use serde::Deserialize;
use sha2::{Digest, Sha256};

use crate::device::{Device, DeviceProperties};
use crate::errors::eIcicleError;
use crate::stream::IcicleStream;

pub type IcicleStreamHandle = *mut c_void;

extern "C" {
    // This symbol is declared only in the private ICICLE source header.  The
    // public runtime API deliberately has no arbitrary-path loader.
    fn icicle_load_backend_file_internal(path: *const c_char) -> eIcicleError;
    fn icicle_set_device(device: &Device) -> eIcicleError;
    fn icicle_set_default_device(device: &Device) -> eIcicleError;
    fn icicle_get_active_device(device: &mut Device) -> eIcicleError;
    fn icicle_is_host_memory(ptr: *const c_void) -> eIcicleError;
    fn icicle_is_active_device_memory(ptr: *const c_void) -> eIcicleError;
    fn icicle_get_device_count(device_count: &mut i32) -> eIcicleError;
    fn icicle_is_device_available(device: &Device) -> eIcicleError;
    pub fn icicle_malloc(ptr: *mut *mut c_void, size: usize) -> eIcicleError;
    pub fn icicle_malloc_async(
        ptr: *mut *mut c_void,
        size: usize,
        stream: IcicleStreamHandle,
    ) -> eIcicleError;
    pub fn icicle_free(ptr: *mut c_void) -> eIcicleError;
    pub fn icicle_free_async(ptr: *mut c_void, stream: IcicleStreamHandle) -> eIcicleError;
    fn icicle_get_available_memory(total: *mut usize, free: *mut usize) -> eIcicleError;
    pub fn icicle_copy_to_host(dst: *mut c_void, src: *const c_void, size: usize) -> eIcicleError;
    pub fn icicle_copy_to_host_async(
        dst: *mut c_void,
        src: *const c_void,
        size: usize,
        stream: IcicleStreamHandle,
    ) -> eIcicleError;
    pub fn icicle_copy_to_device(dst: *mut c_void, src: *const c_void, size: usize)
        -> eIcicleError;
    pub fn icicle_copy_to_device_async(
        dst: *mut c_void,
        src: *const c_void,
        size: usize,
        stream: IcicleStreamHandle,
    ) -> eIcicleError;
    pub fn icicle_copy(dst: *mut c_void, src: *const c_void, size: usize) -> eIcicleError;
    pub fn icicle_copy_async(
        dst: *mut c_void,
        src: *const c_void,
        size: usize,
        stream: IcicleStreamHandle,
    ) -> eIcicleError;
    pub fn icicle_create_stream(stream: *mut IcicleStreamHandle) -> eIcicleError;
    pub fn icicle_destroy_stream(stream: IcicleStreamHandle) -> eIcicleError;
    pub fn icicle_stream_synchronize(stream: IcicleStreamHandle) -> eIcicleError;
    fn icicle_device_synchronize() -> eIcicleError;
    fn icicle_get_device_properties(properties: *mut DeviceProperties) -> eIcicleError;
    fn icicle_get_registered_devices(output: *mut c_char, output_size: usize) -> eIcicleError;
    fn icicle_memset(ptr: *mut c_void, value: i32, size: usize) -> eIcicleError;
    fn icicle_memset_async(
        ptr: *mut c_void,
        value: i32,
        size: usize,
        stream: *mut c_void,
    ) -> eIcicleError;
}

const CUDA_BACKEND_SCHEMA: &str = "ezkl_formal_icicle_cuda_backend_v1";
const CUDA_BACKEND_NAME: &str = "cuda";
const ICICLE_SOURCE_COMMIT: &str = "012e00694f4cf399fe7a42d9cfbfa6cd7a60f876";
const ICICLE_CUDA_ABI: &str = "icicle_cuda_bn254_3.7.0_abi_v1";
const CUDA_BACKEND_ROOT: &str = "resources/icicle_cuda_backend";
const CUDA_BACKEND_MANIFEST: &str = "resources/icicle_cuda_backend/backend.manifest.json";
const CUDA_BACKEND_LIBRARY_RELATIVE_PATHS: [&str; 4] = [
    "resources/icicle_cuda_backend/lib/backend/libicicle_backend_cuda_device.so",
    "resources/icicle_cuda_backend/lib/backend/libicicle_backend_cuda_curve_bn254.so",
    "resources/icicle_cuda_backend/lib/backend/libicicle_backend_cuda_field_bn254.so",
    "resources/icicle_cuda_backend/lib/backend/libicicle_backend_cuda_hash.so",
];
const CUDA_BACKEND_LIBRARY_NAMES: [&str; 4] = [
    "libicicle_backend_cuda_device.so",
    "libicicle_backend_cuda_curve_bn254.so",
    "libicicle_backend_cuda_field_bn254.so",
    "libicicle_backend_cuda_hash.so",
];

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct BackendManifest {
    schema: String,
    backend: String,
    icicle_commit: String,
    abi: String,
    library_inventory: Vec<BackendLibrary>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct BackendLibrary {
    path: String,
    size: u64,
    sha256: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ParentSnapshot {
    device: u64,
    inode: u64,
    mode: u32,
    links: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct FileSnapshot {
    parents: Vec<ParentSnapshot>,
    device: u64,
    inode: u64,
    mode: u32,
    links: u64,
    size: u64,
    digest: [u8; 32],
}

fn backend_error() -> eIcicleError {
    eIcicleError::BackendLoadFailed
}

fn lexical_normalize(path: &Path) -> Result<PathBuf, eIcicleError> {
    let mut normalized = if path.is_absolute() {
        PathBuf::from("/")
    } else {
        PathBuf::new()
    };
    for component in path.components() {
        match component {
            Component::Prefix(prefix) => normalized.push(prefix.as_os_str()),
            Component::RootDir => normalized.push(component.as_os_str()),
            Component::CurDir => {}
            Component::ParentDir => {
                if !normalized.pop() {
                    return Err(backend_error());
                }
            }
            Component::Normal(part) => normalized.push(part),
        }
    }
    if normalized.as_os_str().is_empty() {
        return Err(backend_error());
    }
    Ok(normalized)
}

fn require_directory_components(path: &Path) -> Result<PathBuf, eIcicleError> {
    let lexical = lexical_normalize(path)?;
    let mut current = if lexical.is_absolute() {
        PathBuf::from("/")
    } else {
        PathBuf::new()
    };
    for component in lexical.components() {
        match component {
            Component::RootDir | Component::Prefix(_) => {}
            Component::Normal(part) => {
                current.push(part);
                let metadata = fs::symlink_metadata(&current).map_err(|_| backend_error())?;
                if !metadata.file_type().is_dir() {
                    return Err(backend_error());
                }
            }
            Component::CurDir | Component::ParentDir => return Err(backend_error()),
        }
    }
    let resolved = fs::canonicalize(&lexical).map_err(|_| backend_error())?;
    if !resolved.is_dir() {
        return Err(backend_error());
    }
    Ok(resolved)
}

fn require_local_directory(root: &Path, relative: &str) -> Result<PathBuf, eIcicleError> {
    let relative_path = Path::new(relative);
    if relative_path.is_absolute()
        || relative_path.components().any(|component| {
            matches!(
                component,
                Component::CurDir
                    | Component::ParentDir
                    | Component::RootDir
                    | Component::Prefix(_)
            )
        })
    {
        return Err(backend_error());
    }
    let candidate = root.join(relative_path);
    let resolved = require_directory_components(&candidate)?;
    if !resolved.starts_with(root) {
        return Err(backend_error());
    }
    Ok(resolved)
}

fn require_local_file(root: &Path, relative: &str) -> Result<PathBuf, eIcicleError> {
    let relative_path = Path::new(relative);
    if relative_path.is_absolute()
        || relative_path.components().any(|component| {
            matches!(
                component,
                Component::CurDir
                    | Component::ParentDir
                    | Component::RootDir
                    | Component::Prefix(_)
            )
        })
    {
        return Err(backend_error());
    }
    let lexical = root.join(relative_path);
    let mut current = root.to_path_buf();
    let components: Vec<_> = relative_path.components().collect();
    if components.is_empty() {
        return Err(backend_error());
    }
    for (index, component) in components.iter().enumerate() {
        let Component::Normal(part) = component else {
            return Err(backend_error());
        };
        current.push(part);
        let metadata = fs::symlink_metadata(&current).map_err(|_| backend_error())?;
        if index + 1 == components.len() {
            if !metadata.file_type().is_file() || metadata.nlink() != 1 {
                return Err(backend_error());
            }
        } else if !metadata.file_type().is_dir() {
            return Err(backend_error());
        }
    }
    let resolved = fs::canonicalize(&lexical).map_err(|_| backend_error())?;
    if !resolved.starts_with(root) || resolved != lexical {
        return Err(backend_error());
    }
    Ok(resolved)
}

fn digest_file(path: &Path) -> Result<[u8; 32], eIcicleError> {
    let mut file = fs::File::open(path).map_err(|_| backend_error())?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 1024 * 1024];
    loop {
        let read = file.read(&mut buffer).map_err(|_| backend_error())?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hasher.finalize().into())
}

fn regular_file_snapshot(path: &Path) -> Result<FileSnapshot, eIcicleError> {
    // Re-check every parent component before and after hashing.  A
    // final-file lstat alone would miss replacing a checked directory with a
    // symlink (or with another directory) while digesting or loading it.
    let lexical = lexical_normalize(path)?;
    let mut current = if lexical.is_absolute() {
        PathBuf::from("/")
    } else {
        PathBuf::new()
    };
    let components: Vec<_> = lexical.components().collect();
    let mut parents_before = Vec::new();
    let mut before = None;
    for (index, component) in components.iter().enumerate() {
        let Component::Normal(part) = component else {
            if matches!(component, Component::RootDir | Component::Prefix(_)) {
                continue;
            }
            return Err(backend_error());
        };
        current.push(part);
        let metadata = fs::symlink_metadata(&current).map_err(|_| backend_error())?;
        if index + 1 != components.len() {
            if !metadata.file_type().is_dir() {
                return Err(backend_error());
            }
            parents_before.push(ParentSnapshot {
                device: metadata.dev(),
                inode: metadata.ino(),
                mode: metadata.mode(),
                links: metadata.nlink(),
            });
        } else if !metadata.file_type().is_file() || metadata.nlink() != 1 {
            return Err(backend_error());
        } else {
            before = Some(metadata);
        }
    }
    let before = before.ok_or_else(backend_error)?;
    let digest = digest_file(path)?;
    let mut parents_after = Vec::new();
    let mut after = None;
    let mut current = if lexical.is_absolute() {
        PathBuf::from("/")
    } else {
        PathBuf::new()
    };
    for (index, component) in components.iter().enumerate() {
        let Component::Normal(part) = component else {
            if matches!(component, Component::RootDir | Component::Prefix(_)) {
                continue;
            }
            return Err(backend_error());
        };
        current.push(part);
        let metadata = fs::symlink_metadata(&current).map_err(|_| backend_error())?;
        if index + 1 != components.len() {
            if !metadata.file_type().is_dir() {
                return Err(backend_error());
            }
            parents_after.push(ParentSnapshot {
                device: metadata.dev(),
                inode: metadata.ino(),
                mode: metadata.mode(),
                links: metadata.nlink(),
            });
        } else {
            after = Some(metadata);
        }
    }
    let after = after.ok_or_else(backend_error)?;
    if !after.file_type().is_file()
        || after.nlink() != 1
        || before.dev() != after.dev()
        || before.ino() != after.ino()
        || before.mode() != after.mode()
        || before.nlink() != after.nlink()
        || before.size() != after.size()
        || parents_before != parents_after
    {
        return Err(backend_error());
    }
    Ok(FileSnapshot {
        parents: parents_after,
        device: after.dev(),
        inode: after.ino(),
        mode: after.mode(),
        links: after.nlink(),
        size: after.size(),
        digest,
    })
}

fn sha256_string(digest: [u8; 32]) -> String {
    let mut value = String::from("sha256:");
    for byte in digest {
        value.push_str(&format!("{byte:02x}"));
    }
    value
}

fn exact_directory_entries(path: &Path, expected: &[&str]) -> Result<(), eIcicleError> {
    let mut actual = Vec::new();
    for entry in fs::read_dir(path).map_err(|_| backend_error())? {
        let entry = entry.map_err(|_| backend_error())?;
        let name = entry.file_name();
        if name.to_str().is_none() {
            return Err(backend_error());
        }
        let metadata = fs::symlink_metadata(entry.path()).map_err(|_| backend_error())?;
        if metadata.file_type().is_symlink() {
            return Err(backend_error());
        }
        actual.push(name);
    }
    actual.sort();
    let mut expected_names: Vec<_> = expected
        .iter()
        .map(|name| std::ffi::OsString::from(name))
        .collect();
    expected_names.sort();
    if actual != expected_names {
        return Err(backend_error());
    }
    Ok(())
}

fn read_backend_manifest(
    root: &Path,
    manifest_path: &Path,
) -> Result<(BackendManifest, FileSnapshot), eIcicleError> {
    let initial = regular_file_snapshot(manifest_path)?;
    let bytes = fs::read(manifest_path).map_err(|_| backend_error())?;
    let final_snapshot = regular_file_snapshot(manifest_path)?;
    if initial != final_snapshot {
        return Err(backend_error());
    }
    let manifest: BackendManifest = serde_json::from_slice(&bytes).map_err(|_| backend_error())?;
    if manifest.schema != CUDA_BACKEND_SCHEMA
        || manifest.backend != CUDA_BACKEND_NAME
        || manifest.icicle_commit != ICICLE_SOURCE_COMMIT
        || manifest.abi != ICICLE_CUDA_ABI
        || manifest.library_inventory.len() != CUDA_BACKEND_LIBRARY_RELATIVE_PATHS.len()
    {
        return Err(backend_error());
    }
    if !manifest_path.starts_with(root) {
        return Err(backend_error());
    }
    Ok((manifest, final_snapshot))
}

fn verify_manifest_snapshot(path: &Path, expected: &FileSnapshot) -> Result<(), eIcicleError> {
    if regular_file_snapshot(path)? != *expected {
        return Err(backend_error());
    }
    Ok(())
}

pub fn load_backend_from_local_bundle() -> Result<(), eIcicleError> {
    // The build script embeds the only accepted baseline root.  Resolve it
    // lexically and lstat every component before any resource is opened.
    let baseline_root = require_directory_components(Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../.."
    )))?;
    let backend_root = require_local_directory(&baseline_root, CUDA_BACKEND_ROOT)?;
    let library_root =
        require_local_directory(&baseline_root, "resources/icicle_cuda_backend/lib/backend")?;
    exact_directory_entries(&backend_root, &["backend.manifest.json", "lib"])?;
    exact_directory_entries(&library_root, &CUDA_BACKEND_LIBRARY_NAMES)?;
    let manifest_path = require_local_file(&baseline_root, CUDA_BACKEND_MANIFEST)?;
    let (manifest, manifest_snapshot) = read_backend_manifest(&baseline_root, &manifest_path)?;

    let mut libraries = Vec::with_capacity(CUDA_BACKEND_LIBRARY_RELATIVE_PATHS.len());
    for (index, declaration) in manifest.library_inventory.iter().enumerate() {
        let expected_path = CUDA_BACKEND_LIBRARY_RELATIVE_PATHS[index];
        if declaration.path != expected_path
            || declaration.size == 0
            || declaration.sha256.len() != "sha256:".len() + 64
            || !declaration.sha256.starts_with("sha256:")
            || declaration.sha256[7..]
                .chars()
                .any(|character| !character.is_ascii_hexdigit() || character.is_ascii_uppercase())
        {
            return Err(backend_error());
        }
        let path = require_local_file(&baseline_root, expected_path)?;
        let snapshot = regular_file_snapshot(&path)?;
        if snapshot.size != declaration.size || sha256_string(snapshot.digest) != declaration.sha256
        {
            return Err(backend_error());
        }
        libraries.push((path, snapshot));
    }

    // Load exactly the four authenticated files.  The C++ API accepts one
    // regular file only; it never scans or recursively loads a directory.
    for (path, before) in &libraries {
        verify_manifest_snapshot(&manifest_path, &manifest_snapshot)?;
        for (other_path, other_before) in &libraries {
            if regular_file_snapshot(other_path)? != *other_before {
                return Err(backend_error());
            }
        }
        let c_path =
            CString::new(path.to_str().ok_or_else(backend_error)?).map_err(|_| backend_error())?;
        unsafe { icicle_load_backend_file_internal(c_path.as_ptr()).wrap()? };
        verify_manifest_snapshot(&manifest_path, &manifest_snapshot)?;
        for (other_path, other_before) in &libraries {
            if regular_file_snapshot(other_path)? != *other_before {
                return Err(backend_error());
            }
        }
    }
    let devices = get_registered_devices()?;
    if !devices.iter().any(|device| device == "CUDA") {
        return Err(backend_error());
    }
    Ok(())
}

pub fn set_device(device: &Device) -> Result<(), eIcicleError> {
    let result = unsafe { icicle_set_device(device) };
    if result == eIcicleError::Success {
        Ok(())
    } else {
        Err(result)
    }
}

pub fn set_default_device(device: &Device) -> Result<(), eIcicleError> {
    let result = unsafe { icicle_set_default_device(device) };
    if result == eIcicleError::Success {
        Ok(())
    } else {
        Err(result)
    }
}

pub fn get_active_device() -> Result<Device, eIcicleError> {
    let mut device: Device = Device::new("invalid", -1);
    unsafe { icicle_get_active_device(&mut device).wrap_value::<Device>(device) }
}

pub fn is_host_memory(ptr: *const c_void) -> bool {
    unsafe { eIcicleError::Success == icicle_is_host_memory(ptr) }
}

pub fn is_active_device_memory(ptr: *const c_void) -> bool {
    unsafe { eIcicleError::Success == icicle_is_active_device_memory(ptr) }
}

pub fn get_device_count() -> Result<i32, eIcicleError> {
    let mut device_count = 0;
    unsafe { icicle_get_device_count(&mut device_count).wrap_value::<i32>(device_count) }
}

pub fn is_device_available(device: &Device) -> bool {
    let err = unsafe { icicle_is_device_available(device) };
    err == eIcicleError::Success
}

pub fn get_available_memory() -> Result<(usize, usize), eIcicleError> {
    let mut total: usize = 0;
    let mut free: usize = 0;
    let result = unsafe { icicle_get_available_memory(&mut total, &mut free) };
    if result == eIcicleError::Success {
        Ok((total, free))
    } else {
        Err(result)
    }
}

pub fn device_synchronize() -> Result<(), eIcicleError> {
    unsafe { icicle_device_synchronize().wrap() }
}

pub fn get_device_properties() -> Result<DeviceProperties, eIcicleError> {
    let mut properties = DeviceProperties {
        using_host_memory: false,
        num_memory_regions: 0,
        supports_pinned_memory: false,
    };
    let result = unsafe { icicle_get_device_properties(&mut properties) };
    if result == eIcicleError::Success {
        Ok(properties)
    } else {
        Err(result)
    }
}

pub fn get_registered_devices() -> Result<Vec<String>, eIcicleError> {
    const BUFFER_SIZE: usize = 256;
    let mut buffer = vec![0 as c_char; BUFFER_SIZE];

    unsafe {
        let result = icicle_get_registered_devices(buffer.as_mut_ptr(), BUFFER_SIZE);
        if result != eIcicleError::Success {
            return Err(result);
        }

        let c_str = CStr::from_ptr(buffer.as_ptr());
        let str_slice: &str = c_str.to_str().unwrap();
        let devices: Vec<String> = str_slice.split(',').map(|s| s.to_string()).collect();
        Ok(devices)
    }
}

// This function pre-allocates default memory pool and warms the GPU up
// so that subsequent memory allocations and other calls are not slowed down
pub fn warmup(stream: &IcicleStream) -> Result<(), eIcicleError> {
    let mut device_ptr: *mut c_void = std::ptr::null_mut();
    let free_memory: usize = 1 << 28;
    unsafe {
        icicle_malloc_async(&mut device_ptr, free_memory >> 1, stream.handle).wrap()?;
        icicle_free_async(device_ptr, stream.handle).wrap()
    }
}

pub fn memset(ptr: *mut c_void, value: i32, size: usize) -> eIcicleError {
    unsafe { icicle_memset(ptr, value, size) }
}

pub fn memset_async(
    ptr: *mut c_void,
    value: i32,
    size: usize,
    stream: *mut c_void,
) -> eIcicleError {
    unsafe { icicle_memset_async(ptr, value, size, stream) }
}
