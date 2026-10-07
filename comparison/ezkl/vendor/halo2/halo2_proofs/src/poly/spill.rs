#![allow(unsafe_code)]

use memmap2::{MmapMut, MmapOptions};
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufReader, BufWriter, Write};
use std::marker::PhantomData;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};

/// Read-only, private backing for a native column. The mapped bytes contain
/// valid initialized `F` objects written in this process; they are temporary
/// prover storage, never a proof or verifier artifact.
pub(super) struct MappedValues<F> {
    mapping: MmapMut,
    len: usize,
    _root: Arc<SpillRoot>,
    _marker: PhantomData<F>,
}

impl<F> std::fmt::Debug for MappedValues<F> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("MappedValues")
            .field("len", &self.len)
            .finish()
    }
}

impl<F> MappedValues<F> {
    pub(super) fn as_slice(&self) -> &[F] {
        // SAFETY: the builder publishes only after writing one valid `F` to
        // every aligned slot and checking the exact mapped length.
        unsafe { std::slice::from_raw_parts(self.mapping.as_ptr().cast::<F>(), self.len) }
    }

    pub(super) fn as_mut_slice(&mut self) -> &mut [F] {
        // SAFETY: callers hold `&mut self`, so no immutable or mutable slice
        // from this mapping can remain live while this exclusive slice exists.
        unsafe { std::slice::from_raw_parts_mut(self.mapping.as_mut_ptr().cast::<F>(), self.len) }
    }
}

// MmapMut exposes writes only through `&mut self`; shared access to a
// MappedValues yields immutable field slices. Polynomial's Rust borrow rules
// therefore prevent simultaneous mutation and sharing across worker threads.
unsafe impl<F: Sync> Sync for MappedValues<F> {}

#[derive(Debug)]
struct SpillRoot {
    path: PathBuf,
    next_file: AtomicU64,
}

impl Drop for SpillRoot {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

impl SpillRoot {
    fn create(base: &Path) -> io::Result<Arc<Self>> {
        static NEXT_ROOT: AtomicU64 = AtomicU64::new(0);
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let mut path = None;
        for _ in 0..32 {
            let sequence = NEXT_ROOT.fetch_add(1, Ordering::Relaxed);
            let candidate = base.join(format!(
                "ezkl-halo2-spill-{}-{}-{}",
                std::process::id(),
                timestamp,
                sequence
            ));
            let mut builder = fs::DirBuilder::new();
            #[cfg(unix)]
            {
                use std::os::unix::fs::DirBuilderExt;
                builder.mode(0o700);
            }
            match builder.create(&candidate) {
                Ok(()) => {
                    path = Some(candidate);
                    break;
                }
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(error),
            }
        }
        let path = path.ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::AlreadyExists,
                "could not reserve private polynomial spill directory",
            )
        })?;
        Ok(Arc::new(Self {
            path,
            next_file: AtomicU64::new(0),
        }))
    }

    fn builder<F: Copy>(self: &Arc<Self>, len: usize) -> io::Result<SpillBuilder<F>> {
        let byte_len = len
            .checked_mul(std::mem::size_of::<F>())
            .filter(|len| *len > 0 && *len <= isize::MAX as usize)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "spill size overflow"))?;
        let mut selected = None;
        for _ in 0..32 {
            let index = self.next_file.fetch_add(1, Ordering::Relaxed);
            let candidate = self.path.join(format!("column-{index:016x}.bin"));
            let mut options = OpenOptions::new();
            options.read(true).write(true).create_new(true);
            #[cfg(unix)]
            {
                use std::os::unix::fs::OpenOptionsExt;
                options.mode(0o600);
            }
            match options.open(&candidate) {
                Ok(file) => {
                    selected = Some((candidate, file));
                    break;
                }
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(error),
            }
        }
        let (path, file) = selected.ok_or_else(|| {
            io::Error::new(io::ErrorKind::AlreadyExists, "could not reserve spill file")
        })?;
        if let Err(error) = reserve_file(&file, byte_len) {
            drop(file);
            let _ = fs::remove_file(&path);
            return Err(error);
        }
        let mapping = match unsafe { MmapOptions::new().len(byte_len).map_mut(&file) } {
            Ok(mapping) => mapping,
            Err(error) => {
                drop(file);
                let _ = fs::remove_file(&path);
                return Err(error);
            }
        };
        let ptr = mapping.as_ptr();
        if (ptr as usize) % std::mem::align_of::<F>() != 0 {
            drop(mapping);
            drop(file);
            let _ = fs::remove_file(&path);
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "spill mapping is not aligned for its field type",
            ));
        }
        drop(file);
        Ok(SpillBuilder {
            mapping: Some(mapping),
            path: Some(path),
            len,
            written: 0,
            root: Arc::clone(self),
            _marker: PhantomData,
        })
    }

    fn append_log(self: &Arc<Self>) -> io::Result<SpillLog> {
        let mut selected = None;
        for _ in 0..32 {
            let index = self.next_file.fetch_add(1, Ordering::Relaxed);
            let path = self.path.join(format!("column-events-{index:016x}.bin"));
            let mut options = OpenOptions::new();
            options.write(true).create_new(true);
            #[cfg(unix)]
            {
                use std::os::unix::fs::OpenOptionsExt;
                options.mode(0o600);
            }
            match options.open(&path) {
                Ok(file) => {
                    selected = Some((path, file));
                    break;
                }
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(error),
            }
        }
        let (path, file) = selected.ok_or_else(|| {
            io::Error::new(io::ErrorKind::AlreadyExists, "could not reserve spill event log")
        })?;
        Ok(SpillLog {
            path,
            writer: Some(BufWriter::new(file)),
            bytes_written: 0,
            reserved_bytes: 0,
            _root: Arc::clone(self),
        })
    }
}

fn reserve_file(file: &File, byte_len: usize) -> io::Result<()> {
    #[cfg(target_os = "linux")]
    {
        use std::os::fd::AsRawFd;
        let length = libc::off_t::try_from(byte_len)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "spill size overflow"))?;
        // Reserve blocks before mapping writable pages so a full filesystem
        // fails here instead of faulting later with SIGBUS.
        let result = unsafe { libc::posix_fallocate(file.as_raw_fd(), 0, length) };
        if result != 0 {
            return Err(io::Error::from_raw_os_error(result));
        }
        Ok(())
    }
    #[cfg(not(target_os = "linux"))]
    {
        file.set_len(byte_len as u64)
    }
}

fn reserve_log_file(file: &File, byte_len: u64) -> io::Result<()> {
    #[cfg(target_os = "linux")]
    {
        use std::os::fd::AsRawFd;
        let length = libc::off_t::try_from(byte_len)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "spill log size overflow"))?;
        let result = unsafe { libc::posix_fallocate(file.as_raw_fd(), 0, length) };
        if result != 0 {
            return Err(io::Error::from_raw_os_error(result));
        }
        Ok(())
    }
    #[cfg(not(target_os = "linux"))]
    {
        file.set_len(byte_len)
    }
}

/// Append-only serialized native witness events. Capacity is reserved in
/// bounded increments before writes, and the file is private scratch owned by
/// its active spill scope.
pub(crate) struct SpillLog {
    path: PathBuf,
    writer: Option<BufWriter<File>>,
    bytes_written: u64,
    reserved_bytes: u64,
    _root: Arc<SpillRoot>,
}

impl SpillLog {
    const RESERVATION_CHUNK_BYTES: u64 = 1 << 20;

    pub(crate) fn append(&mut self, record: &[u8]) -> io::Result<()> {
        let record_len = u64::try_from(record.len())
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "spill log record too large"))?;
        let end = self
            .bytes_written
            .checked_add(record_len)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "spill log size overflow"))?;
        if end > self.reserved_bytes {
            let chunks = end
                .checked_add(Self::RESERVATION_CHUNK_BYTES - 1)
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "spill log size overflow"))?
                / Self::RESERVATION_CHUNK_BYTES;
            let capacity = chunks
                .checked_mul(Self::RESERVATION_CHUNK_BYTES)
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "spill log size overflow"))?;
            let writer = self
                .writer
                .as_ref()
                .expect("spill log writer is present until reader creation");
            reserve_log_file(writer.get_ref(), capacity)?;
            self.reserved_bytes = capacity;
        }
        self.writer
            .as_mut()
            .expect("spill log writer is present until reader creation")
            .write_all(record)?;
        self.bytes_written = end;
        Ok(())
    }

    pub(crate) fn reader(&mut self) -> io::Result<BufReader<File>> {
        if let Some(mut writer) = self.writer.take() {
            writer.flush()?;
            writer.get_ref().set_len(self.bytes_written)?;
        }
        Ok(BufReader::new(File::open(&self.path)?))
    }

    pub(crate) fn bytes_written(&self) -> u64 {
        self.bytes_written
    }
}

impl Drop for SpillLog {
    fn drop(&mut self) {
        self.writer.take();
        let _ = fs::remove_file(&self.path);
    }
}

pub(super) struct SpillBuilder<F> {
    mapping: Option<MmapMut>,
    path: Option<PathBuf>,
    len: usize,
    written: usize,
    root: Arc<SpillRoot>,
    _marker: PhantomData<F>,
}

impl<F> Drop for SpillBuilder<F> {
    fn drop(&mut self) {
        if let Some(path) = self.path.take() {
            let _ = fs::remove_file(path);
        }
    }
}

impl<F: Copy> SpillBuilder<F> {
    pub(super) fn write_value(&mut self, value: F) -> io::Result<()> {
        self.write_chunk(std::slice::from_ref(&value))
    }

    pub(super) fn write_chunk(&mut self, values: &[F]) -> io::Result<()> {
        let end = self
            .written
            .checked_add(values.len())
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "spill offset overflow"))?;
        if end > self.len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "spill write exceeds declared field count",
            ));
        }
        let ptr = self
            .mapping
            .as_mut()
            .expect("spill mapping is present until finish")
            .as_mut_ptr()
            .cast::<F>();
        for (offset, value) in values.iter().copied().enumerate() {
            // SAFETY: the target slot is in-bounds, aligned, and receives a
            // valid initialized value supplied by the native field decoder.
            unsafe { ptr.add(self.written + offset).write(value) };
        }
        self.written = end;
        Ok(())
    }

    pub(super) fn finish(mut self) -> io::Result<MappedValues<F>> {
        if self.written != self.len {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "spill column was not fully initialized",
            ));
        }
        let mapping = self
            .mapping
            .take()
            .expect("spill mapping is present until finish");
        if let Some(path) = self.path.take() {
            fs::remove_file(path)?;
        }
        Ok(MappedValues {
            mapping,
            len: self.len,
            _root: Arc::clone(&self.root),
            _marker: PhantomData,
        })
    }
}

fn active_spill_root() -> io::Result<Arc<SpillRoot>> {
    let active = ACTIVE_SPILL_ROOT.get_or_init(|| Mutex::new(Weak::new()));
    let mut guard = active.lock().map_err(|_| {
        io::Error::new(io::ErrorKind::Other, "polynomial spill directory lock poisoned")
    })?;
    if let Some(root) = guard.upgrade() {
        return Ok(root);
    }
    let root = SpillRoot::create(&std::env::temp_dir())?;
    *guard = Arc::downgrade(&root);
    Ok(root)
}

#[derive(Debug)]
/// Owns a private directory for temporary native polynomial columns.
pub struct NativeSpillScope {
    _root: Arc<SpillRoot>,
}

impl NativeSpillScope {
    pub(super) fn new(base: &Path) -> io::Result<Self> {
        fs::create_dir_all(base)?;
        let metadata = fs::symlink_metadata(base)?;
        if metadata.file_type().is_symlink() || !metadata.is_dir() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "native polynomial spill base must be a non-symlink directory",
            ));
        }
        let active = ACTIVE_SPILL_ROOT.get_or_init(|| Mutex::new(Weak::new()));
        let mut guard = active.lock().map_err(|_| {
            io::Error::new(io::ErrorKind::Other, "polynomial spill directory lock poisoned")
        })?;
        if guard.upgrade().is_some() {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                "another native polynomial spill scope is still active",
            ));
        }
        let root = SpillRoot::create(base)?;
        *guard = Arc::downgrade(&root);
        Ok(Self { _root: root })
    }
}

static ACTIVE_SPILL_ROOT: OnceLock<Mutex<Weak<SpillRoot>>> = OnceLock::new();

pub(super) fn map_values<F: Copy>(values: &[F]) -> io::Result<MappedValues<F>> {
    let root = active_spill_root()?;
    let mut builder = root.builder(values.len())?;
    builder.write_chunk(values)?;
    builder.finish()
}

pub(super) fn spill_builder<F: Copy>(len: usize) -> io::Result<SpillBuilder<F>> {
    active_spill_root()?.builder(len)
}

pub(crate) fn spill_log() -> io::Result<SpillLog> {
    active_spill_root()?.append_log()
}
