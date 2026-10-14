#![allow(unsafe_code)]

use memmap2::{MmapMut, MmapOptions};
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufReader, BufWriter, Write};
use std::marker::PhantomData;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};

const PERMUTATION_PAGE_BYTES: usize = 4096;
const NATIVE_SUCCESSOR_READ_CACHE_PAGE_BYTES: usize = PERMUTATION_PAGE_BYTES;
/// Maximum resident bytes for the fixed native-successor read cache,
/// including page buffers, slot index, and their vector headers.
pub const NATIVE_SUCCESSOR_READ_CACHE_BYTES: usize = 4 * 1024 * 1024;
const PERMUTATION_RECORDS_PER_PAGE: u64 =
    (PERMUTATION_PAGE_BYTES / (2 * std::mem::size_of::<u64>())) as u64;
const PERMUTATION_DATA_CACHE_PAGES: usize = 4096;
const PERMUTATION_DIRECTORY_CACHE_PAGES: usize = 1024;

/// Sparse disk-backed state for the native permutation assembly. Each logical
/// cell stores its non-identity successor and representative/root size in two
/// little-endian u64 words. A zero page-directory entry means the entire page
/// still has the native implicit defaults (identity successor, singleton root).
pub(crate) struct NativePermutationStore {
    directory: SparsePageCache,
    data: SparsePageCache,
    directory_path: PathBuf,
    data_path: PathBuf,
    cell_count: u64,
    max_data_pages: u64,
    max_allocated_bytes: u64,
    next_data_page: u64,
    copy_calls: u64,
    successor_edges: u64,
    nondefault_records: u64,
    _root: Arc<SpillRoot>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct NativePermutationAllocationPlan {
    pub(crate) logical_cells: u64,
    pub(crate) data_file_virtual_bytes: u64,
    pub(crate) directory_file_virtual_bytes: u64,
    pub(crate) data_allocation_bitmap_bytes: u64,
    pub(crate) directory_allocation_bitmap_bytes: u64,
    pub(crate) bounded_cache_bytes: u64,
}

impl std::fmt::Debug for NativePermutationStore {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("NativePermutationStore")
            .field("cell_count", &self.cell_count)
            .field("data_pages_allocated", &self.next_data_page)
            .field("copy_calls", &self.copy_calls)
            .finish()
    }
}

impl NativePermutationStore {
    const REPRESENTATIVE_TAG: u64 = 1 << 63;

    pub(crate) fn allocation_plan(cell_count: usize) -> io::Result<NativePermutationAllocationPlan> {
        let cell_count = u64::try_from(cell_count)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "permutation cell count overflow"))?;
        if cell_count >= Self::REPRESENTATIVE_TAG {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "permutation cell identifiers exceed the checked 63-bit encoding",
            ));
        }
        let data_pages = cell_count
            .checked_add(PERMUTATION_RECORDS_PER_PAGE - 1)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "permutation page count overflow"))?
            / PERMUTATION_RECORDS_PER_PAGE;
        let directory_pages = data_pages
            .checked_mul(std::mem::size_of::<u64>() as u64)
            .and_then(|bytes| bytes.checked_add(PERMUTATION_PAGE_BYTES as u64 - 1))
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "permutation directory size overflow"))?
            / PERMUTATION_PAGE_BYTES as u64;
        let data_file_virtual_bytes = data_pages
            .checked_mul(PERMUTATION_PAGE_BYTES as u64)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "permutation backing size overflow"))?;
        let directory_file_virtual_bytes = directory_pages
            .checked_mul(PERMUTATION_PAGE_BYTES as u64)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "permutation directory backing size overflow"))?;
        let data_allocation_bitmap_bytes = data_pages
            .checked_add(7)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "permutation data bitmap size overflow"))?
            / 8;
        let directory_allocation_bitmap_bytes = directory_pages
            .checked_add(7)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "permutation bitmap size overflow"))?
            / 8;
        let cache_bytes = |page_capacity: usize| -> Option<usize> {
            let index_capacity = page_capacity.checked_mul(2)?.checked_next_power_of_two()?;
            page_capacity
                .checked_mul(PERMUTATION_PAGE_BYTES + std::mem::size_of::<CachedPage>())?
                .checked_add(index_capacity.checked_mul(std::mem::size_of::<CacheSlotIndex>())?)
        };
        let bounded_cache_bytes = cache_bytes(PERMUTATION_DATA_CACHE_PAGES)
            .and_then(|bytes| bytes.checked_add(cache_bytes(PERMUTATION_DIRECTORY_CACHE_PAGES)?))
            .and_then(|bytes| bytes.checked_add(usize::try_from(data_allocation_bitmap_bytes).ok()?))
            .and_then(|bytes| bytes.checked_add(usize::try_from(directory_allocation_bitmap_bytes).ok()?))
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "permutation cache size overflow"))?;
        for byte_len in [data_file_virtual_bytes, directory_file_virtual_bytes] {
            libc::off_t::try_from(byte_len)
                .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "permutation backing exceeds off_t"))?;
        }
        Ok(NativePermutationAllocationPlan {
            logical_cells: cell_count,
            data_file_virtual_bytes,
            directory_file_virtual_bytes,
            data_allocation_bitmap_bytes,
            directory_allocation_bitmap_bytes,
            bounded_cache_bytes: u64::try_from(bounded_cache_bytes)
                .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "permutation cache size overflow"))?,
        })
    }

    pub(crate) fn new(cell_count: usize) -> io::Result<Self> {
        Self::new_with_budget(cell_count, u64::MAX)
    }

    pub(crate) fn new_with_budget(
        cell_count: usize,
        max_allocated_bytes: u64,
    ) -> io::Result<Self> {
        let plan = Self::allocation_plan(cell_count)?;
        let cell_count = plan.logical_cells;
        let max_data_pages = plan.data_file_virtual_bytes / PERMUTATION_PAGE_BYTES as u64;
        let directory_pages = plan.directory_file_virtual_bytes / PERMUTATION_PAGE_BYTES as u64;
        let root = active_spill_root()?;
        let (directory_path, directory_file) = root.create_backing_file("permutation-directory")?;
        let (data_path, data_file) = match root.create_backing_file("permutation-data") {
            Ok(pair) => pair,
            Err(error) => {
                let _ = fs::remove_file(&directory_path);
                return Err(error);
            }
        };
        let directory = match SparsePageCache::new(
            directory_file,
            directory_pages,
            PERMUTATION_DIRECTORY_CACHE_PAGES,
            true,
        ) {
            Ok(cache) => cache,
            Err(error) => {
                let _ = fs::remove_file(&directory_path);
                let _ = fs::remove_file(&data_path);
                return Err(error);
            }
        };
        let data = match SparsePageCache::new(
            data_file,
            max_data_pages,
            PERMUTATION_DATA_CACHE_PAGES,
            true,
        ) {
            Ok(cache) => cache,
            Err(error) => {
                let _ = fs::remove_file(&directory_path);
                let _ = fs::remove_file(&data_path);
                return Err(error);
            }
        };
        Ok(Self {
            directory,
            data,
            directory_path,
            data_path,
            cell_count,
            max_data_pages,
            max_allocated_bytes,
            next_data_page: 0,
            copy_calls: 0,
            successor_edges: 0,
            nondefault_records: 0,
            _root: root,
        })
    }

    fn checked_cell(&self, cell: usize) -> io::Result<u64> {
        let cell = u64::try_from(cell)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "permutation cell identifier overflow"))?;
        if cell >= self.cell_count || cell >= Self::REPRESENTATIVE_TAG {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "permutation cell identifier is outside the declared shape",
            ));
        }
        Ok(cell)
    }

    fn directory_page_id(&mut self, data_page: u64) -> io::Result<Option<u64>> {
        if data_page >= self.max_data_pages {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "permutation page is outside the declared shape"));
        }
        let byte_offset = data_page
            .checked_mul(8)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "permutation directory offset overflow"))?;
        let encoded = self.directory.read_u64(byte_offset)?;
        if encoded == 0 {
            Ok(None)
        } else {
            let page_id = encoded - 1;
            if page_id >= self.next_data_page {
                return Err(io::Error::new(io::ErrorKind::InvalidData, "permutation directory references an unallocated page"));
            }
            Ok(Some(page_id))
        }
    }

    fn ensure_data_page(&mut self, logical_page: u64) -> io::Result<u64> {
        if let Some(physical_page) = self.directory_page_id(logical_page)? {
            return Ok(physical_page);
        }
        if self.next_data_page >= self.max_data_pages {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "permutation data page capacity exhausted"));
        }
        let physical_page = self.next_data_page;
        let encoded_page = physical_page
            .checked_add(1)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "permutation page encoding overflow"))?;
        let directory_offset = logical_page
            .checked_mul(8)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "permutation directory offset overflow"))?;
        let directory_page = directory_offset / PERMUTATION_PAGE_BYTES as u64;
        let data_page_cost = if self.data.is_allocated(physical_page)? {
            0
        } else {
            PERMUTATION_PAGE_BYTES as u64
        };
        let directory_page_cost = if self.directory.is_allocated(directory_page)? {
            0
        } else {
            PERMUTATION_PAGE_BYTES as u64
        };
        let current_bytes = self
            .data
            .allocated_pages()
            .checked_add(self.directory.allocated_pages())
            .and_then(|pages| pages.checked_mul(PERMUTATION_PAGE_BYTES as u64))
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "permutation allocated-byte count overflow"))?;
        let projected_bytes = current_bytes
            .checked_add(data_page_cost)
            .and_then(|bytes| bytes.checked_add(directory_page_cost))
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "permutation allocation budget overflow"))?;
        if projected_bytes > self.max_allocated_bytes {
            return Err(io::Error::new(
                io::ErrorKind::OutOfMemory,
                format!(
                    "native permutation backing budget exceeded: need {projected_bytes} bytes, limit {}",
                    self.max_allocated_bytes
                ),
            ));
        }
        self.data.reserve_page(physical_page)?;
        self.next_data_page = self
            .next_data_page
            .checked_add(1)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "permutation page counter overflow"))?;
        self.directory.write_u64(directory_offset, encoded_page)?;
        Ok(physical_page)
    }

    fn allocated_backing_bytes(&self) -> io::Result<u64> {
        self.data
            .allocated_pages()
            .checked_add(self.directory.allocated_pages())
            .and_then(|pages| pages.checked_mul(PERMUTATION_PAGE_BYTES as u64))
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "permutation allocated-byte count overflow",
                )
            })
    }

    fn word_offset(cell_in_page: u64, word: u64) -> io::Result<u64> {
        cell_in_page
            .checked_mul(16)
            .and_then(|offset| offset.checked_add(word * 8))
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "permutation record offset overflow"))
    }

    fn read_record(&mut self, cell: u64) -> io::Result<(u64, u64)> {
        if cell >= self.cell_count {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "permutation cell is outside the declared shape"));
        }
        let logical_page = cell / PERMUTATION_RECORDS_PER_PAGE;
        let in_page = cell % PERMUTATION_RECORDS_PER_PAGE;
        let Some(physical_page) = self.directory_page_id(logical_page)? else {
            return Ok((0, 0));
        };
        let offset = Self::word_offset(in_page, 0)?;
        let base = physical_page
            .checked_mul(PERMUTATION_PAGE_BYTES as u64)
            .and_then(|base| base.checked_add(offset))
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "permutation record offset overflow"))?;
        let second_word = base
            .checked_add(8)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "permutation record offset overflow"))?;
        let first = self.data.read_u64(base)?;
        let second = self.data.read_u64(second_word)?;
        Ok((first, second))
    }

    fn write_record(&mut self, cell: u64, first: u64, second: u64) -> io::Result<()> {
        if cell >= self.cell_count {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "permutation cell is outside the declared shape"));
        }
        let logical_page = cell / PERMUTATION_RECORDS_PER_PAGE;
        let in_page = cell % PERMUTATION_RECORDS_PER_PAGE;
        let physical_page = self.ensure_data_page(logical_page)?;
        let base = physical_page
            .checked_mul(PERMUTATION_PAGE_BYTES as u64)
            .and_then(|offset| offset.checked_add(Self::word_offset(in_page, 0).ok()?))
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "permutation record offset overflow"))?;
        let second_word = base
            .checked_add(8)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "permutation record offset overflow"))?;
        let (old_first, old_second) = self.read_record(cell)?;
        self.data.write_u64(base, first)?;
        self.data.write_u64(second_word, second)?;
        let was_default = old_first == 0 && old_second == 0;
        let is_default = first == 0 && second == 0;
        if was_default && !is_default {
            self.nondefault_records = self.nondefault_records.checked_add(1).ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "permutation record count overflow")
            })?;
        } else if !was_default && is_default {
            self.nondefault_records = self.nondefault_records.checked_sub(1).ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "permutation record count underflow")
            })?;
        }
        Ok(())
    }

    pub(crate) fn mapping_at(&mut self, cell: usize) -> io::Result<usize> {
        let cell = self.checked_cell(cell)?;
        let (encoded, _) = self.read_record(cell)?;
        let target = if encoded == 0 { cell } else { encoded - 1 };
        if target >= self.cell_count {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "permutation successor is outside the declared shape"));
        }
        usize::try_from(target)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "permutation successor does not fit usize"))
    }

    pub(crate) fn set_mapping(&mut self, cell: usize, successor: usize) -> io::Result<()> {
        let cell = self.checked_cell(cell)?;
        let successor = self.checked_cell(successor)?;
        let (old_successor, representative) = self.read_record(cell)?;
        let encoded_successor = if cell == successor {
            0
        } else {
            (successor as u64)
                .checked_add(1)
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "permutation successor encoding overflow"))?
        };
        self.write_record(cell, encoded_successor, representative)?;
        if old_successor == 0 && encoded_successor != 0 {
            self.successor_edges = self.successor_edges.checked_add(1).ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "permutation successor count overflow")
            })?;
        } else if old_successor != 0 && encoded_successor == 0 {
            self.successor_edges = self.successor_edges.checked_sub(1).ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "permutation successor count underflow")
            })?;
        }
        Ok(())
    }

    pub(crate) fn representative(&mut self, cell: usize) -> io::Result<usize> {
        let cell = self.checked_cell(cell)?;
        let (_, encoded) = self.read_record(cell)?;
        let representative = if encoded == 0 || encoded & Self::REPRESENTATIVE_TAG != 0 {
            cell
        } else {
            encoded - 1
        };
        if representative >= self.cell_count {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "permutation representative is outside the declared shape"));
        }
        usize::try_from(representative)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "permutation representative does not fit usize"))
    }

    pub(crate) fn cycle_size(&mut self, representative: usize) -> io::Result<usize> {
        let representative = self.checked_cell(representative)?;
        let (_, encoded) = self.read_record(representative)?;
        let size = if encoded == 0 {
            1
        } else if encoded & Self::REPRESENTATIVE_TAG != 0 {
            encoded & !Self::REPRESENTATIVE_TAG
        } else {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "permutation cycle-size query is not at its root"));
        };
        if size == 0 || size > self.cell_count {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "permutation cycle size is invalid"));
        }
        usize::try_from(size)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "permutation cycle size does not fit usize"))
    }

    pub(crate) fn set_representative(&mut self, cell: usize, representative: usize) -> io::Result<()> {
        let cell = self.checked_cell(cell)?;
        let representative = self.checked_cell(representative)?;
        let (successor, prior) = self.read_record(cell)?;
        let encoded = if cell == representative {
            if prior & Self::REPRESENTATIVE_TAG != 0 { prior } else { 0 }
        } else {
            representative + 1
        };
        self.write_record(cell, successor, encoded)
    }

    pub(crate) fn set_cycle_size(&mut self, representative: usize, size: usize) -> io::Result<()> {
        let representative = self.checked_cell(representative)?;
        let size = u64::try_from(size)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "permutation cycle size overflow"))?;
        if size == 0 || size > self.cell_count || size >= Self::REPRESENTATIVE_TAG {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "permutation cycle size is outside the declared shape"));
        }
        let (successor, _) = self.read_record(representative)?;
        self.write_record(representative, successor, Self::REPRESENTATIVE_TAG | size)
    }

    pub(crate) fn clear_cycle_size(&mut self, representative: usize) -> io::Result<()> {
        let representative = self.checked_cell(representative)?;
        let (successor, encoded) = self.read_record(representative)?;
        if encoded & Self::REPRESENTATIVE_TAG == 0 && encoded != 0 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "permutation size clear is not at a root"));
        }
        self.write_record(representative, successor, 0)
    }

    pub(crate) fn mapping_range(
        &mut self,
        column_start: usize,
        row_start: usize,
        out: &mut [(usize, usize)],
        domain_size: usize,
    ) -> io::Result<()> {
        let domain_size_u64 = u64::try_from(domain_size)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "permutation domain size overflow"))?;
        if domain_size_u64 == 0
            || row_start
                .checked_add(out.len())
                .map(|end| end > domain_size)
                .unwrap_or(true)
        {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "permutation mapping range is outside the declared domain"));
        }
        let column_start = u64::try_from(column_start)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "permutation column offset overflow"))?
            .checked_mul(domain_size_u64)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "permutation column offset overflow"))?;
        let mut output_start = 0usize;
        while output_start < out.len() {
            let cell = column_start
                .checked_add(row_start as u64)
                .and_then(|base| base.checked_add(output_start as u64))
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "permutation cell offset overflow"))?;
            let page_offset = (cell % PERMUTATION_RECORDS_PER_PAGE) as usize;
            let count = (out.len() - output_start).min(
                (PERMUTATION_RECORDS_PER_PAGE as usize) - page_offset,
            );
            let logical_page = cell / PERMUTATION_RECORDS_PER_PAGE;
            let dir_offset = logical_page
                .checked_mul(8)
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "permutation directory offset overflow"))?;
            let encoded_page = self.directory.read_u64(dir_offset)?;
            if encoded_page == 0 {
                for (offset, result) in out[output_start..output_start + count]
                    .iter_mut()
                    .enumerate()
                {
                    let source = usize::try_from(cell + offset as u64).map_err(|_| {
                        io::Error::new(io::ErrorKind::InvalidInput, "permutation cell offset does not fit usize")
                    })?;
                    *result = (source / domain_size, source % domain_size);
                }
            } else {
                let physical_page = encoded_page - 1;
                if physical_page >= self.next_data_page {
                    return Err(io::Error::new(io::ErrorKind::InvalidData, "permutation directory references an unallocated page"));
                }
                let mut successors = [0u64; PERMUTATION_RECORDS_PER_PAGE as usize];
                self.data.read_first_words(physical_page, 0, &mut successors)?;
                for (offset, result) in out[output_start..output_start + count]
                    .iter_mut()
                    .enumerate()
                {
                    let source = cell + offset as u64;
                    let encoded = successors[page_offset + offset];
                    let target = if encoded == 0 { source } else { encoded - 1 };
                    if target >= self.cell_count {
                        return Err(io::Error::new(io::ErrorKind::InvalidData, "permutation successor is outside the declared shape"));
                    }
                    let target = usize::try_from(target).map_err(|_| {
                        io::Error::new(io::ErrorKind::InvalidData, "permutation successor does not fit usize")
                    })?;
                    *result = (target / domain_size, target % domain_size);
                }
            }
            output_start += count;
        }
        Ok(())
    }

    pub(crate) fn for_each_successor(
        &mut self,
        mut visit: impl FnMut(u64, u64) -> io::Result<()>,
    ) -> io::Result<usize> {
        let mut count = 0usize;
        let next_data_page = self.next_data_page;
        let cell_count = self.cell_count;
        let data = &mut self.data;
        self.directory
            .for_each_nonzero_u64(self.max_data_pages, |logical_page, page_id_encoded| {
                let page_id = page_id_encoded - 1;
                if page_id >= next_data_page {
                    return Err(io::Error::new(io::ErrorKind::InvalidData, "permutation directory references an unallocated page"));
                }
                let mut encoded_successors = [0u64; PERMUTATION_RECORDS_PER_PAGE as usize];
                data.read_first_words(page_id, 0, &mut encoded_successors)?;
                for (slot, encoded) in encoded_successors.into_iter().enumerate() {
                    let source = logical_page
                        .checked_mul(PERMUTATION_RECORDS_PER_PAGE)
                        .and_then(|base| base.checked_add(slot as u64))
                        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "permutation source cell overflow"))?;
                    if source >= cell_count {
                        break;
                    }
                    if encoded != 0 {
                        visit(source, encoded - 1)?;
                        count = count.checked_add(1).ok_or_else(|| {
                            io::Error::new(io::ErrorKind::InvalidData, "permutation successor count overflow")
                        })?;
                    }
                }
                Ok(())
            })?;
        Ok(count)
    }

    pub(crate) fn copy_call(&mut self) -> io::Result<()> {
        self.copy_calls = self
            .copy_calls
            .checked_add(1)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "permutation copy call count overflow"))?;
        Ok(())
    }

    /// Convert the native successor map into a sorted, disk-backed pair list.
    /// The scan follows logical cell order, so serialized pair order remains
    /// byte-for-byte compatible with the existing sorted Vec format.
    pub(crate) fn into_successors(mut self) -> io::Result<NativeSuccessorStore> {
        self.flush()?;
        let expected = self.successor_edges;
        let current_bytes = self.allocated_backing_bytes()?;
        let remaining_budget = self
            .max_allocated_bytes
            .checked_sub(current_bytes)
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "permutation backing already exceeds its configured budget",
                )
            })?;
        let mut successors = NativeSuccessorStore::new_with_budget(expected, remaining_budget)?;
        const PAIRS_PER_WRITE: usize = 4096;
        let mut bytes = Vec::new();
        bytes.try_reserve_exact(PAIRS_PER_WRITE * 16).map_err(|error| {
            io::Error::new(io::ErrorKind::OutOfMemory, format!("cannot reserve bounded permutation output buffer: {error}"))
        })?;
        let mut emitted = 0u64;
        self.for_each_successor(|source, target| {
            bytes.extend_from_slice(&source.to_le_bytes());
            bytes.extend_from_slice(&target.to_le_bytes());
            emitted = emitted.checked_add(1).ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "permutation successor count overflow")
            })?;
            if bytes.len() == PAIRS_PER_WRITE * 16 {
                let start = emitted
                    .checked_sub(PAIRS_PER_WRITE as u64)
                    .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "permutation output offset underflow"))?;
                successors.write_bytes(start, &bytes)?;
                bytes.clear();
            }
            Ok(())
        })?;
        if emitted != expected {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "native permutation successor counter disagrees with the sorted scan",
            ));
        }
        if !bytes.is_empty() {
            let start = emitted
                .checked_sub((bytes.len() / 16) as u64)
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "permutation output offset underflow"))?;
            successors.write_bytes(start, &bytes)?;
        }
        successors.flush()?;
        Ok(successors)
    }

    pub(crate) fn stats(&self) -> io::Result<NativePermutationStats> {
        let data_page_bytes = self.next_data_page.checked_mul(PERMUTATION_PAGE_BYTES as u64).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "permutation data backing byte count overflow")
        })?;
        let directory_pages = self.directory.allocated_pages();
        let directory_page_bytes = directory_pages.checked_mul(PERMUTATION_PAGE_BYTES as u64).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "permutation directory backing byte count overflow")
        })?;
        let cache_bytes = u64::try_from(self.data.cache_bytes())
            .ok()
            .and_then(|bytes| u64::try_from(self.directory.cache_bytes()).ok().and_then(|other| bytes.checked_add(other)))
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "permutation cache byte count overflow"))?;
        let allocation_bitmap_bytes = u64::try_from(self.data.allocation_bitmap_bytes())
            .ok()
            .and_then(|bytes| u64::try_from(self.directory.allocation_bitmap_bytes()).ok().and_then(|other| bytes.checked_add(other)))
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "permutation allocation bitmap byte count overflow"))?;
        Ok(NativePermutationStats {
            logical_cells: self.cell_count,
            copy_calls: self.copy_calls,
            successor_edges: self.successor_edges,
            nondefault_records: self.nondefault_records,
            data_pages_allocated: self.next_data_page,
            directory_pages_allocated: directory_pages,
            data_page_bytes,
            directory_page_bytes,
            cache_bytes,
            allocation_bitmap_bytes,
        })
    }

    pub(crate) fn flush(&mut self) -> io::Result<()> {
        self.data.flush()?;
        self.directory.flush()
    }
}

impl Drop for NativePermutationStore {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.directory_path);
        let _ = fs::remove_file(&self.data_path);
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct NativePermutationStats {
    pub(crate) logical_cells: u64,
    pub(crate) copy_calls: u64,
    pub(crate) successor_edges: u64,
    pub(crate) nondefault_records: u64,
    pub(crate) data_pages_allocated: u64,
    pub(crate) directory_pages_allocated: u64,
    pub(crate) data_page_bytes: u64,
    pub(crate) directory_page_bytes: u64,
    pub(crate) cache_bytes: u64,
    pub(crate) allocation_bitmap_bytes: u64,
}

#[derive(Clone)]
pub(crate) struct NativeSuccessorStore {
    backing: Arc<NativeSuccessorBacking>,
    len: u64,
    reserved_bytes: u64,
}

#[derive(Clone, Copy)]
enum NativeSuccessorEncoding {
    RawPairs,
    PackedIndices { bits_per_index: u8 },
}

struct NativeSuccessorBacking {
    // A file-backed PK view shares the authenticated PK file and must never
    // unlink it. Temporary keygen stores own their path and root instead.
    path: Option<PathBuf>,
    file: Arc<File>,
    file_offset: u64,
    encoding: NativeSuccessorEncoding,
    read_cache: Mutex<Option<NativeSuccessorReadCache>>,
    _root: Option<Arc<SpillRoot>>,
}

struct NativeSuccessorReadPage {
    page_index: Option<u64>,
    bytes: [u8; NATIVE_SUCCESSOR_READ_CACHE_PAGE_BYTES],
    referenced: bool,
}

struct NativeSuccessorReadCache {
    pages: Vec<NativeSuccessorReadPage>,
    slots: Vec<CacheSlotIndex>,
    clock_hand: usize,
}

// 1010 inline 4 KiB pages plus a 2048-entry index fit within the public
// 4 MiB total cache budget, including fixed metadata on 64-bit targets.
const NATIVE_SUCCESSOR_READ_CACHE_PAGES: usize = 1010;

impl NativeSuccessorReadCache {
    fn new(page_capacity: usize) -> io::Result<Self> {
        let mut pages = Vec::new();
        pages.try_reserve_exact(page_capacity).map_err(|error| {
            io::Error::new(io::ErrorKind::OutOfMemory, format!("cannot reserve native successor read cache: {error}"))
        })?;
        pages.resize_with(page_capacity, || NativeSuccessorReadPage {
            page_index: None,
            bytes: [0; NATIVE_SUCCESSOR_READ_CACHE_PAGE_BYTES],
            referenced: false,
        });
        let slot_count = page_capacity
            .checked_mul(2)
            .and_then(usize::checked_next_power_of_two)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "native successor cache index size overflow"))?;
        let mut slots = Vec::new();
        slots.try_reserve_exact(slot_count).map_err(|error| {
            io::Error::new(io::ErrorKind::OutOfMemory, format!("cannot reserve native successor cache index: {error}"))
        })?;
        slots.resize_with(slot_count, CacheSlotIndex::empty);
        let cache = Self { pages, slots, clock_hand: 0 };
        if cache.allocated_bytes() > NATIVE_SUCCESSOR_READ_CACHE_BYTES {
            return Err(io::Error::new(io::ErrorKind::OutOfMemory, "native successor read cache exceeds its fixed byte budget"));
        }
        Ok(cache)
    }

    fn allocated_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            .saturating_add(self.pages.capacity().saturating_mul(std::mem::size_of::<NativeSuccessorReadPage>()))
            .saturating_add(self.slots.capacity().saturating_mul(std::mem::size_of::<CacheSlotIndex>()))
    }

    fn pair_at(
        &mut self,
        file: &File,
        file_offset: u64,
        file_len: u64,
        encoding: NativeSuccessorEncoding,
        len: u64,
        index: u64,
    ) -> io::Result<(u64, u64)> {
        if index >= len {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "successor index is outside the declared pair count"));
        }
        match encoding {
            NativeSuccessorEncoding::RawPairs => {
                let offset = index
                    .checked_mul(16)
                    .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "successor read offset overflow"))?;
                let mut raw = [0u8; 16];
                self.read_cached_bytes(file, file_offset, file_len, offset, &mut raw)?;
                Ok((
                    u64::from_le_bytes(raw[..8].try_into().unwrap()),
                    u64::from_le_bytes(raw[8..].try_into().unwrap()),
                ))
            }
            NativeSuccessorEncoding::PackedIndices { bits_per_index } => {
                let bits = u64::from(bits_per_index);
                if bits == 0 || bits > 64 {
                    return Err(io::Error::new(io::ErrorKind::InvalidData, "packed successor index width is invalid"));
                }
                let first = index
                    .checked_mul(2)
                    .and_then(|value| value.checked_mul(bits))
                    .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "packed successor bit offset overflow"))?;
                let second = first
                    .checked_add(bits)
                    .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "packed successor bit offset overflow"))?;
                Ok((
                    self.read_packed_index(file, file_offset, file_len, first, bits_per_index)?,
                    self.read_packed_index(file, file_offset, file_len, second, bits_per_index)?,
                ))
            }
        }
    }

    fn read_packed_index(
        &mut self,
        file: &File,
        file_offset: u64,
        file_len: u64,
        bit_offset: u64,
        bits_per_index: u8,
    ) -> io::Result<u64> {
        let shift = (bit_offset % 8) as usize;
        let byte_offset = bit_offset / 8;
        let bits = usize::from(bits_per_index);
        let byte_count = (shift + bits + 7) / 8;
        if byte_count > 9 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "packed successor index exceeds the bounded decoder"));
        }
        let mut encoded = [0u8; 9];
        self.read_cached_bytes(file, file_offset, file_len, byte_offset, &mut encoded[..byte_count])?;
        let mut value = 0u128;
        for (byte_index, byte) in encoded[..byte_count].iter().enumerate() {
            value |= u128::from(*byte) << (byte_index * 8);
        }
        let mask = if bits == 64 { u64::MAX as u128 } else { (1u128 << bits) - 1 };
        Ok(((value >> shift) & mask) as u64)
    }

    fn read_cached_bytes(
        &mut self,
        file: &File,
        file_offset: u64,
        file_len: u64,
        mut offset: u64,
        mut out: &mut [u8],
    ) -> io::Result<()> {
        if offset.checked_add(out.len() as u64).map(|end| end > file_len).unwrap_or(true) {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "successor read exceeds backing view"));
        }
        while !out.is_empty() {
            let page_index = offset / NATIVE_SUCCESSOR_READ_CACHE_PAGE_BYTES as u64;
            let page_offset = usize::try_from(offset % NATIVE_SUCCESSOR_READ_CACHE_PAGE_BYTES as u64)
                .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "successor page offset overflow"))?;
            let slot = self.page_slot(file, file_offset, page_index, file_len)?;
            let count = out.len().min(NATIVE_SUCCESSOR_READ_CACHE_PAGE_BYTES - page_offset);
            out[..count].copy_from_slice(&self.pages[slot].bytes[page_offset..page_offset + count]);
            offset = offset
                .checked_add(count as u64)
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "successor read offset overflow"))?;
            out = &mut out[count..];
        }
        Ok(())
    }

    fn lower_bound_source(
        &mut self,
        file: &File,
        file_offset: u64,
        file_len: u64,
        encoding: NativeSuccessorEncoding,
        len: u64,
        source: u64,
    ) -> io::Result<u64> {
        let mut low = 0u64;
        let mut high = len;
        while low < high {
            let middle = low + (high - low) / 2;
            if self.pair_at(file, file_offset, file_len, encoding, len, middle)?.0 < source {
                low = middle + 1;
            } else {
                high = middle;
            }
        }
        Ok(low)
    }

    fn source_ordinal(
        &mut self,
        file: &File,
        file_offset: u64,
        file_len: u64,
        encoding: NativeSuccessorEncoding,
        len: u64,
        source: u64,
    ) -> io::Result<Option<u64>> {
        let ordinal = self.lower_bound_source(file, file_offset, file_len, encoding, len, source)?;
        if ordinal == len {
            return Ok(None);
        }
        Ok((self.pair_at(file, file_offset, file_len, encoding, len, ordinal)?.0 == source).then_some(ordinal))
    }

    fn page_slot(&mut self, file: &File, file_offset: u64, page_index: u64, file_len: u64) -> io::Result<usize> {
        if let Some(slot) = self.lookup_slot(page_index) {
            self.pages[slot].referenced = true;
            return Ok(slot);
        }
        if self.pages.is_empty() {
            return Err(io::Error::new(io::ErrorKind::OutOfMemory, "native successor read cache has zero pages"));
        }
        let slot = self.choose_victim();
        if let Some(old_page) = self.pages[slot].page_index {
            self.remove_slot(old_page);
        }
        let relative_offset = page_index
            .checked_mul(NATIVE_SUCCESSOR_READ_CACHE_PAGE_BYTES as u64)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "successor page offset overflow"))?;
        let offset = file_offset.checked_add(relative_offset)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "successor file offset overflow"))?;
        libc::off_t::try_from(offset)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "successor page exceeds off_t"))?;
        if relative_offset >= file_len {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "successor page starts beyond backing file"));
        }
        let count = usize::try_from((file_len - relative_offset).min(NATIVE_SUCCESSOR_READ_CACHE_PAGE_BYTES as u64))
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "successor page length overflow"))?;
        let page = &mut self.pages[slot];
        page.bytes.fill(0);
        page.page_index = None;
        read_exact_at(file, &mut page.bytes[..count], offset)?;
        page.page_index = Some(page_index);
        page.referenced = true;
        self.insert_slot(page_index, slot)?;
        Ok(slot)
    }

    fn lookup_slot(&self, page_index: u64) -> Option<usize> {
        if self.slots.is_empty() {
            return None;
        }
        let key = page_index.checked_add(1)?;
        let mask = self.slots.len() - 1;
        let mut index = SparsePageCache::hash_page(page_index) & mask;
        for _ in 0..self.slots.len() {
            let entry = self.slots[index];
            if entry.key == CacheSlotIndex::EMPTY {
                return None;
            }
            if entry.key == key {
                return Some(entry.slot);
            }
            index = (index + 1) & mask;
        }
        None
    }

    fn insert_slot(&mut self, page_index: u64, slot: usize) -> io::Result<()> {
        let key = page_index
            .checked_add(1)
            .filter(|key| *key != CacheSlotIndex::TOMBSTONE)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "native successor cache key overflow"))?;
        let mask = self.slots.len() - 1;
        let mut index = SparsePageCache::hash_page(page_index) & mask;
        let mut tombstone = None;
        for _ in 0..self.slots.len() {
            let entry = self.slots[index];
            if entry.key == CacheSlotIndex::EMPTY {
                let target = tombstone.unwrap_or(index);
                self.slots[target] = CacheSlotIndex { key, slot };
                return Ok(());
            }
            if entry.key == CacheSlotIndex::TOMBSTONE && tombstone.is_none() {
                tombstone = Some(index);
            } else if entry.key == key {
                return Err(io::Error::new(io::ErrorKind::InvalidData, "duplicate native successor cache page"));
            }
            index = (index + 1) & mask;
        }
        if let Some(target) = tombstone {
            self.slots[target] = CacheSlotIndex { key, slot };
            Ok(())
        } else {
            Err(io::Error::new(io::ErrorKind::OutOfMemory, "native successor cache index is full"))
        }
    }

    fn remove_slot(&mut self, page_index: u64) {
        let Some(key) = page_index.checked_add(1) else { return };
        let mask = self.slots.len() - 1;
        let mut index = SparsePageCache::hash_page(page_index) & mask;
        for _ in 0..self.slots.len() {
            let entry = self.slots[index];
            if entry.key == CacheSlotIndex::EMPTY {
                return;
            }
            if entry.key == key {
                self.slots[index].key = CacheSlotIndex::TOMBSTONE;
                return;
            }
            index = (index + 1) & mask;
        }
    }

    fn choose_victim(&mut self) -> usize {
        for (index, page) in self.pages.iter().enumerate() {
            if page.page_index.is_none() {
                return index;
            }
        }
        loop {
            let index = self.clock_hand;
            self.clock_hand = (self.clock_hand + 1) % self.pages.len();
            if self.pages[index].referenced {
                self.pages[index].referenced = false;
            } else {
                return index;
            }
        }
    }
}

impl std::fmt::Debug for NativeSuccessorStore {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("NativeSuccessorStore")
            .field("len", &self.len)
            .finish()
    }
}

impl Drop for NativeSuccessorBacking {
    fn drop(&mut self) {
        if let Some(path) = &self.path {
            let _ = fs::remove_file(path);
        }
    }
}

impl NativeSuccessorStore {
    pub(crate) fn new(len: u64) -> io::Result<Self> {
        Self::new_with_budget(len, u64::MAX)
    }

    pub(crate) fn new_with_budget(len: u64, max_bytes: u64) -> io::Result<Self> {
        let byte_len = len
            .checked_mul(16)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "successor backing size overflow"))?;
        let reserved_bytes = byte_len
            .checked_add(PERMUTATION_PAGE_BYTES as u64 - 1)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "successor allocation extent overflow"))?
            / PERMUTATION_PAGE_BYTES as u64
            * PERMUTATION_PAGE_BYTES as u64;
        if reserved_bytes > max_bytes {
            return Err(io::Error::new(
                io::ErrorKind::OutOfMemory,
                format!(
                    "native successor backing budget exceeded: need {reserved_bytes} bytes, limit {max_bytes}"
                ),
            ));
        }
        let off_t_len = libc::off_t::try_from(byte_len)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "successor backing exceeds off_t"))?;
        let root = active_spill_root()?;
        let (path, file) = root.create_backing_file("permutation-successors")?;
        let initialized = (|| {
            file.set_len(off_t_len as u64)?;
            if byte_len > 0 {
                reserve_file_range(&file, 0, byte_len)?;
            }
            file.sync_data()
        })();
        if let Err(error) = initialized {
            drop(file);
            let _ = fs::remove_file(&path);
            return Err(error);
        }
        Ok(Self {
            backing: Arc::new(NativeSuccessorBacking {
                path: Some(path),
                file: Arc::new(file),
                file_offset: 0,
                encoding: NativeSuccessorEncoding::RawPairs,
                read_cache: Mutex::new(None),
                _root: Some(root),
            }),
            len,
            reserved_bytes,
        })
    }

    /// Creates a read-only view over packed successor indices in an
    /// authenticated proving-key file. The view borrows the PK's file and
    /// therefore never reserves temporary successor bytes or unlinks it.
    pub(crate) fn packed_file_view(
        file: Arc<File>,
        file_offset: u64,
        len: u64,
        bits_per_index: u8,
    ) -> io::Result<Self> {
        if bits_per_index > 64 || (len > 0 && bits_per_index == 0) {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "packed successor index width is invalid"));
        }
        let packed_bits = len
            .checked_mul(2)
            .and_then(|count| count.checked_mul(u64::from(bits_per_index)))
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "packed successor payload size overflow"))?;
        let packed_bytes = packed_bits
            .checked_add(7)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "packed successor payload size overflow"))?
            / 8;
        let file_end = file_offset
            .checked_add(packed_bytes)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "packed successor file range overflow"))?;
        if file_end > file.metadata()?.len() {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "packed successor view exceeds proving-key file"));
        }
        libc::off_t::try_from(file_end)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "packed successor file range exceeds off_t"))?;
        Ok(Self {
            backing: Arc::new(NativeSuccessorBacking {
                path: None,
                file,
                file_offset,
                encoding: NativeSuccessorEncoding::PackedIndices { bits_per_index },
                read_cache: Mutex::new(None),
                _root: None,
            }),
            len,
            reserved_bytes: 0,
        })
    }

    pub(crate) fn len(&self) -> u64 {
        self.len
    }

    pub(crate) fn write_bytes(&mut self, start: u64, bytes: &[u8]) -> io::Result<()> {
        if !matches!(self.backing.encoding, NativeSuccessorEncoding::RawPairs) || self.backing.path.is_none() {
            return Err(io::Error::new(io::ErrorKind::PermissionDenied, "successor view is read-only"));
        }
        if bytes.len() % 16 != 0 {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "successor write is not pair-aligned"));
        }
        let count = u64::try_from(bytes.len() / 16)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "successor write length overflow"))?;
        let end = start
            .checked_add(count)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "successor write range overflow"))?;
        if end > self.len {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "successor write exceeds declared pair count"));
        }
        let offset = start
            .checked_mul(16)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "successor write offset overflow"))?;
        write_all_at(&self.backing.file, bytes, offset)
    }

    pub(crate) fn read_pairs(&self, start: u64, out: &mut [(u64, u64)]) -> io::Result<()> {
        let count = u64::try_from(out.len())
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "successor read length overflow"))?;
        if start
            .checked_add(count)
            .map(|end| end > self.len)
            .unwrap_or(true)
        {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "successor read exceeds declared pair count"));
        }
        if matches!(self.backing.encoding, NativeSuccessorEncoding::PackedIndices { .. }) {
            for (offset, pair) in out.iter_mut().enumerate() {
                *pair = self.pair_at(start + offset as u64)?;
            }
            return Ok(());
        }
        let byte_len = out
            .len()
            .checked_mul(16)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "successor read buffer size overflow"))?;
        let offset = start
            .checked_mul(16)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "successor read offset overflow"))?;
        let mut bytes = Vec::new();
        bytes.try_reserve_exact(byte_len).map_err(|error| {
            io::Error::new(io::ErrorKind::OutOfMemory, format!("cannot reserve bounded successor read buffer: {error}"))
        })?;
        bytes.resize(byte_len, 0);
        read_exact_at(&self.backing.file, &mut bytes, offset)?;
        for (pair, raw) in out.iter_mut().zip(bytes.chunks_exact(16)) {
            pair.0 = u64::from_le_bytes(raw[..8].try_into().unwrap());
            pair.1 = u64::from_le_bytes(raw[8..].try_into().unwrap());
        }
        Ok(())
    }

    pub(crate) fn read_pairs_inplace(
        &self,
        start: u64,
        out: &mut [(u64, u64)],
        bytes: &mut Vec<u8>,
    ) -> io::Result<()> {
        let count = u64::try_from(out.len())
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "successor read length overflow"))?;
        if start
            .checked_add(count)
            .map(|end| end > self.len)
            .unwrap_or(true)
        {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "successor read exceeds declared pair count"));
        }
        if matches!(self.backing.encoding, NativeSuccessorEncoding::PackedIndices { .. }) {
            let file_len = self.packed_file_len()?;
            let mut cache = self.backing.read_cache.lock().map_err(|_| {
                io::Error::new(io::ErrorKind::Other, "native successor read cache mutex is poisoned")
            })?;
            if cache.is_none() {
                *cache = Some(NativeSuccessorReadCache::new(NATIVE_SUCCESSOR_READ_CACHE_PAGES)?);
            }
            let cache = cache.as_mut().ok_or_else(|| {
                io::Error::new(io::ErrorKind::Other, "native successor read cache was not initialized")
            })?;
            for (offset, pair) in out.iter_mut().enumerate() {
                *pair = cache.pair_at(
                    &self.backing.file,
                    self.backing.file_offset,
                    file_len,
                    self.backing.encoding,
                    self.len,
                    start + offset as u64,
                )?;
            }
            bytes.clear();
            return Ok(());
        }
        let byte_len = out
            .len()
            .checked_mul(16)
            .filter(|len| *len <= 4096 * 16)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "successor range exceeds bounded read chunk"))?;
        let offset = start
            .checked_mul(16)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "successor read offset overflow"))?;
        if bytes.capacity() < byte_len {
            bytes.try_reserve_exact(byte_len - bytes.len()).map_err(|error| {
                io::Error::new(io::ErrorKind::OutOfMemory, format!("cannot reserve bounded successor read buffer: {error}"))
            })?;
        }
        bytes.resize(byte_len, 0);
        read_exact_at(&self.backing.file, bytes, offset)?;
        for (pair, raw) in out.iter_mut().zip(bytes.chunks_exact(16)) {
            pair.0 = u64::from_le_bytes(raw[..8].try_into().unwrap());
            pair.1 = u64::from_le_bytes(raw[8..].try_into().unwrap());
        }
        Ok(())
    }

    pub(crate) fn pair_at(&self, index: u64) -> io::Result<(u64, u64)> {
        let mut cache = self.backing.read_cache.lock().map_err(|_| {
            io::Error::new(io::ErrorKind::Other, "native successor read cache mutex is poisoned")
        })?;
        if cache.is_none() {
            *cache = Some(NativeSuccessorReadCache::new(NATIVE_SUCCESSOR_READ_CACHE_PAGES)?);
        }
        cache
            .as_mut()
            .ok_or_else(|| io::Error::new(io::ErrorKind::Other, "native successor read cache was not initialized"))?
            .pair_at(
                &self.backing.file,
                self.backing.file_offset,
                self.packed_file_len()?,
                self.backing.encoding,
                self.len,
                index,
            )
    }

    pub(crate) fn lower_bound_source(&self, source: u64) -> io::Result<u64> {
        let mut cache = self.backing.read_cache.lock().map_err(|_| {
            io::Error::new(io::ErrorKind::Other, "native successor read cache mutex is poisoned")
        })?;
        if cache.is_none() {
            *cache = Some(NativeSuccessorReadCache::new(NATIVE_SUCCESSOR_READ_CACHE_PAGES)?);
        }
        cache
            .as_mut()
            .ok_or_else(|| io::Error::new(io::ErrorKind::Other, "native successor read cache was not initialized"))?
            .lower_bound_source(
                &self.backing.file,
                self.backing.file_offset,
                self.packed_file_len()?,
                self.backing.encoding,
                self.len,
                source,
            )
    }

    fn source_ordinal(&self, source: u64) -> io::Result<Option<u64>> {
        let mut cache = self.backing.read_cache.lock().map_err(|_| {
            io::Error::new(io::ErrorKind::Other, "native successor read cache mutex is poisoned")
        })?;
        if cache.is_none() {
            *cache = Some(NativeSuccessorReadCache::new(NATIVE_SUCCESSOR_READ_CACHE_PAGES)?);
        }
        cache
            .as_mut()
            .ok_or_else(|| io::Error::new(io::ErrorKind::Other, "native successor read cache was not initialized"))?
            .source_ordinal(
                &self.backing.file,
                self.backing.file_offset,
                self.packed_file_len()?,
                self.backing.encoding,
                self.len,
                source,
            )
    }

    pub(crate) fn validate_bijection(&self, max_cells: u64, max_total_backing_bytes: u64) -> io::Result<NativeSuccessorValidationStats> {
        let successor_bytes = self.disk_bytes();
        let target_bitmap_budget = max_total_backing_bytes.checked_sub(successor_bytes).ok_or_else(|| {
            io::Error::new(io::ErrorKind::OutOfMemory, "successor backing exceeds the configured validation budget")
        })?;
        let mut targets = NativeTargetBitmap::new_with_budget(max_cells, target_bitmap_budget)?;
        const VALIDATION_CHUNK_PAIRS: usize = 4096;
        let mut pairs = Vec::new();
        pairs.try_reserve_exact(VALIDATION_CHUNK_PAIRS).map_err(|error| {
            io::Error::new(io::ErrorKind::OutOfMemory, format!("cannot reserve bounded successor validation buffer: {error}"))
        })?;
        pairs.resize(VALIDATION_CHUNK_PAIRS, (0u64, 0u64));
        let mut raw = Vec::new();
        raw.try_reserve_exact(VALIDATION_CHUNK_PAIRS * 16).map_err(|error| {
            io::Error::new(io::ErrorKind::OutOfMemory, format!("cannot reserve bounded successor validation byte buffer: {error}"))
        })?;
        let mut previous_source = None;
        let mut start = 0u64;
        while start < self.len {
            let count = usize::try_from((self.len - start).min(VALIDATION_CHUNK_PAIRS as u64))
                .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "successor chunk length overflow"))?;
            self.read_pairs_inplace(start, &mut pairs[..count], &mut raw)?;
            for &(source, target) in &pairs[..count] {
                if source >= max_cells || target >= max_cells || source == target {
                    return Err(io::Error::new(io::ErrorKind::InvalidData, "successor entry is outside the permutation domain or is identity"));
                }
                if previous_source.map(|previous| previous >= source).unwrap_or(false) {
                    return Err(io::Error::new(io::ErrorKind::InvalidData, "successor sources are not strictly ordered"));
                }
                previous_source = Some(source);
                if !targets.insert(target)? {
                    return Err(io::Error::new(io::ErrorKind::InvalidData, "successor targets are not unique"));
                }
            }
            start = start
                .checked_add(count as u64)
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "successor validation offset overflow"))?;
        }
        // Source-order validation above proves there are exactly `len`
        // distinct in-range sources. Targets are also exactly `len` distinct
        // in-range cells. Clearing every source from the target bitmap proves
        // the two sets are equal, without a binary search per target or a
        // second full-domain bitmap.
        targets.flush()?;
        let mut visited = 0u64;
        let mut start = 0u64;
        while start < self.len {
            let count = usize::try_from((self.len - start).min(VALIDATION_CHUNK_PAIRS as u64))
                .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "successor closure chunk length overflow"))?;
            self.read_pairs_inplace(start, &mut pairs[..count], &mut raw)?;
            for &(source, _) in &pairs[..count] {
                if !targets.take(source)? {
                    return Err(io::Error::new(io::ErrorKind::InvalidData, "successor target is not in the native source set"));
                }
                visited = visited.checked_add(1).ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "successor validation visit count overflow")
                })?;
            }
            start = start
                .checked_add(count as u64)
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "successor closure offset overflow"))?;
        }
        if visited != self.len {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "successor source-set closure count mismatch"));
        }
        targets.flush()?;
        Ok(NativeSuccessorValidationStats {
            target_bitmap_virtual_bytes: targets.virtual_bytes(),
            target_bitmap_allocated_bytes: targets.allocated_bytes(),
            target_bitmap_cache_bytes: targets.cache_bytes(),
            target_bitmap_allocation_bitmap_bytes: targets.allocation_bitmap_bytes(),
        })
    }

    pub(crate) fn flush(&self) -> io::Result<()> {
        if self.backing.path.is_some() {
            self.backing.file.sync_data()
        } else {
            Ok(())
        }
    }

    pub(crate) fn disk_bytes(&self) -> u64 {
        self.reserved_bytes
    }

    fn packed_file_len(&self) -> io::Result<u64> {
        match self.backing.encoding {
            NativeSuccessorEncoding::RawPairs => self
                .len
                .checked_mul(16)
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "successor backing size overflow")),
            NativeSuccessorEncoding::PackedIndices { bits_per_index } => self
                .len
                .checked_mul(2)
                .and_then(|count| count.checked_mul(u64::from(bits_per_index)))
                .and_then(|bits| bits.checked_add(7))
                .map(|bits| bits / 8)
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "packed successor backing size overflow")),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct NativeSuccessorValidationStats {
    pub(crate) target_bitmap_virtual_bytes: u64,
    pub(crate) target_bitmap_allocated_bytes: u64,
    pub(crate) target_bitmap_cache_bytes: u64,
    pub(crate) target_bitmap_allocation_bitmap_bytes: u64,
}

/// Fixed native target-membership cache bytes, including its page payload,
/// page and index descriptors, and embedded cache headers. The allocation
/// bitmap is reported separately because it scales with the permutation cell count.
pub const NATIVE_TARGET_BITMAP_FIXED_CACHE_BYTES: u64 =
    (std::mem::size_of::<NativeTargetBitmap>()
        + std::mem::size_of::<SparsePageCache>()
        + NativeTargetBitmap::CACHE_PAGES
            * (PERMUTATION_PAGE_BYTES + std::mem::size_of::<CachedPage>())
        + (NativeTargetBitmap::CACHE_PAGES * 2) * std::mem::size_of::<CacheSlotIndex>()) as u64;

/// Return the exact source-derived size of the sparse target-bitmap page
/// allocation bitmap for `permutation_cells`, with overflow checked.
pub fn native_target_bitmap_allocation_bitmap_bytes(permutation_cells: u64) -> Option<u64> {
    let bitmap_bytes = permutation_cells.checked_add(7)? / 8;
    let page_count = bitmap_bytes.checked_add(PERMUTATION_PAGE_BYTES as u64 - 1)?
        / PERMUTATION_PAGE_BYTES as u64;
    Some(page_count.checked_add(7)? / 8)
}

/// Bounded-memory bitmap used to authenticate target membership while reading
/// a serialized native successor list. Bits are indexed by the source's sorted
/// ordinal, not by the potentially much larger circuit-cell identifier.
pub(crate) struct NativeTargetBitmap {
    path: PathBuf,
    bits: SparsePageCache,
    len: u64,
    virtual_bytes: u64,
    _root: Arc<SpillRoot>,
}

impl NativeTargetBitmap {
    const CACHE_PAGES: usize = 16;

    pub(crate) fn new(bit_count: u64) -> io::Result<Self> {
        Self::new_with_budget(bit_count, u64::MAX)
    }

    pub(crate) fn new_with_budget(bit_count: u64, max_allocated_bytes: u64) -> io::Result<Self> {
        let virtual_bytes = bit_count
            .checked_add(7)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "successor validation bitmap size overflow"))?
            / 8;
        let pages = virtual_bytes
            .checked_add(PERMUTATION_PAGE_BYTES as u64 - 1)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "successor validation bitmap page count overflow"))?
            / PERMUTATION_PAGE_BYTES as u64;
        let root = active_spill_root()?;
        let (path, file) = root.create_backing_file("permutation-target-validation")?;
        let bits = match SparsePageCache::new_with_budget(file, pages, Self::CACHE_PAGES, true, max_allocated_bytes) {
            Ok(bits) => bits,
            Err(error) => {
                let _ = fs::remove_file(&path);
                return Err(error);
            }
        };
        Ok(Self {
            path,
            bits,
            len: bit_count,
            virtual_bytes,
            _root: root,
        })
    }

    /// Sets a target bit, returning false if it was already set.
    pub(crate) fn insert(&mut self, cell: u64) -> io::Result<bool> {
        if cell >= self.len {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "successor validation cell is out of range"));
        }
        let byte_offset = cell / 8;
        let mask = 1u8 << (cell % 8);
        let old = self.bits.read_u8(byte_offset)?;
        if old & mask != 0 {
            return Ok(false);
        }
        self.bits.write_u8(byte_offset, old | mask)?;
        Ok(true)
    }

    /// Clears one bit and returns whether it was previously set.
    pub(crate) fn take(&mut self, cell: u64) -> io::Result<bool> {
        if cell >= self.len {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "successor validation cell is out of range"));
        }
        let byte_offset = cell / 8;
        let mask = 1u8 << (cell % 8);
        let old = self.bits.read_u8(byte_offset)?;
        if old & mask == 0 {
            return Ok(false);
        }
        self.bits.write_u8(byte_offset, old & !mask)?;
        Ok(true)
    }

    pub(crate) fn flush(&mut self) -> io::Result<()> {
        self.bits.flush()
    }

    pub(crate) fn virtual_bytes(&self) -> u64 {
        self.virtual_bytes
    }

    pub(crate) fn allocated_bytes(&self) -> u64 {
        self.bits
            .allocated_pages()
            .saturating_mul(PERMUTATION_PAGE_BYTES as u64)
    }

    pub(crate) fn cache_bytes(&self) -> u64 {
        self.bits.cache_bytes() as u64
    }

    pub(crate) fn allocation_bitmap_bytes(&self) -> u64 {
        self.bits.allocation_bitmap_bytes() as u64
    }
}

impl Drop for NativeTargetBitmap {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

struct CachedPage {
    page_index: Option<u64>,
    bytes: Box<[u8; PERMUTATION_PAGE_BYTES]>,
    dirty: bool,
    referenced: bool,
    reserved: bool,
}

impl CachedPage {
    fn empty() -> Self {
        Self {
            page_index: None,
            bytes: Box::new([0; PERMUTATION_PAGE_BYTES]),
            dirty: false,
            referenced: false,
            reserved: false,
        }
    }
}

struct SparsePageCache {
    file: File,
    page_count: u64,
    pages: Vec<CachedPage>,
    slots: Vec<CacheSlotIndex>,
    clock_hand: usize,
    track_allocations: bool,
    allocation_bits: Vec<u8>,
    allocated_pages: u64,
    max_allocated_bytes: u64,
}

#[derive(Clone, Copy)]
struct CacheSlotIndex {
    key: u64,
    slot: usize,
}

impl CacheSlotIndex {
    const EMPTY: u64 = 0;
    const TOMBSTONE: u64 = u64::MAX;

    fn empty() -> Self {
        Self {
            key: Self::EMPTY,
            slot: 0,
        }
    }
}

impl SparsePageCache {
    fn new(file: File, page_count: u64, cache_pages: usize, track_allocations: bool) -> io::Result<Self> {
        Self::new_with_budget(file, page_count, cache_pages, track_allocations, u64::MAX)
    }

    fn new_with_budget(file: File, page_count: u64, cache_pages: usize, track_allocations: bool, max_allocated_bytes: u64) -> io::Result<Self> {
        let byte_len = page_count
            .checked_mul(PERMUTATION_PAGE_BYTES as u64)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "permutation backing size overflow"))?;
        let off_t_len = libc::off_t::try_from(byte_len)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "permutation backing exceeds off_t"))?;
        file.set_len(off_t_len as u64)?;
        let mut pages = Vec::new();
        pages.try_reserve_exact(cache_pages).map_err(|error| {
            io::Error::new(io::ErrorKind::OutOfMemory, format!("cannot reserve bounded permutation cache: {error}"))
        })?;
        pages.resize_with(cache_pages, CachedPage::empty);
        let slot_count = cache_pages
            .checked_mul(2)
            .and_then(usize::checked_next_power_of_two)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "permutation cache index size overflow"))?;
        let mut slots = Vec::new();
        slots.try_reserve_exact(slot_count).map_err(|error| {
            io::Error::new(io::ErrorKind::OutOfMemory, format!("cannot reserve bounded permutation cache index: {error}"))
        })?;
        slots.resize_with(slot_count, CacheSlotIndex::empty);
        let allocation_bits_len = if track_allocations {
            usize::try_from(page_count.saturating_add(7) / 8).map_err(|_| {
                io::Error::new(io::ErrorKind::OutOfMemory, "permutation allocation bitmap does not fit usize")
            })?
        } else {
            0
        };
        let mut allocation_bits = Vec::new();
        allocation_bits.try_reserve_exact(allocation_bits_len).map_err(|error| {
            io::Error::new(io::ErrorKind::OutOfMemory, format!("cannot reserve permutation allocation bitmap: {error}"))
        })?;
        allocation_bits.resize(allocation_bits_len, 0);
        Ok(Self {
            file,
            page_count,
            pages,
            slots,
            clock_hand: 0,
            track_allocations,
            allocation_bits,
            allocated_pages: 0,
            max_allocated_bytes,
        })
    }

    fn cache_bytes(&self) -> usize {
        self.pages.len() * PERMUTATION_PAGE_BYTES
            + self.pages.capacity() * std::mem::size_of::<CachedPage>()
            + self.slots.capacity() * std::mem::size_of::<CacheSlotIndex>()
            + self.allocation_bits.capacity()
    }

    fn allocated_pages(&self) -> u64 {
        self.allocated_pages
    }

    fn allocation_bitmap_bytes(&self) -> usize {
        self.allocation_bits.capacity()
    }

    fn slot_for(&mut self, page_index: u64) -> io::Result<usize> {
        if page_index >= self.page_count {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "permutation backing page is outside the declared shape"));
        }
        if let Some(slot) = self.lookup_slot(page_index) {
            self.pages[slot].referenced = true;
            return Ok(slot);
        }
        if self.pages.is_empty() {
            return Err(io::Error::new(io::ErrorKind::OutOfMemory, "permutation cache has zero pages"));
        }
        let slot = self.choose_victim()?;
        self.flush_slot(slot)?;
        if let Some(old_page) = self.pages[slot].page_index {
            self.remove_slot(old_page);
        }
        let offset = page_index
            .checked_mul(PERMUTATION_PAGE_BYTES as u64)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "permutation backing offset overflow"))?;
        read_exact_at(&self.file, &mut self.pages[slot].bytes[..], offset)?;
        self.pages[slot].page_index = Some(page_index);
        self.pages[slot].dirty = false;
        self.pages[slot].referenced = true;
        self.pages[slot].reserved = self.is_allocated(page_index)?
            || self.pages[slot].bytes.iter().any(|byte| *byte != 0);
        self.insert_slot(page_index, slot)?;
        Ok(slot)
    }

    fn hash_page(page_index: u64) -> usize {
        let mut value = page_index.wrapping_add(0x9e3779b97f4a7c15);
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
        (value ^ (value >> 31)) as usize
    }

    fn lookup_slot(&self, page_index: u64) -> Option<usize> {
        let key = page_index.checked_add(1)?;
        let mask = self.slots.len() - 1;
        let mut index = Self::hash_page(page_index) & mask;
        for _ in 0..self.slots.len() {
            let entry = self.slots[index];
            if entry.key == CacheSlotIndex::EMPTY {
                return None;
            }
            if entry.key == key {
                return Some(entry.slot);
            }
            index = (index + 1) & mask;
        }
        None
    }

    fn insert_slot(&mut self, page_index: u64, slot: usize) -> io::Result<()> {
        let key = page_index
            .checked_add(1)
            .filter(|key| *key != CacheSlotIndex::TOMBSTONE)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "permutation cache index key overflow"))?;
        let mask = self.slots.len() - 1;
        let mut index = Self::hash_page(page_index) & mask;
        let mut tombstone = None;
        for _ in 0..self.slots.len() {
            let entry = self.slots[index];
            if entry.key == CacheSlotIndex::EMPTY {
                let target = tombstone.unwrap_or(index);
                self.slots[target] = CacheSlotIndex { key, slot };
                return Ok(());
            }
            if entry.key == CacheSlotIndex::TOMBSTONE && tombstone.is_none() {
                tombstone = Some(index);
            } else if entry.key == key {
                return Err(io::Error::new(io::ErrorKind::InvalidData, "duplicate permutation cache page key"));
            }
            index = (index + 1) & mask;
        }
        if let Some(target) = tombstone {
            self.slots[target] = CacheSlotIndex { key, slot };
            Ok(())
        } else {
            Err(io::Error::new(io::ErrorKind::OutOfMemory, "permutation cache index is full"))
        }
    }

    fn remove_slot(&mut self, page_index: u64) {
        let Some(key) = page_index.checked_add(1) else {
            return;
        };
        let mask = self.slots.len() - 1;
        let mut index = Self::hash_page(page_index) & mask;
        for _ in 0..self.slots.len() {
            let entry = self.slots[index];
            if entry.key == CacheSlotIndex::EMPTY {
                return;
            }
            if entry.key == key {
                self.slots[index].key = CacheSlotIndex::TOMBSTONE;
                return;
            }
            index = (index + 1) & mask;
        }
    }

    fn choose_victim(&mut self) -> io::Result<usize> {
        for (index, page) in self.pages.iter().enumerate() {
            if page.page_index.is_none() {
                return Ok(index);
            }
        }
        loop {
            let index = self.clock_hand;
            self.clock_hand = (self.clock_hand + 1) % self.pages.len();
            if self.pages[index].referenced {
                self.pages[index].referenced = false;
            } else {
                return Ok(index);
            }
        }
    }

    fn is_allocated(&self, page_index: u64) -> io::Result<bool> {
        if !self.track_allocations {
            return Ok(false);
        }
        let byte = usize::try_from(page_index / 8)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "permutation bitmap index overflow"))?;
        let bit = (page_index % 8) as u8;
        Ok(self.allocation_bits[byte] & (1 << bit) != 0)
    }

    fn reserve_slot(&mut self, slot: usize) -> io::Result<()> {
        if self.pages[slot].reserved {
            return Ok(());
        }
        let page_index = self.pages[slot]
            .page_index
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "cannot reserve an unloaded permutation page"))?;
        if self.track_allocations && self.is_allocated(page_index)? {
            self.pages[slot].reserved = true;
            return Ok(());
        }
        let projected_bytes = self
            .allocated_pages
            .checked_add(1)
            .and_then(|pages| pages.checked_mul(PERMUTATION_PAGE_BYTES as u64))
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "permutation backing allocation count overflow"))?;
        if projected_bytes > self.max_allocated_bytes {
            return Err(io::Error::new(
                io::ErrorKind::OutOfMemory,
                format!("native sparse backing budget exceeded: need {projected_bytes} bytes, limit {}", self.max_allocated_bytes),
            ));
        }
        reserve_file_range(
            &self.file,
            page_index
                .checked_mul(PERMUTATION_PAGE_BYTES as u64)
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "permutation reservation offset overflow"))?,
            PERMUTATION_PAGE_BYTES as u64,
        )?;
        if self.track_allocations {
            let byte = usize::try_from(page_index / 8)
                .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "permutation bitmap index overflow"))?;
            let bit = (page_index % 8) as u8;
            self.allocation_bits[byte] |= 1 << bit;
        }
        self.allocated_pages = self
            .allocated_pages
            .checked_add(1)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "permutation allocated page count overflow"))?;
        self.pages[slot].reserved = true;
        Ok(())
    }

    fn reserve_page(&mut self, page_index: u64) -> io::Result<()> {
        let slot = self.slot_for(page_index)?;
        self.reserve_slot(slot)
    }

    fn read_u64(&mut self, byte_offset: u64) -> io::Result<u64> {
        let end = byte_offset
            .checked_add(8)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "permutation word offset overflow"))?;
        let file_len = self.page_count * PERMUTATION_PAGE_BYTES as u64;
        if end > file_len {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "permutation word is outside the declared backing"));
        }
        let page_index = byte_offset / PERMUTATION_PAGE_BYTES as u64;
        let in_page = (byte_offset % PERMUTATION_PAGE_BYTES as u64) as usize;
        if in_page > PERMUTATION_PAGE_BYTES - 8 {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "permutation word crosses a backing page"));
        }
        let slot = self.slot_for(page_index)?;
        Ok(u64::from_le_bytes(self.pages[slot].bytes[in_page..in_page + 8].try_into().unwrap()))
    }

    fn read_u8(&mut self, byte_offset: u64) -> io::Result<u8> {
        let file_len = self
            .page_count
            .checked_mul(PERMUTATION_PAGE_BYTES as u64)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "permutation backing size overflow"))?;
        if byte_offset >= file_len {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "permutation byte is outside the declared backing"));
        }
        let page_index = byte_offset / PERMUTATION_PAGE_BYTES as u64;
        let in_page = (byte_offset % PERMUTATION_PAGE_BYTES as u64) as usize;
        let slot = self.slot_for(page_index)?;
        Ok(self.pages[slot].bytes[in_page])
    }

    fn write_u8(&mut self, byte_offset: u64, value: u8) -> io::Result<()> {
        let file_len = self
            .page_count
            .checked_mul(PERMUTATION_PAGE_BYTES as u64)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "permutation backing size overflow"))?;
        if byte_offset >= file_len {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "permutation byte is outside the declared backing"));
        }
        let page_index = byte_offset / PERMUTATION_PAGE_BYTES as u64;
        let in_page = (byte_offset % PERMUTATION_PAGE_BYTES as u64) as usize;
        let slot = self.slot_for(page_index)?;
        self.reserve_slot(slot)?;
        self.pages[slot].bytes[in_page] = value;
        self.pages[slot].dirty = true;
        Ok(())
    }

    fn read_first_words(
        &mut self,
        page_index: u64,
        first_record: usize,
        out: &mut [u64],
    ) -> io::Result<()> {
        if first_record
            .checked_add(out.len())
            .map(|end| end > PERMUTATION_RECORDS_PER_PAGE as usize)
            .unwrap_or(true)
        {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "permutation record range crosses a page"));
        }
        let slot = self.slot_for(page_index)?;
        for (offset, value) in out.iter_mut().enumerate() {
            let byte_offset = (first_record + offset) * 16;
            *value = u64::from_le_bytes(
                self.pages[slot].bytes[byte_offset..byte_offset + 8]
                    .try_into()
                    .unwrap(),
            );
        }
        Ok(())
    }

    fn for_each_nonzero_u64(
        &mut self,
        word_count: u64,
        mut visit: impl FnMut(u64, u64) -> io::Result<()>,
    ) -> io::Result<()> {
        let page_words = (PERMUTATION_PAGE_BYTES / 8) as u64;
        for page_index in 0..self.page_count {
            if self.track_allocations && !self.is_allocated(page_index)? {
                continue;
            }
            let first_word = page_index
                .checked_mul(page_words)
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "permutation directory word offset overflow"))?;
            if first_word >= word_count {
                break;
            }
            let slot = self.slot_for(page_index)?;
            let page = &self.pages[slot].bytes;
            for index in 0..page_words {
                let word = first_word + index;
                if word >= word_count {
                    break;
                }
                let offset = index as usize * 8;
                let value = u64::from_le_bytes(page[offset..offset + 8].try_into().unwrap());
                if value != 0 {
                    visit(word, value)?;
                }
            }
        }
        Ok(())
    }

    fn write_u64(&mut self, byte_offset: u64, value: u64) -> io::Result<()> {
        let end = byte_offset
            .checked_add(8)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "permutation word offset overflow"))?;
        let file_len = self.page_count * PERMUTATION_PAGE_BYTES as u64;
        if end > file_len {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "permutation word is outside the declared backing"));
        }
        let page_index = byte_offset / PERMUTATION_PAGE_BYTES as u64;
        let in_page = (byte_offset % PERMUTATION_PAGE_BYTES as u64) as usize;
        if in_page > PERMUTATION_PAGE_BYTES - 8 {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "permutation word crosses a backing page"));
        }
        let slot = self.slot_for(page_index)?;
        self.reserve_slot(slot)?;
        self.pages[slot].bytes[in_page..in_page + 8].copy_from_slice(&value.to_le_bytes());
        self.pages[slot].dirty = true;
        Ok(())
    }

    fn flush_slot(&mut self, slot: usize) -> io::Result<()> {
        if !self.pages[slot].dirty {
            return Ok(());
        }
        self.reserve_slot(slot)?;
        let page_index = self.pages[slot]
            .page_index
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "dirty permutation cache page is unloaded"))?;
        let offset = page_index
            .checked_mul(PERMUTATION_PAGE_BYTES as u64)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "permutation flush offset overflow"))?;
        write_all_at(&self.file, &self.pages[slot].bytes[..], offset)?;
        self.pages[slot].dirty = false;
        Ok(())
    }

    fn flush(&mut self) -> io::Result<()> {
        for slot in 0..self.pages.len() {
            self.flush_slot(slot)?;
        }
        self.file.sync_data()
    }
}

#[cfg(unix)]
fn read_exact_at(file: &File, mut buffer: &mut [u8], mut offset: u64) -> io::Result<()> {
    use std::os::unix::fs::FileExt;
    while !buffer.is_empty() {
        match file.read_at(buffer, offset)? {
            0 => return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "short permutation backing read")),
            read => {
                offset = offset.checked_add(read as u64).ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidInput, "permutation read offset overflow")
                })?;
                buffer = &mut buffer[read..];
            }
        }
    }
    Ok(())
}

#[cfg(unix)]
fn write_all_at(file: &File, mut buffer: &[u8], mut offset: u64) -> io::Result<()> {
    use std::os::unix::fs::FileExt;
    while !buffer.is_empty() {
        match file.write_at(buffer, offset)? {
            0 => return Err(io::Error::new(io::ErrorKind::WriteZero, "short permutation backing write")),
            written => {
                offset = offset.checked_add(written as u64).ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidInput, "permutation write offset overflow")
                })?;
                buffer = &buffer[written..];
            }
        }
    }
    Ok(())
}

fn reserve_file_range(file: &File, offset: u64, byte_len: u64) -> io::Result<()> {
    #[cfg(target_os = "linux")]
    {
        use std::os::fd::AsRawFd;
        let offset = libc::off_t::try_from(offset)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "permutation reservation offset exceeds off_t"))?;
        let byte_len = libc::off_t::try_from(byte_len)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "permutation reservation length exceeds off_t"))?;
        let result = unsafe { libc::posix_fallocate(file.as_raw_fd(), offset, byte_len) };
        if result != 0 {
            return Err(io::Error::from_raw_os_error(result));
        }
        Ok(())
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (file, offset, byte_len);
        Err(io::Error::new(io::ErrorKind::Unsupported, "permutation spill requires positional allocation support"))
    }
}

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

    pub(super) fn truncate(&mut self, len: usize) {
        assert!(len <= self.len);
        self.len = len;
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

    fn create_backing_file(self: &Arc<Self>, prefix: &str) -> io::Result<(PathBuf, File)> {
        for _ in 0..32 {
            let index = self.next_file.fetch_add(1, Ordering::Relaxed);
            let path = self.path.join(format!("{prefix}-{index:016x}.bin"));
            let mut options = OpenOptions::new();
            options.read(true).write(true).create_new(true);
            #[cfg(unix)]
            {
                use std::os::unix::fs::OpenOptionsExt;
                options.mode(0o600);
            }
            match options.open(&path) {
                Ok(file) => return Ok((path, file)),
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(error),
            }
        }
        Err(io::Error::new(
            io::ErrorKind::AlreadyExists,
            "could not reserve sparse native permutation backing files",
        ))
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

#[cfg(test)]
mod native_successor_read_cache_tests {
    use super::{NativeSuccessorReadCache, NATIVE_SUCCESSOR_READ_CACHE_BYTES};
    use std::fs::{self, File, OpenOptions};
    use std::io::{self, Write};
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_TEST_FILE: AtomicU64 = AtomicU64::new(0);

    #[test]
    fn bounded_cache_preserves_sorted_pair_lookups_across_clock_eviction() -> io::Result<()> {
        const PAIR_COUNT: u64 = 2048;
        const PAGE_PAIRS: u64 = 4096 / 16;
        let path = std::env::temp_dir().join(format!(
            "native-successor-cache-{}-{}.bin",
            std::process::id(),
            NEXT_TEST_FILE.fetch_add(1, Ordering::Relaxed),
        ));
        let mut cleanup = RemoveFile(path.clone());
        let mut file = OpenOptions::new().read(true).write(true).create_new(true).open(&path)?;
        let mut bytes = Vec::with_capacity(PAIR_COUNT as usize * 16);
        for index in 0..PAIR_COUNT {
            let source = index * 3;
            bytes.extend_from_slice(&source.to_le_bytes());
            bytes.extend_from_slice(&(source + 1).to_le_bytes());
        }
        file.write_all(&bytes)?;
        file.flush()?;

        let mut cache = NativeSuccessorReadCache::new(2)?;
        let sequence = [0, PAGE_PAIRS - 1, PAGE_PAIRS, 0, PAGE_PAIRS * 2, PAGE_PAIRS, 0, PAGE_PAIRS * 7];
        for index in sequence {
            let expected_source = index * 3;
            assert_eq!(
                cache.pair_at(&file, PAIR_COUNT, index)?,
                (expected_source, expected_source + 1),
            );
        }
        assert_eq!(cache.pages.iter().filter(|page| page.page_index.is_some()).count(), 2);
        assert!(cache.allocated_bytes() <= NATIVE_SUCCESSOR_READ_CACHE_BYTES);

        for index in [0, 1, 255, 256, 512, 1023, PAIR_COUNT - 1] {
            assert_eq!(cache.source_ordinal(&file, PAIR_COUNT, index * 3)?, Some(index));
            assert_eq!(cache.source_ordinal(&file, PAIR_COUNT, index * 3 + 1)?, None);
        }
        drop(cache);
        drop(file);
        cleanup.remove_now()?;
        Ok(())
    }

    struct RemoveFile(PathBuf);

    impl RemoveFile {
        fn remove_now(&mut self) -> io::Result<()> {
            fs::remove_file(&self.0)
        }
    }

    impl Drop for RemoveFile {
        fn drop(&mut self) {
            let _ = fs::remove_file(&self.0);
        }
    }
}
