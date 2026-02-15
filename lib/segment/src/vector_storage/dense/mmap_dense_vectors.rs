use std::io::Write;
use std::mem::{MaybeUninit, size_of};
use std::path::Path;
use std::sync::Arc;

use bitvec::prelude::BitSlice;
use common::ext::BitSliceExt as _;
use common::maybe_uninit::maybe_uninit_fill_from;
use common::types::PointOffsetType;
use fs_err::{File, OpenOptions};
use memmap2::Mmap;
use memory::madvise::{Advice, AdviceSetting, Madviseable};
use memory::mmap_ops::{self, MULTI_MMAP_IS_SUPPORTED};
use memory::mmap_type::{MmapBitSlice, MmapFlusher};
use parking_lot::Mutex;

use crate::common::error_logging::LogError;
use crate::common::operation_error::{OperationError, OperationResult};
use crate::data_types::primitive::PrimitiveVectorElement;
#[cfg(target_os = "linux")]
use crate::vector_storage::async_io::UringReader;
#[cfg(not(target_os = "linux"))]
use crate::vector_storage::async_io_mock::UringReader;
use crate::vector_storage::common::VECTOR_READ_BATCH_SIZE;
use crate::vector_storage::mmap_endian::MmapEndianConvertible;
use crate::vector_storage::query_scorer::is_read_with_prefetch_efficient;
use crate::vector_storage::{AccessPattern, Random, Sequential};

const HEADER_SIZE: usize = 4;
const VECTORS_HEADER: &[u8; HEADER_SIZE] = b"data";
const DELETED_HEADER: &[u8; HEADER_SIZE] = b"drop";
const DELETED_LAYOUT_BLOCK_BYTES: usize = size_of::<u64>();

/// Mem-mapped file for dense vectors
#[derive(Debug)]
pub struct MmapDenseVectors<T: PrimitiveVectorElement + MmapEndianConvertible> {
    pub dim: usize,
    pub num_vectors: usize,
    /// Main vector data mmap for read/write
    ///
    /// Has an exact size to fit a header and `num_vectors` of vectors.
    /// Best suited for random reads.
    mmap: Arc<Mmap>,
    /// Read-only mmap best suited for sequential reads
    ///
    /// `None` on platforms that do not support multiple memory maps to the same file.
    /// Use [`mmap_seq`] utility function to access this mmap if available.
    _mmap_seq: Option<Arc<Mmap>>,
    /// Context for io_uring-base async IO
    #[cfg_attr(not(target_os = "linux"), allow(dead_code))]
    uring_reader: Option<Mutex<UringReader<T>>>,
    /// Memory mapped deletion flags
    deleted: MmapBitSlice,
    /// Current number of deleted vectors.
    pub deleted_count: usize,
    /// Cached decoded vectors for BE hosts.
    decoded_vectors: Option<Vec<T>>,
}

impl<T: PrimitiveVectorElement + MmapEndianConvertible> MmapDenseVectors<T> {
    #[inline]
    fn typed_slice_from_bytes(bytes: &[u8], values_count: usize) -> &[T] {
        debug_assert_eq!(bytes.len(), values_count * size_of::<T>());
        // Safety:
        // - caller provides exact element count for `bytes`
        // - vector payload starts after a fixed 4-byte header and uses element types
        //   with <= 4-byte alignment, so typed view alignment is preserved
        unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast::<T>(), values_count) }
    }

    pub fn open(
        vectors_path: &Path,
        deleted_path: &Path,
        dim: usize,
        with_async_io: bool,
        madvise: AdviceSetting,
        populate: bool,
    ) -> OperationResult<Self> {
        // Allocate/open vectors mmap
        ensure_mmap_file_size(vectors_path, VECTORS_HEADER, None)
            .describe("Create mmap data file")?;

        // Validate file length before mmap: empty files can't be mmapped on some platforms, and
        // short/partial headers must never underflow arithmetic below.
        let vectors_len = std::fs::metadata(vectors_path)?.len() as usize;
        if vectors_len < HEADER_SIZE {
            return Err(OperationError::service_error(format!(
                "Invalid mmap vectors file {} size {vectors_len}, expected at least {HEADER_SIZE}",
                vectors_path.display(),
            )));
        }

        let mmap = mmap_ops::open_read_mmap(vectors_path, madvise, populate)
            .describe("Open mmap for reading")?;

        if mmap.len() < HEADER_SIZE {
            // Defensive check: if `mmap_ops` ever returns a smaller mapping than metadata
            // reported, we must still fail safe.
            return Err(OperationError::service_error(format!(
                "Invalid mmap vectors file {} mapping size {}, expected at least {HEADER_SIZE}",
                vectors_path.display(),
                mmap.len(),
            )));
        }
        if &mmap[..HEADER_SIZE] != VECTORS_HEADER {
            return Err(OperationError::service_error(format!(
                "Invalid mmap vectors file {} header, expected {:?}",
                vectors_path.display(),
                VECTORS_HEADER,
            )));
        }

        let vector_bytes = dim.checked_mul(size_of::<T>()).ok_or_else(|| {
            OperationError::service_error("Vector byte size overflow when opening mmap".to_string())
        })?;
        if vector_bytes == 0 {
            return Err(OperationError::service_error(
                "Vector byte size is zero when opening mmap".to_string(),
            ));
        }

        let payload_len = mmap
            .len()
            .checked_sub(HEADER_SIZE)
            .ok_or_else(|| OperationError::service_error("Vectors mmap size underflow".to_string()))?;
        if payload_len % vector_bytes != 0 {
            return Err(OperationError::service_error(format!(
                "Invalid mmap vectors file {} size {}, expected header + N * {vector_bytes}",
                vectors_path.display(),
                mmap.len(),
            )));
        }

        // Only open second mmap for sequential reads if supported
        let mmap_seq = if *MULTI_MMAP_IS_SUPPORTED {
            let mmap_seq = mmap_ops::open_read_mmap(
                vectors_path,
                AdviceSetting::Advice(Advice::Sequential),
                populate,
            )
            .describe("Open mmap for sequential reading")?;
            Some(Arc::new(mmap_seq))
        } else {
            None
        };

        let num_vectors = payload_len / vector_bytes;
        let decoded_vectors = if cfg!(target_endian = "big") {
            Some(Self::decode_vectors(&mmap, dim, num_vectors)?)
        } else {
            None
        };

        // Allocate/open deleted mmap
        let deleted_mmap_size = deleted_mmap_size(num_vectors);
        ensure_mmap_file_size(deleted_path, DELETED_HEADER, Some(deleted_mmap_size as u64))
            .describe("Create mmap deleted file")?;
        let deleted_mmap = mmap_ops::open_write_mmap(deleted_path, AdviceSetting::Global, false)
            .describe("Open mmap deleted for writing")?;

        if deleted_mmap.len() < deleted_mmap_data_start() {
            return Err(OperationError::service_error(format!(
                "Invalid mmap deleted file {} size {}, expected at least {}",
                deleted_path.display(),
                deleted_mmap.len(),
                deleted_mmap_data_start(),
            )));
        }
        if &deleted_mmap[..HEADER_SIZE] != DELETED_HEADER {
            return Err(OperationError::service_error(format!(
                "Invalid mmap deleted file {} header, expected {:?}",
                deleted_path.display(),
                DELETED_HEADER,
            )));
        }

        // Advise kernel that we'll need this page soon so the kernel can prepare
        #[cfg(unix)]
        if let Err(err) = deleted_mmap.advise(memmap2::Advice::WillNeed) {
            log::error!("Failed to advise MADV_WILLNEED for deleted flags: {err}");
        }

        // Transform into mmap BitSlice
        let deleted = MmapBitSlice::try_from(deleted_mmap, deleted_mmap_data_start())?;
        let deleted_count = deleted.count_ones();

        let uring_reader = if with_async_io {
            // Keep file handle open for async IO
            let vectors_file = File::open(vectors_path)?;
            let raw_size = dim * size_of::<T>();
            Some(UringReader::new(vectors_file, raw_size, HEADER_SIZE)?)
        } else {
            None
        };

        Ok(MmapDenseVectors {
            dim,
            num_vectors,
            mmap: mmap.into(),
            _mmap_seq: mmap_seq,
            uring_reader: uring_reader.map(Mutex::new),
            deleted,
            deleted_count,
            decoded_vectors,
        })
    }

    #[inline]
    fn decode_vectors(mmap: &Mmap, dim: usize, num_vectors: usize) -> OperationResult<Vec<T>> {
        let values_count = dim.checked_mul(num_vectors).ok_or_else(|| {
            OperationError::service_error("mmap vectors values_count overflow".to_string())
        })?;
        let values_size = values_count.checked_mul(size_of::<T>()).ok_or_else(|| {
            OperationError::service_error("mmap vectors values_size overflow".to_string())
        })?;
        let byte_slice = &mmap[HEADER_SIZE..HEADER_SIZE + values_size];
        let stored = Self::typed_slice_from_bytes(byte_slice, values_count);
        Ok(stored
            .iter()
            .map(|value| T::from_le_storage(*value))
            .collect())
    }

    pub fn has_async_reader(&self) -> bool {
        self.uring_reader.is_some()
    }

    pub fn flusher(&self) -> MmapFlusher {
        self.deleted.flusher()
    }

    pub fn data_offset(&self, key: PointOffsetType) -> Option<usize> {
        let vector_data_length = self.dim * size_of::<T>();
        let offset = (key as usize) * vector_data_length + HEADER_SIZE;
        if key >= (self.num_vectors as PointOffsetType) {
            return None;
        }
        Some(offset)
    }

    pub fn raw_size(&self) -> usize {
        self.dim * size_of::<T>()
    }

    fn raw_vector_offset<P: AccessPattern>(&self, offset: usize) -> &[T] {
        if let Some(decoded_vectors) = &self.decoded_vectors {
            let vector_start = (offset - HEADER_SIZE) / size_of::<T>();
            let vector_end = vector_start + self.dim;
            return &decoded_vectors[vector_start..vector_end];
        }

        let mmap: &Mmap = if P::IS_SEQUENTIAL {
            self._mmap_seq.as_deref().unwrap_or(self.mmap.as_ref())
        } else {
            self.mmap.as_ref()
        };
        let byte_slice = &mmap[offset..(offset + self.raw_size())];
        let arr = Self::typed_slice_from_bytes(byte_slice, self.dim);
        &arr[0..self.dim]
    }

    /// Returns reference to vector data by key
    fn get_vector<P: AccessPattern>(&self, key: PointOffsetType) -> &[T] {
        self.get_vector_opt::<P>(key).expect("vector not found")
    }

    /// Returns an optional reference to vector data by key
    pub fn get_vector_opt<P: AccessPattern>(&self, key: PointOffsetType) -> Option<&[T]> {
        self.data_offset(key)
            .map(|offset| self.raw_vector_offset::<P>(offset))
    }

    pub fn for_each_in_batch<F: FnMut(usize, &[T])>(&self, keys: &[PointOffsetType], mut f: F) {
        debug_assert!(keys.len() <= VECTOR_READ_BATCH_SIZE);

        // The `f` is most likely a scorer function.
        // Fetching all vectors first then scoring them is more cache friendly
        // then fetching and scoring in a single loop.
        let mut vectors_buffer = [MaybeUninit::uninit(); VECTOR_READ_BATCH_SIZE];
        let vectors = if is_read_with_prefetch_efficient(keys) {
            let iter = keys.iter().map(|key| self.get_vector::<Sequential>(*key));
            maybe_uninit_fill_from(&mut vectors_buffer, iter).0
        } else {
            let iter = keys.iter().map(|key| self.get_vector::<Random>(*key));
            maybe_uninit_fill_from(&mut vectors_buffer, iter).0
        };

        for (i, vec) in vectors.iter().enumerate() {
            f(i, vec);
        }
    }

    /// Marks the key as deleted.
    ///
    /// Returns true if the key was not deleted before, and it is now deleted.
    pub fn delete(&mut self, key: PointOffsetType) -> bool {
        let is_deleted = !self.deleted.replace(key as usize, true);
        if is_deleted {
            self.deleted_count += 1;
        }
        is_deleted
    }

    pub fn is_deleted_vector(&self, key: PointOffsetType) -> bool {
        self.deleted.get_bit(key as usize).unwrap_or(false)
    }

    /// Get [`BitSlice`] representation for deleted vectors with deletion flags
    ///
    /// The size of this slice is not guaranteed. It may be smaller/larger than the number of
    /// vectors in this segment.
    pub fn deleted_vector_bitslice(&self) -> &BitSlice {
        &self.deleted
    }

    fn process_points_simple(
        &self,
        points: impl Iterator<Item = PointOffsetType>,
        mut callback: impl FnMut(usize, PointOffsetType, &[T]),
    ) {
        for (idx, point) in points.enumerate() {
            let vector = self.get_vector::<Random>(point);
            callback(idx, point, vector);
        }
    }

    /// Reads vectors for the given ids and calls the callback for each vector.
    /// Tries to utilize asynchronous IO if possible.
    /// In particular, uses io_uring on Linux and simple synchronous IO otherwise.
    pub fn read_vectors_async(
        &self,
        points: impl Iterator<Item = PointOffsetType>,
        callback: impl FnMut(usize, PointOffsetType, &[T]),
    ) -> OperationResult<()> {
        match &self.uring_reader {
            None => self.process_points_simple(points, callback),

            #[cfg(target_os = "linux")]
            Some(uring_reader) => {
                // Use `UringReader` on Linux
                let mut uring_guard = uring_reader.lock();
                uring_guard.read_stream(points, callback)?;
            }

            #[cfg(not(target_os = "linux"))]
            Some(_) => {
                // Fallback to synchronous processing on non-Linux platforms
                self.process_points_simple(points, callback);
            }
        }
        Ok(())
    }

    pub fn populate(&self) {
        #[expect(clippy::used_underscore_binding)]
        if let Some(mmap_seq) = &self._mmap_seq {
            mmap_seq.populate();
        }
    }
}

/// Ensure the given mmap file exists and is the given size
///
/// # Arguments
/// * `path`: path of the file.
/// * `header`: header to set when the file is newly created.
/// * `size`: set the file size in bytes, filled with zeroes.
fn ensure_mmap_file_size(path: &Path, header: &[u8], size: Option<u64>) -> OperationResult<()> {
    // If it exists, only set the length
    if path.exists() {
        if let Some(size) = size {
            let file = OpenOptions::new().write(true).open(path)?;
            file.set_len(size)?;
        }
        return Ok(());
    }

    // Create file, and make it the correct size
    let mut file = File::create(path)?;
    file.write_all(header)?;
    if let Some(size) = size
        && size > header.len() as u64
    {
        file.set_len(size)?;
    }
    Ok(())
}

/// Get start position of flags `BitSlice` in deleted mmap.
#[inline]
const fn deleted_mmap_data_start() -> usize {
    HEADER_SIZE.div_ceil(DELETED_LAYOUT_BLOCK_BYTES) * DELETED_LAYOUT_BLOCK_BYTES
}

/// Calculate size for deleted mmap to hold the given number of vectors.
///
/// The mmap will hold a file header and an aligned `BitSlice`.
fn deleted_mmap_size(num: usize) -> usize {
    let num_bytes = num.div_ceil(8);
    let data_size = num_bytes.next_multiple_of(DELETED_LAYOUT_BLOCK_BYTES);
    deleted_mmap_data_start() + data_size
}

#[cfg(test)]
mod tests {
    use fs_err as fs;
    use tempfile::Builder;

    use super::*;
    use crate::data_types::vectors::VectorElementType;

    #[test]
    fn test_deleted_mmap_layout_is_fixed_width() {
        assert_eq!(deleted_mmap_data_start(), 8);
        assert_eq!(deleted_mmap_size(0), 8);
        assert_eq!(deleted_mmap_size(1), 16);
        assert_eq!(deleted_mmap_size(64), 16);
        assert_eq!(deleted_mmap_size(65), 24);
    }

    #[test]
    fn test_open_rejects_partial_vectors_header() {
        let dir = Builder::new().prefix("storage_dir").tempdir().unwrap();
        let vectors_path = dir.path().join("data.mmap");
        let deleted_path = dir.path().join("drop.mmap");

        fs::write(&vectors_path, &b"da"[..]).unwrap();

        let err = MmapDenseVectors::<VectorElementType>::open(
            &vectors_path,
            &deleted_path,
            2,
            false,
            AdviceSetting::Global,
            false,
        )
        .unwrap_err();
        assert!(err.to_string().contains("Invalid mmap vectors file"));
    }

    #[test]
    fn test_open_rejects_vectors_header_mismatch() {
        let dir = Builder::new().prefix("storage_dir").tempdir().unwrap();
        let vectors_path = dir.path().join("data.mmap");
        let deleted_path = dir.path().join("drop.mmap");

        fs::write(&vectors_path, &b"nope"[..]).unwrap();

        let err = MmapDenseVectors::<VectorElementType>::open(
            &vectors_path,
            &deleted_path,
            2,
            false,
            AdviceSetting::Global,
            false,
        )
        .unwrap_err();
        assert!(err.to_string().contains("Invalid mmap vectors file"));
    }

    #[test]
    fn test_open_rejects_truncated_vectors_payload() {
        let dir = Builder::new().prefix("storage_dir").tempdir().unwrap();
        let vectors_path = dir.path().join("data.mmap");
        let deleted_path = dir.path().join("drop.mmap");

        // dim=2, f32 => vector_bytes=8. Provide only 4 bytes payload (half vector).
        let mut raw = Vec::new();
        raw.extend_from_slice(VECTORS_HEADER);
        raw.extend_from_slice(&1.0f32.to_le_bytes());
        fs::write(&vectors_path, raw).unwrap();

        let err = MmapDenseVectors::<VectorElementType>::open(
            &vectors_path,
            &deleted_path,
            2,
            false,
            AdviceSetting::Global,
            false,
        )
        .unwrap_err();
        assert!(err.to_string().contains("expected header + N"));
    }

    #[test]
    fn test_open_accepts_header_only_vectors_file() {
        let dir = Builder::new().prefix("storage_dir").tempdir().unwrap();
        let vectors_path = dir.path().join("data.mmap");
        let deleted_path = dir.path().join("drop.mmap");

        fs::write(&vectors_path, VECTORS_HEADER).unwrap();

        let opened = MmapDenseVectors::<VectorElementType>::open(
            &vectors_path,
            &deleted_path,
            2,
            false,
            AdviceSetting::Global,
            false,
        )
        .unwrap();
        assert_eq!(opened.num_vectors, 0);
        assert_eq!(opened.deleted_count, 0);
    }

    #[test]
    fn test_open_rejects_deleted_header_mismatch() {
        let dir = Builder::new().prefix("storage_dir").tempdir().unwrap();
        let vectors_path = dir.path().join("data.mmap");
        let deleted_path = dir.path().join("drop.mmap");

        fs::write(&vectors_path, VECTORS_HEADER).unwrap();
        fs::write(&deleted_path, &b"nope"[..]).unwrap();

        let err = MmapDenseVectors::<VectorElementType>::open(
            &vectors_path,
            &deleted_path,
            2,
            false,
            AdviceSetting::Global,
            false,
        )
        .unwrap_err();
        assert!(err.to_string().contains("Invalid mmap deleted file"));
    }
}
