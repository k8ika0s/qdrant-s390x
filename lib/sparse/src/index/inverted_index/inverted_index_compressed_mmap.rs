use std::any::TypeId;
use std::borrow::Cow;
use std::io::{BufWriter, Write};
use std::marker::PhantomData;
use std::mem::{offset_of, size_of};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use bitpacking::BitPacker as _;
use common::counter::hardware_counter::HardwareCounterCell;
use common::fs::{atomic_save_json, clear_disk_cache, read_json};
use common::mmap::{Advice, AdviceSetting, Madviseable};
#[expect(deprecated, reason = "legacy code")]
use common::mmap::{
    create_and_ensure_length, open_read_mmap, transmute_from_u8_to_slice, transmute_to_u8,
    transmute_to_u8_slice,
};
use common::storage_version::StorageVersion;
use common::types::PointOffsetType;
use memmap2::Mmap;
use serde::{Deserialize, Serialize};

use super::INDEX_FILE_NAME;
use super::inverted_index_compressed_immutable_ram::InvertedIndexCompressedImmutableRam;
use crate::common::sparse_vector::RemappedSparseVector;
use crate::common::types::{DimId, DimOffset, QuantizedU8, QuantizedU8Params, Weight};
use crate::index::compressed_posting_list::{
    CompressedPostingChunk, CompressedPostingList, CompressedPostingListIterator,
    CompressedPostingListView,
};
use crate::index::inverted_index::InvertedIndex;
use crate::index::inverted_index::inverted_index_ram::InvertedIndexRam;
use crate::index::posting_list_common::GenericPostingElement;

const INDEX_CONFIG_FILE_NAME: &str = "inverted_index_config.json";

pub struct Version;

impl StorageVersion for Version {
    fn current_raw() -> &'static str {
        "0.2.0"
    }
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct InvertedIndexFileHeader {
    /// Number of posting lists
    pub posting_count: usize,
    /// Number of unique vectors indexed
    pub vector_count: usize,
    /// Total size of all searchable sparse vectors in bytes
    // This is an option because earlier versions of the index did not store this information.
    // In case it is not present, it will be calculated on load.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_sparse_size: Option<usize>,
}

/// Inverted flatten index from dimension id to posting list
#[derive(Debug)]
pub struct InvertedIndexCompressedMmap<W: Weight> {
    path: PathBuf,
    mmap: Arc<Mmap>,
    decoded_postings: Option<Vec<CompressedPostingList<W>>>,
    pub file_header: InvertedIndexFileHeader,
    _phantom: PhantomData<W>,
}

#[derive(Debug, Default, Clone)]
#[repr(C)]
struct PostingListFileHeader<W: Weight> {
    pub ids_start: u64,
    pub last_id: u32,
    /// Possible values: 0, 4, 8, ..., 512.
    /// Step = 4 = `BLOCK_LEN / u32::BITS` = `128 / 32`.
    /// Max = 512 = `BLOCK_LEN * size_of::<u32>()` = `128 * 4`.
    pub ids_len: u32,
    pub chunks_count: u32,
    pub quantization_params: W::QuantizationParams,
}

#[derive(Debug, Clone, Copy)]
struct PostingListFileHeaderDecoded<W: Weight> {
    ids_start: u64,
    last_id: u32,
    ids_len: u32,
    chunks_count: u32,
    quantization_params: W::QuantizationParams,
}

impl<W: Weight> InvertedIndex for InvertedIndexCompressedMmap<W> {
    type Iter<'a> = CompressedPostingListIterator<'a, W>;

    type Version = Version;

    fn is_on_disk(&self) -> bool {
        true
    }

    fn open(path: &Path) -> std::io::Result<Self> {
        Self::load(path)
    }

    fn save(&self, path: &Path) -> std::io::Result<()> {
        debug_assert_eq!(path, self.path);

        // If Self instance exists, it's either constructed by using `open()` (which reads index
        // files), or using `from_ram_index()` (which writes them). Both assume that the files
        // exist. If any of the files are missing, then something went wrong.
        for file in Self::files(path) {
            debug_assert!(file.exists());
        }

        Ok(())
    }

    fn get<'a>(
        &'a self,
        id: DimOffset,
        hw_counter: &'a HardwareCounterCell,
    ) -> Option<CompressedPostingListIterator<'a, W>> {
        self.get(id, hw_counter)
            .map(|posting_list| posting_list.iter())
    }

    fn len(&self) -> usize {
        self.file_header.posting_count
    }

    fn posting_list_len(&self, id: &DimOffset, hw_counter: &HardwareCounterCell) -> Option<usize> {
        self.get(*id, hw_counter)
            .map(|posting_list| posting_list.len())
    }

    fn files(path: &Path) -> Vec<PathBuf> {
        vec![
            Self::index_file_path(path),
            Self::index_config_file_path(path),
        ]
    }

    fn immutable_files(path: &Path) -> Vec<PathBuf> {
        // `InvertedIndexCompressedMmap` is always immutable
        Self::files(path)
    }

    fn remove(&mut self, _id: PointOffsetType, _old_vector: RemappedSparseVector) {
        panic!("Cannot remove from a read-only Mmap inverted index")
    }

    fn upsert(
        &mut self,
        _id: PointOffsetType,
        _vector: RemappedSparseVector,
        _old_vector: Option<RemappedSparseVector>,
    ) {
        panic!("Cannot upsert into a read-only Mmap inverted index")
    }

    fn from_ram_index<P: AsRef<Path>>(
        ram_index: Cow<InvertedIndexRam>,
        path: P,
    ) -> std::io::Result<Self> {
        let index = InvertedIndexCompressedImmutableRam::from_ram_index(ram_index, &path)?;
        Self::convert_and_save(&index, path)
    }

    fn vector_count(&self) -> usize {
        self.file_header.vector_count
    }

    fn total_sparse_vectors_size(&self) -> usize {
        debug_assert!(
            self.file_header.total_sparse_size.is_some(),
            "The field should be populated from the file, or on load"
        );
        self.file_header.total_sparse_size.unwrap_or(0)
    }

    fn max_index(&self) -> Option<DimId> {
        match self.file_header.posting_count {
            0 => None,
            len => Some(len as DimId - 1),
        }
    }
}

impl<W: Weight> InvertedIndexCompressedMmap<W> {
    const HEADER_SIZE: usize = size_of::<PostingListFileHeader<W>>();

    pub fn index_file_path(path: &Path) -> PathBuf {
        path.join(INDEX_FILE_NAME)
    }

    pub fn index_config_file_path(path: &Path) -> PathBuf {
        path.join(INDEX_CONFIG_FILE_NAME)
    }

    pub fn get<'a>(
        &'a self,
        id: DimId,
        hw_counter: &'a HardwareCounterCell,
    ) -> Option<CompressedPostingListView<'a, W>> {
        // check that the id is not out of bounds (posting_count includes the empty zeroth entry)
        if id >= self.file_header.posting_count as DimId {
            return None;
        }

        if let Some(decoded_postings) = &self.decoded_postings {
            hw_counter.vector_io_read().incr_delta(Self::HEADER_SIZE);
            return decoded_postings
                .get(id as usize)
                .map(|posting| posting.view(hw_counter));
        }

        // TODO Safety.
        let header: PostingListFileHeader<W> = unsafe {
            self.slice_part::<PostingListFileHeader<W>>(
                u64::from(id) * Self::HEADER_SIZE as u64,
                1u32,
            )
        }[0]
        .clone();

        hw_counter.vector_io_read().incr_delta(Self::HEADER_SIZE);

        let remainders_start = header.ids_start
            + u64::from(header.ids_len)
            + u64::from(header.chunks_count) * size_of::<CompressedPostingChunk<W>>() as u64;

        let remainders_end = if id + 1 < self.file_header.posting_count as DimId {
            // TODO Safety
            (unsafe {
                self.slice_part::<PostingListFileHeader<W>>(
                    u64::from(id + 1) * Self::HEADER_SIZE as u64,
                    1u32,
                )
            })[0]
                .ids_start
        } else {
            self.mmap.len() as u64
        };

        if remainders_end
            .checked_sub(remainders_start)
            .is_some_and(|len| len % size_of::<GenericPostingElement<W>>() as u64 != 0)
        {
            return None;
        }

        Some(CompressedPostingListView::new(
            // TODO Safety
            unsafe { self.slice_part(header.ids_start, header.ids_len) },
            // TODO Safety
            unsafe {
                self.slice_part(
                    header.ids_start + u64::from(header.ids_len),
                    header.chunks_count,
                )
            },
            // TODO Safety
            unsafe {
                #[expect(deprecated, reason = "legacy code")]
                transmute_from_u8_to_slice(
                    &self.mmap[remainders_start as usize..remainders_end as usize],
                )
            },
            header.last_id.checked_sub(1),
            header.quantization_params,
            hw_counter,
        ))
    }

    // TODO Safety
    unsafe fn slice_part<T>(&self, start: impl Into<u64>, count: impl Into<u64>) -> &[T] {
        let start = start.into() as usize;
        let end = start + count.into() as usize * size_of::<T>();
        // Safety: safe because of the method safety invariants.
        #[expect(deprecated, reason = "legacy code")]
        unsafe {
            transmute_from_u8_to_slice(&self.mmap[start..end])
        }
    }

    fn invalid_data(message: impl Into<String>) -> std::io::Error {
        std::io::Error::new(std::io::ErrorKind::InvalidData, message.into())
    }

    fn weight_size() -> Option<usize> {
        if TypeId::of::<W>() == TypeId::of::<f32>() {
            Some(size_of::<f32>())
        } else if TypeId::of::<W>() == TypeId::of::<half::f16>() {
            Some(size_of::<u16>())
        } else if TypeId::of::<W>() == TypeId::of::<u8>() {
            Some(size_of::<u8>())
        } else if TypeId::of::<W>() == TypeId::of::<QuantizedU8>() {
            Some(size_of::<u8>())
        } else {
            None
        }
    }

    fn quantization_params_size() -> Option<usize> {
        if TypeId::of::<W>() == TypeId::of::<f32>()
            || TypeId::of::<W>() == TypeId::of::<half::f16>()
            || TypeId::of::<W>() == TypeId::of::<u8>()
        {
            Some(0)
        } else if TypeId::of::<W>() == TypeId::of::<QuantizedU8>() {
            Some(size_of::<QuantizedU8Params>())
        } else {
            None
        }
    }

    fn decode_weight_le(bytes: &[u8]) -> std::io::Result<W> {
        if TypeId::of::<W>() == TypeId::of::<f32>() {
            if bytes.len() != size_of::<f32>() {
                return Err(Self::invalid_data("invalid f32 sparse weight size"));
            }
            let value = f32::from_le_bytes(bytes.try_into().expect("slice size checked"));
            return Ok(unsafe { std::mem::transmute_copy::<f32, W>(&value) });
        }
        if TypeId::of::<W>() == TypeId::of::<half::f16>() {
            if bytes.len() != size_of::<u16>() {
                return Err(Self::invalid_data("invalid f16 sparse weight size"));
            }
            let bits = u16::from_le_bytes(bytes.try_into().expect("slice size checked"));
            let value = half::f16::from_bits(bits);
            return Ok(unsafe { std::mem::transmute_copy::<half::f16, W>(&value) });
        }
        if TypeId::of::<W>() == TypeId::of::<u8>() {
            if bytes.len() != size_of::<u8>() {
                return Err(Self::invalid_data("invalid u8 sparse weight size"));
            }
            let value = bytes[0];
            return Ok(unsafe { std::mem::transmute_copy::<u8, W>(&value) });
        }
        if TypeId::of::<W>() == TypeId::of::<QuantizedU8>() {
            if bytes.len() != size_of::<u8>() {
                return Err(Self::invalid_data("invalid QuantizedU8 sparse weight size"));
            }
            let value = QuantizedU8::from_raw(bytes[0]);
            return Ok(unsafe { std::mem::transmute_copy::<QuantizedU8, W>(&value) });
        }

        Err(Self::invalid_data(format!(
            "unsupported sparse weight type {} for mmap endianness conversion",
            std::any::type_name::<W>()
        )))
    }

    fn encode_weight_le(value: W, out: &mut [u8]) -> std::io::Result<()> {
        if TypeId::of::<W>() == TypeId::of::<f32>() {
            if out.len() != size_of::<f32>() {
                return Err(Self::invalid_data("invalid f32 sparse weight output size"));
            }
            let value = unsafe { std::mem::transmute_copy::<W, f32>(&value) };
            out.copy_from_slice(&value.to_le_bytes());
            return Ok(());
        }
        if TypeId::of::<W>() == TypeId::of::<half::f16>() {
            if out.len() != size_of::<u16>() {
                return Err(Self::invalid_data("invalid f16 sparse weight output size"));
            }
            let value = unsafe { std::mem::transmute_copy::<W, half::f16>(&value) };
            out.copy_from_slice(&value.to_bits().to_le_bytes());
            return Ok(());
        }
        if TypeId::of::<W>() == TypeId::of::<u8>() {
            if out.len() != size_of::<u8>() {
                return Err(Self::invalid_data("invalid u8 sparse weight output size"));
            }
            let value = unsafe { std::mem::transmute_copy::<W, u8>(&value) };
            out[0] = value;
            return Ok(());
        }
        if TypeId::of::<W>() == TypeId::of::<QuantizedU8>() {
            if out.len() != size_of::<u8>() {
                return Err(Self::invalid_data(
                    "invalid QuantizedU8 sparse weight output size",
                ));
            }
            let value = unsafe { std::mem::transmute_copy::<W, QuantizedU8>(&value) };
            out[0] = value.raw();
            return Ok(());
        }

        Err(Self::invalid_data(format!(
            "unsupported sparse weight type {} for mmap endianness conversion",
            std::any::type_name::<W>()
        )))
    }

    fn decode_quantization_params_le(bytes: &[u8]) -> std::io::Result<W::QuantizationParams> {
        if TypeId::of::<W>() == TypeId::of::<f32>()
            || TypeId::of::<W>() == TypeId::of::<half::f16>()
            || TypeId::of::<W>() == TypeId::of::<u8>()
        {
            if !bytes.is_empty() {
                return Err(Self::invalid_data(
                    "unit quantization params must be empty for sparse mmap header",
                ));
            }
            let unit = ();
            return Ok(unsafe { std::mem::transmute_copy::<(), W::QuantizationParams>(&unit) });
        }

        if TypeId::of::<W>() == TypeId::of::<QuantizedU8>() {
            if bytes.len() != size_of::<QuantizedU8Params>() {
                return Err(Self::invalid_data("invalid QuantizedU8 params size"));
            }
            let min = f32::from_le_bytes(bytes[0..4].try_into().expect("slice size checked"));
            let diff256 = f32::from_le_bytes(bytes[4..8].try_into().expect("slice size checked"));
            let params = QuantizedU8Params::from_parts(min, diff256);
            return Ok(unsafe {
                std::mem::transmute_copy::<QuantizedU8Params, W::QuantizationParams>(&params)
            });
        }

        Err(Self::invalid_data(format!(
            "unsupported sparse quantization params type for {}",
            std::any::type_name::<W>()
        )))
    }

    fn encode_quantization_params_le(
        params: W::QuantizationParams,
        out: &mut [u8],
    ) -> std::io::Result<()> {
        if TypeId::of::<W>() == TypeId::of::<f32>()
            || TypeId::of::<W>() == TypeId::of::<half::f16>()
            || TypeId::of::<W>() == TypeId::of::<u8>()
        {
            if !out.is_empty() {
                return Err(Self::invalid_data(
                    "unit quantization params output must be empty",
                ));
            }
            return Ok(());
        }

        if TypeId::of::<W>() == TypeId::of::<QuantizedU8>() {
            if out.len() != size_of::<QuantizedU8Params>() {
                return Err(Self::invalid_data("invalid QuantizedU8 params output size"));
            }
            let params = unsafe {
                std::mem::transmute_copy::<W::QuantizationParams, QuantizedU8Params>(&params)
            };
            let (min, diff256) = params.parts();
            out[0..4].copy_from_slice(&min.to_le_bytes());
            out[4..8].copy_from_slice(&diff256.to_le_bytes());
            return Ok(());
        }

        Err(Self::invalid_data(format!(
            "unsupported sparse quantization params type for {}",
            std::any::type_name::<W>()
        )))
    }

    fn encode_posting_header_le(
        header: &PostingListFileHeaderDecoded<W>,
        out: &mut [u8],
    ) -> std::io::Result<()> {
        const IDS_START_OFFSET: usize = 0;
        const LAST_ID_OFFSET: usize = 8;
        const IDS_LEN_OFFSET: usize = 12;
        const CHUNKS_COUNT_OFFSET: usize = 16;
        const QPARAMS_OFFSET: usize = 20;

        if out.len() != Self::HEADER_SIZE {
            return Err(Self::invalid_data(
                "invalid sparse posting header output size",
            ));
        }

        let params_size = Self::quantization_params_size().ok_or_else(|| {
            Self::invalid_data(format!(
                "unsupported sparse quantization params for {}",
                std::any::type_name::<W>()
            ))
        })?;

        if QPARAMS_OFFSET + params_size > out.len() {
            return Err(Self::invalid_data("invalid sparse posting header layout"));
        }

        out.fill(0);
        out[IDS_START_OFFSET..IDS_START_OFFSET + size_of::<u64>()]
            .copy_from_slice(&header.ids_start.to_le_bytes());
        out[LAST_ID_OFFSET..LAST_ID_OFFSET + size_of::<u32>()]
            .copy_from_slice(&header.last_id.to_le_bytes());
        out[IDS_LEN_OFFSET..IDS_LEN_OFFSET + size_of::<u32>()]
            .copy_from_slice(&header.ids_len.to_le_bytes());
        out[CHUNKS_COUNT_OFFSET..CHUNKS_COUNT_OFFSET + size_of::<u32>()]
            .copy_from_slice(&header.chunks_count.to_le_bytes());
        Self::encode_quantization_params_le(
            header.quantization_params,
            &mut out[QPARAMS_OFFSET..QPARAMS_OFFSET + params_size],
        )?;

        Ok(())
    }

    fn decode_posting_header_le(data: &[u8]) -> std::io::Result<PostingListFileHeaderDecoded<W>> {
        const IDS_START_OFFSET: usize = 0;
        const LAST_ID_OFFSET: usize = 8;
        const IDS_LEN_OFFSET: usize = 12;
        const CHUNKS_COUNT_OFFSET: usize = 16;
        const QPARAMS_OFFSET: usize = 20;

        if data.len() != Self::HEADER_SIZE {
            return Err(Self::invalid_data("invalid sparse posting header size"));
        }

        let params_size = Self::quantization_params_size().ok_or_else(|| {
            Self::invalid_data(format!(
                "unsupported sparse quantization params for {}",
                std::any::type_name::<W>()
            ))
        })?;

        if QPARAMS_OFFSET + params_size > data.len() {
            return Err(Self::invalid_data("invalid sparse posting header layout"));
        }

        let ids_start = u64::from_le_bytes(
            data[IDS_START_OFFSET..IDS_START_OFFSET + size_of::<u64>()]
                .try_into()
                .expect("slice size checked"),
        );
        let last_id = u32::from_le_bytes(
            data[LAST_ID_OFFSET..LAST_ID_OFFSET + size_of::<u32>()]
                .try_into()
                .expect("slice size checked"),
        );
        let ids_len = u32::from_le_bytes(
            data[IDS_LEN_OFFSET..IDS_LEN_OFFSET + size_of::<u32>()]
                .try_into()
                .expect("slice size checked"),
        );
        let chunks_count = u32::from_le_bytes(
            data[CHUNKS_COUNT_OFFSET..CHUNKS_COUNT_OFFSET + size_of::<u32>()]
                .try_into()
                .expect("slice size checked"),
        );

        let quantization_params = Self::decode_quantization_params_le(
            &data[QPARAMS_OFFSET..QPARAMS_OFFSET + params_size],
        )?;

        Ok(PostingListFileHeaderDecoded {
            ids_start,
            last_id,
            ids_len,
            chunks_count,
            quantization_params,
        })
    }

    fn decode_chunks_le(
        bytes: &[u8],
        count: usize,
    ) -> std::io::Result<Vec<CompressedPostingChunk<W>>> {
        let chunk_size = size_of::<CompressedPostingChunk<W>>();
        let expected_len = count
            .checked_mul(chunk_size)
            .ok_or_else(|| Self::invalid_data("sparse chunks size overflow"))?;
        if bytes.len() != expected_len {
            return Err(Self::invalid_data("invalid sparse chunks byte size"));
        }

        let weight_size = Self::weight_size().ok_or_else(|| {
            Self::invalid_data(format!(
                "unsupported sparse weight type {}",
                std::any::type_name::<W>()
            ))
        })?;
        const WEIGHTS_OFFSET: usize = size_of::<u32>() * 2;
        let weights_per_chunk = bitpacking::BitPacker4x::BLOCK_LEN;
        let expected_weight_bytes = weights_per_chunk
            .checked_mul(weight_size)
            .ok_or_else(|| Self::invalid_data("sparse chunk weight size overflow"))?;

        if WEIGHTS_OFFSET + expected_weight_bytes > chunk_size {
            return Err(Self::invalid_data("invalid sparse chunk layout"));
        }

        let mut chunks = Vec::with_capacity(count);
        for chunk_bytes in bytes.chunks_exact(chunk_size) {
            let initial =
                u32::from_le_bytes(chunk_bytes[0..4].try_into().expect("slice size checked"));
            let offset =
                u32::from_le_bytes(chunk_bytes[4..8].try_into().expect("slice size checked"));
            let mut weights = Vec::with_capacity(weights_per_chunk);
            for i in 0..weights_per_chunk {
                let start = WEIGHTS_OFFSET + i * weight_size;
                let end = start + weight_size;
                weights.push(Self::decode_weight_le(&chunk_bytes[start..end])?);
            }
            let weights: [W; bitpacking::BitPacker4x::BLOCK_LEN] = weights
                .try_into()
                .map_err(|_| Self::invalid_data("invalid sparse chunk weight count"))?;
            chunks.push(CompressedPostingChunk::from_parts(initial, offset, weights));
        }
        Ok(chunks)
    }

    fn write_chunks_le(
        writer: &mut impl Write,
        chunks: &[CompressedPostingChunk<W>],
    ) -> std::io::Result<()> {
        let weight_size = Self::weight_size().ok_or_else(|| {
            Self::invalid_data(format!(
                "unsupported sparse weight type {}",
                std::any::type_name::<W>()
            ))
        })?;
        let chunk_size = size_of::<CompressedPostingChunk<W>>();
        const WEIGHTS_OFFSET: usize = size_of::<u32>() * 2;

        for chunk in chunks {
            let mut bytes = vec![0u8; chunk_size];
            bytes[0..4].copy_from_slice(&chunk.initial().to_le_bytes());
            bytes[4..8].copy_from_slice(&chunk.offset().to_le_bytes());
            for (i, &weight) in chunk.weights().iter().enumerate() {
                let start = WEIGHTS_OFFSET + i * weight_size;
                let end = start + weight_size;
                Self::encode_weight_le(weight, &mut bytes[start..end])?;
            }
            writer.write_all(&bytes)?;
        }
        Ok(())
    }

    fn decode_remainders_le(bytes: &[u8]) -> std::io::Result<Vec<GenericPostingElement<W>>> {
        let remainder_size = size_of::<GenericPostingElement<W>>();
        if remainder_size == 0 || !bytes.len().is_multiple_of(remainder_size) {
            return Err(Self::invalid_data("invalid sparse remainders byte size"));
        }

        let weight_size = Self::weight_size().ok_or_else(|| {
            Self::invalid_data(format!(
                "unsupported sparse weight type {}",
                std::any::type_name::<W>()
            ))
        })?;
        let weight_offset = offset_of!(GenericPostingElement<W>, weight);
        if weight_offset + weight_size > remainder_size {
            return Err(Self::invalid_data("invalid sparse remainders layout"));
        }

        let mut remainders = Vec::with_capacity(bytes.len() / remainder_size);
        for remainder in bytes.chunks_exact(remainder_size) {
            let record_id =
                u32::from_le_bytes(remainder[0..4].try_into().expect("slice size checked"));
            let weight =
                Self::decode_weight_le(&remainder[weight_offset..weight_offset + weight_size])?;
            remainders.push(GenericPostingElement { record_id, weight });
        }
        Ok(remainders)
    }

    fn write_remainders_le(
        writer: &mut impl Write,
        remainders: &[GenericPostingElement<W>],
    ) -> std::io::Result<()> {
        let remainder_size = size_of::<GenericPostingElement<W>>();
        let weight_size = Self::weight_size().ok_or_else(|| {
            Self::invalid_data(format!(
                "unsupported sparse weight type {}",
                std::any::type_name::<W>()
            ))
        })?;
        let weight_offset = offset_of!(GenericPostingElement<W>, weight);
        if weight_offset + weight_size > remainder_size {
            return Err(Self::invalid_data("invalid sparse remainders layout"));
        }

        for remainder in remainders {
            let mut bytes = vec![0u8; remainder_size];
            bytes[0..4].copy_from_slice(&remainder.record_id.to_le_bytes());
            Self::encode_weight_le(
                remainder.weight,
                &mut bytes[weight_offset..weight_offset + weight_size],
            )?;
            writer.write_all(&bytes)?;
        }
        Ok(())
    }

    fn decode_postings_le(
        data: &[u8],
        posting_count: usize,
    ) -> std::io::Result<Vec<CompressedPostingList<W>>> {
        let header_bytes = posting_count
            .checked_mul(Self::HEADER_SIZE)
            .ok_or_else(|| Self::invalid_data("sparse header size overflow"))?;
        if header_bytes > data.len() {
            return Err(Self::invalid_data(
                "sparse header region exceeds file length",
            ));
        }

        let mut headers = Vec::with_capacity(posting_count);
        for i in 0..posting_count {
            let start = i * Self::HEADER_SIZE;
            let end = start + Self::HEADER_SIZE;
            headers.push(Self::decode_posting_header_le(&data[start..end])?);
        }

        let chunk_size = size_of::<CompressedPostingChunk<W>>();
        let mut postings = Vec::with_capacity(posting_count);
        for (i, header) in headers.iter().enumerate() {
            let ids_start = usize::try_from(header.ids_start).map_err(|_| {
                Self::invalid_data("ids_start does not fit target architecture address space")
            })?;
            let ids_len = header.ids_len as usize;
            let chunks_count = header.chunks_count as usize;
            let ids_end = ids_start
                .checked_add(ids_len)
                .ok_or_else(|| Self::invalid_data("sparse id_data size overflow"))?;
            let chunks_end = ids_end
                .checked_add(
                    chunk_size
                        .checked_mul(chunks_count)
                        .ok_or_else(|| Self::invalid_data("sparse chunks size overflow"))?,
                )
                .ok_or_else(|| Self::invalid_data("sparse chunks end overflow"))?;
            let remainders_end = if i + 1 < headers.len() {
                usize::try_from(headers[i + 1].ids_start).map_err(|_| {
                    Self::invalid_data(
                        "next ids_start does not fit target architecture address space",
                    )
                })?
            } else {
                data.len()
            };

            if !(ids_start <= ids_end
                && ids_end <= chunks_end
                && chunks_end <= remainders_end
                && remainders_end <= data.len())
            {
                return Err(Self::invalid_data(
                    "invalid sparse posting boundaries in mmap file",
                ));
            }

            let id_data = data[ids_start..ids_end].to_vec();
            let chunks = Self::decode_chunks_le(&data[ids_end..chunks_end], chunks_count)?;
            let remainders = Self::decode_remainders_le(&data[chunks_end..remainders_end])?;

            postings.push(CompressedPostingList::from_parts(
                id_data,
                chunks,
                remainders,
                header.last_id.checked_sub(1),
                header.quantization_params,
            ));
        }

        Ok(postings)
    }

    pub fn convert_and_save<P: AsRef<Path>>(
        index: &InvertedIndexCompressedImmutableRam<W>,
        path: P,
    ) -> std::io::Result<Self> {
        let total_posting_headers_size =
            index.postings.as_slice().len() * size_of::<PostingListFileHeader<W>>();

        // Ignore HW on load
        let hw_counter = HardwareCounterCell::disposable();

        let file_length = total_posting_headers_size
            + index
                .postings
                .as_slice()
                .iter()
                .map(|p| p.view(&hw_counter).store_size().total)
                .sum::<usize>();
        let file_path = Self::index_file_path(path.as_ref());
        let file = create_and_ensure_length(file_path.as_ref(), file_length)?;

        let mut buf = BufWriter::new(file);

        if cfg!(target_endian = "big") {
            // Save posting headers in little-endian while preserving existing repr(C) layout size.
            let mut offset: usize = total_posting_headers_size;
            for posting in index.postings.as_slice() {
                let posting_view = posting.view(&hw_counter);
                let store_size = posting_view.store_size();
                let posting_header = PostingListFileHeaderDecoded::<W> {
                    ids_start: offset as u64,
                    ids_len: store_size.id_data_bytes as u32,
                    chunks_count: store_size.chunks_count as u32,
                    last_id: posting_view.last_id().map_or(0, |id| id + 1),
                    quantization_params: posting_view.multiplier(),
                };
                let mut posting_header_bytes = vec![0u8; Self::HEADER_SIZE];
                Self::encode_posting_header_le(&posting_header, &mut posting_header_bytes)?;
                buf.write_all(&posting_header_bytes)?;
                offset += store_size.total;
            }

            // Save posting payloads in little-endian while preserving existing struct layout.
            for posting in index.postings.as_slice() {
                let posting_view = posting.view(&hw_counter);
                let (id_data, chunks, remainders) = posting_view.parts();
                buf.write_all(id_data)?;
                Self::write_chunks_le(&mut buf, chunks)?;
                Self::write_remainders_le(&mut buf, remainders)?;
            }
        } else {
            // Save posting headers
            let mut offset: usize = total_posting_headers_size;
            for posting in index.postings.as_slice() {
                let store_size = posting.view(&hw_counter).store_size();
                let posting_header = PostingListFileHeader::<W> {
                    ids_start: offset as u64,
                    ids_len: store_size.id_data_bytes as u32,
                    chunks_count: store_size.chunks_count as u32,
                    last_id: posting.view(&hw_counter).last_id().map_or(0, |id| id + 1),
                    quantization_params: posting.view(&hw_counter).multiplier(),
                };
                // TODO Safety
                #[expect(deprecated, reason = "legacy code")]
                buf.write_all(unsafe { transmute_to_u8(&posting_header) })?;
                offset += store_size.total;
            }

            // Save posting elements
            for posting in index.postings.as_slice() {
                let posting_view = posting.view(&hw_counter);
                let (id_data, chunks, remainders) = posting_view.parts();
                buf.write_all(id_data)?;
                // TODO Safety
                #[expect(deprecated, reason = "legacy code")]
                buf.write_all(unsafe { transmute_to_u8_slice(chunks) })?;
                // TODO Safety
                #[expect(deprecated, reason = "legacy code")]
                buf.write_all(unsafe { transmute_to_u8_slice(remainders) })?;
            }
        }

        // Explicitly fsync file contents to ensure durability
        buf.flush()?;
        let file = buf.into_inner().unwrap();
        file.sync_all()?;

        // save header properties
        let file_header = InvertedIndexFileHeader {
            posting_count: index.postings.as_slice().len(),
            vector_count: index.vector_count,
            total_sparse_size: Some(index.total_sparse_size),
        };

        atomic_save_json(&Self::index_config_file_path(path.as_ref()), &file_header)?;

        Ok(Self {
            path: path.as_ref().to_owned(),
            mmap: Arc::new(open_read_mmap(
                file_path.as_ref(),
                AdviceSetting::Global,
                false,
            )?),
            decoded_postings: if cfg!(target_endian = "big") {
                Some(index.postings.as_slice().to_vec())
            } else {
                None
            },
            file_header,
            _phantom: PhantomData,
        })
    }

    pub fn load<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        // read index config file
        let config_file_path = Self::index_config_file_path(path.as_ref());
        // if the file header does not exist, the index is malformed
        let file_header: InvertedIndexFileHeader = read_json(&config_file_path)?;
        // read index data into mmap
        let file_path = Self::index_file_path(path.as_ref());
        let mmap = open_read_mmap(
            file_path.as_ref(),
            AdviceSetting::from(Advice::Normal),
            false,
        )?;

        let decoded_postings = if cfg!(target_endian = "big") {
            Some(Self::decode_postings_le(
                mmap.as_ref(),
                file_header.posting_count,
            )?)
        } else {
            None
        };

        let mut index = Self {
            path: path.as_ref().to_owned(),
            mmap: Arc::new(mmap),
            decoded_postings,
            file_header,
            _phantom: PhantomData,
        };

        let hw_counter = HardwareCounterCell::disposable();

        if index.file_header.total_sparse_size.is_none() {
            index.file_header.total_sparse_size =
                Some(index.calculate_total_sparse_size(&hw_counter));
            atomic_save_json(&config_file_path, &index.file_header)?;
        }

        Ok(index)
    }

    fn calculate_total_sparse_size(&self, hw_counter: &HardwareCounterCell) -> usize {
        (0..self.file_header.posting_count as DimId)
            .filter_map(|id| {
                self.get(id, hw_counter)
                    .map(|posting| posting.store_size().total)
            })
            .sum()
    }

    /// Populate all pages in the mmap.
    /// Block until all pages are populated.
    pub fn populate(&self) -> std::io::Result<()> {
        self.mmap.populate();
        Ok(())
    }

    /// Drop disk cache.
    pub fn clear_cache(&self) -> std::io::Result<()> {
        clear_disk_cache(&self.path)
    }
}

#[cfg(test)]
mod tests {
    use fs_err as fs;
    use tempfile::Builder;

    use super::*;
    use crate::common::types::QuantizedU8;
    use crate::index::inverted_index::inverted_index_ram_builder::InvertedIndexBuilder;

    fn compare_indexes<W: Weight>(
        inverted_index_ram: &InvertedIndexCompressedImmutableRam<W>,
        inverted_index_mmap: &InvertedIndexCompressedMmap<W>,
    ) {
        let hw_counter = HardwareCounterCell::new();
        for id in 0..inverted_index_ram.postings.len() as DimId {
            let posting_list_ram = inverted_index_ram
                .postings
                .get(id as usize)
                .unwrap()
                .view(&hw_counter);
            let posting_list_mmap = inverted_index_mmap.get(id, &hw_counter).unwrap();

            let mmap_parts = posting_list_mmap.parts();
            let ram_parts = posting_list_ram.parts();

            assert_eq!(mmap_parts, ram_parts);
        }
    }

    fn compare_with_decoded_from_raw_file<W: Weight>(
        inverted_index_ram: &InvertedIndexCompressedImmutableRam<W>,
        path: &Path,
        posting_count: usize,
    ) {
        let hw_counter = HardwareCounterCell::new();
        let bytes = fs::read(InvertedIndexCompressedMmap::<W>::index_file_path(path)).unwrap();
        let decoded =
            InvertedIndexCompressedMmap::<W>::decode_postings_le(&bytes, posting_count).unwrap();

        assert_eq!(decoded.len(), posting_count);
        for (id, posting_list_decoded) in decoded.iter().enumerate() {
            let posting_list_ram = inverted_index_ram
                .postings
                .get(id)
                .unwrap()
                .view(&hw_counter);
            assert_eq!(
                posting_list_decoded.view(&hw_counter).parts(),
                posting_list_ram.parts()
            );
        }
    }

    #[test]
    fn test_inverted_index_mmap() {
        check_inverted_index_mmap::<f32>();
        check_inverted_index_mmap::<half::f16>();
        check_inverted_index_mmap::<u8>();
        check_inverted_index_mmap::<QuantizedU8>();
    }

    fn check_inverted_index_mmap<W: Weight>() {
        let hw_counter = HardwareCounterCell::new();

        // skip 4th dimension
        let mut builder = InvertedIndexBuilder::new();
        builder.add(1, [(1, 10.0), (2, 10.0), (3, 10.0), (5, 10.0)].into());
        builder.add(2, [(1, 20.0), (2, 20.0), (3, 20.0), (5, 20.0)].into());
        builder.add(3, [(1, 30.0), (2, 30.0), (3, 30.0)].into());
        builder.add(4, [(1, 1.0), (2, 1.0)].into());
        builder.add(5, [(1, 2.0)].into());
        builder.add(6, [(1, 3.0)].into());
        builder.add(7, [(1, 4.0)].into());
        builder.add(8, [(1, 5.0)].into());
        builder.add(9, [(1, 6.0)].into());
        let inverted_index_ram = builder.build();
        let tmp_dir_path = Builder::new().prefix("test_index_dir1").tempdir().unwrap();
        let inverted_index_ram = InvertedIndexCompressedImmutableRam::from_ram_index(
            Cow::Borrowed(&inverted_index_ram),
            &tmp_dir_path,
        )
        .unwrap();

        let tmp_dir_path = Builder::new().prefix("test_index_dir2").tempdir().unwrap();

        {
            let inverted_index_mmap = InvertedIndexCompressedMmap::<W>::convert_and_save(
                &inverted_index_ram,
                &tmp_dir_path,
            )
            .unwrap();

            compare_indexes(&inverted_index_ram, &inverted_index_mmap);
            compare_with_decoded_from_raw_file(
                &inverted_index_ram,
                tmp_dir_path.path(),
                inverted_index_mmap.file_header.posting_count,
            );
        }
        let inverted_index_mmap = InvertedIndexCompressedMmap::<W>::load(&tmp_dir_path).unwrap();
        // posting_count: 0th entry is always empty + 1st + 2nd + 3rd + 4th empty + 5th
        assert_eq!(inverted_index_mmap.file_header.posting_count, 6);
        assert_eq!(inverted_index_mmap.file_header.vector_count, 9);

        compare_indexes(&inverted_index_ram, &inverted_index_mmap);

        assert!(inverted_index_mmap.get(0, &hw_counter).unwrap().is_empty()); // the first entry is always empty as dimension ids start at 1
        assert_eq!(inverted_index_mmap.get(1, &hw_counter).unwrap().len(), 9);
        assert_eq!(inverted_index_mmap.get(2, &hw_counter).unwrap().len(), 4);
        assert_eq!(inverted_index_mmap.get(3, &hw_counter).unwrap().len(), 3);
        assert!(inverted_index_mmap.get(4, &hw_counter).unwrap().is_empty()); // return empty posting list info for intermediary empty ids
        assert_eq!(inverted_index_mmap.get(5, &hw_counter).unwrap().len(), 2);
        // index after the last values are None
        assert!(inverted_index_mmap.get(6, &hw_counter).is_none());
        assert!(inverted_index_mmap.get(7, &hw_counter).is_none());
        assert!(inverted_index_mmap.get(100, &hw_counter).is_none());
    }

    #[test]
    fn test_decode_postings_le_rejects_truncated_header() {
        let mut builder = InvertedIndexBuilder::new();
        builder.add(1, [(1, 10.0), (2, 20.0)].into());
        let inverted_index_ram = builder.build();
        let tmp_dir_path = Builder::new()
            .prefix("test_index_dir_hdr")
            .tempdir()
            .unwrap();
        let inverted_index_ram = InvertedIndexCompressedImmutableRam::from_ram_index(
            Cow::Borrowed(&inverted_index_ram),
            &tmp_dir_path,
        )
        .unwrap();

        let inverted_index_mmap = InvertedIndexCompressedMmap::<f32>::convert_and_save(
            &inverted_index_ram,
            &tmp_dir_path,
        )
        .unwrap();
        let posting_count = inverted_index_mmap.file_header.posting_count;
        let mut bytes = fs::read(InvertedIndexCompressedMmap::<f32>::index_file_path(
            tmp_dir_path.path(),
        ))
        .unwrap();
        bytes.truncate(posting_count * InvertedIndexCompressedMmap::<f32>::HEADER_SIZE - 1);

        assert!(
            InvertedIndexCompressedMmap::<f32>::decode_postings_le(&bytes, posting_count).is_err()
        );
    }

    #[test]
    fn test_decode_postings_le_rejects_corrupt_offsets() {
        let mut builder = InvertedIndexBuilder::new();
        builder.add(1, [(1, 10.0), (2, 20.0)].into());
        let inverted_index_ram = builder.build();
        let tmp_dir_path = Builder::new()
            .prefix("test_index_dir_off")
            .tempdir()
            .unwrap();
        let inverted_index_ram = InvertedIndexCompressedImmutableRam::from_ram_index(
            Cow::Borrowed(&inverted_index_ram),
            &tmp_dir_path,
        )
        .unwrap();

        let inverted_index_mmap = InvertedIndexCompressedMmap::<f32>::convert_and_save(
            &inverted_index_ram,
            &tmp_dir_path,
        )
        .unwrap();
        let posting_count = inverted_index_mmap.file_header.posting_count;
        let mut bytes = fs::read(InvertedIndexCompressedMmap::<f32>::index_file_path(
            tmp_dir_path.path(),
        ))
        .unwrap();
        let bogus_start = bytes.len() as u64 + 1024;
        bytes[0..8].copy_from_slice(&bogus_start.to_le_bytes());

        assert!(
            InvertedIndexCompressedMmap::<f32>::decode_postings_le(&bytes, posting_count).is_err()
        );
    }
}
