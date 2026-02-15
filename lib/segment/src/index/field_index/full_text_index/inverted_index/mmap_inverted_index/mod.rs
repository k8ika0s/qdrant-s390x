use std::collections::HashMap;
use std::io::Write as _;
use std::path::PathBuf;

use bitvec::vec::BitVec;
use common::counter::hardware_counter::HardwareCounterCell;
use common::mmap_hashmap::{MmapHashMap, READ_ENTRY_OVERHEAD};
use common::types::PointOffsetType;
use itertools::Either;
use io::file_operations::atomic_save;
use memory::fadvise::clear_disk_cache;
use memory::madvise::AdviceSetting;
use memory::mmap_ops;
use memory::mmap_type::MmapBitSlice;
use mmap_postings::{MmapPostingValue, MmapPostings};

use super::immutable_inverted_index::ImmutableInvertedIndex;
use super::immutable_postings_enum::ImmutablePostings;
use super::mmap_inverted_index::mmap_postings_enum::MmapPostingsEnum;
use super::positions::Positions;
use super::postings_iterator::{
    intersect_compressed_postings_iterator, merge_compressed_postings_iterator,
};
use super::{InvertedIndex, ParsedQuery, TokenId, TokenSet};
use crate::common::Flusher;
use crate::common::mmap_bitslice_buffered_update_wrapper::MmapBitSliceBufferedUpdateWrapper;
use crate::common::operation_error::{OperationError, OperationResult};
use crate::index::field_index::full_text_index::inverted_index::Document;
use crate::index::field_index::full_text_index::inverted_index::postings_iterator::{
    check_compressed_postings_phrase, intersect_compressed_postings_phrase_iterator,
};

pub(super) mod mmap_postings;
pub mod mmap_postings_enum;

const POSTINGS_FILE: &str = "postings.dat";
const VOCAB_FILE: &str = "vocab.dat";
const POINT_TO_TOKENS_COUNT_FILE: &str = "point_to_tokens_count.dat";
const DELETED_POINTS_FILE: &str = "deleted_points.dat";

const POINT_TO_TOKENS_COUNT_MAGIC: &[u8; 4] = b"pttc";
const POINT_TO_TOKENS_COUNT_VERSION: u32 = 1;
const POINT_TO_TOKENS_COUNT_HEADER_SIZE: usize = 16;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LegacyEndian {
    Little,
    Big,
}

fn legacy_usize_from_le_bytes(bytes: &[u8]) -> usize {
    match std::mem::size_of::<usize>() {
        8 => u64::from_le_bytes(bytes.try_into().expect("usize-size mismatch")) as usize,
        4 => u32::from_le_bytes(bytes.try_into().expect("usize-size mismatch")) as usize,
        other => unreachable!("unsupported usize size: {other}"),
    }
}

fn legacy_usize_from_be_bytes(bytes: &[u8]) -> usize {
    match std::mem::size_of::<usize>() {
        8 => u64::from_be_bytes(bytes.try_into().expect("usize-size mismatch")) as usize,
        4 => u32::from_be_bytes(bytes.try_into().expect("usize-size mismatch")) as usize,
        other => unreachable!("unsupported usize size: {other}"),
    }
}

fn detect_legacy_counts_endian(bytes: &[u8]) -> LegacyEndian {
    let word = std::mem::size_of::<usize>();
    debug_assert!(word == 4 || word == 8, "unexpected usize size: {word}");
    if bytes.is_empty() {
        return if cfg!(target_endian = "little") {
            LegacyEndian::Little
        } else {
            LegacyEndian::Big
        };
    }

    let len = bytes.len() / word;
    let sample = len.min(256);

    let mut max_le: usize = 0;
    let mut max_be: usize = 0;
    let mut over_u32_le: usize = 0;
    let mut over_u32_be: usize = 0;

    for i in 0..sample {
        let chunk = &bytes[i * word..(i + 1) * word];
        let le = legacy_usize_from_le_bytes(chunk);
        let be = legacy_usize_from_be_bytes(chunk);
        max_le = max_le.max(le);
        max_be = max_be.max(be);
        if le > u32::MAX as usize {
            over_u32_le += 1;
        }
        if be > u32::MAX as usize {
            over_u32_be += 1;
        }
    }

    if over_u32_le < over_u32_be {
        return LegacyEndian::Little;
    }
    if over_u32_be < over_u32_le {
        return LegacyEndian::Big;
    }
    if max_le < max_be {
        return LegacyEndian::Little;
    }
    if max_be < max_le {
        return LegacyEndian::Big;
    }

    // All-zero, or perfectly ambiguous. Fall back to native.
    if cfg!(target_endian = "little") {
        LegacyEndian::Little
    } else {
        LegacyEndian::Big
    }
}

pub(in crate::index::field_index::full_text_index) struct PointToTokensCount {
    mmap: memmap2::MmapMut,
    len: usize,
}

impl PointToTokensCount {
    fn validate_header(bytes: &[u8]) -> OperationResult<usize> {
        if bytes.len() < POINT_TO_TOKENS_COUNT_HEADER_SIZE {
            return Err(OperationError::service_error(format!(
                "Corrupted {POINT_TO_TOKENS_COUNT_FILE}: file too small ({})",
                bytes.len()
            )));
        }

        let magic: [u8; 4] = bytes[0..4].try_into().expect("slice length mismatch");
        if &magic != POINT_TO_TOKENS_COUNT_MAGIC {
            return Err(OperationError::service_error(format!(
                "Corrupted {POINT_TO_TOKENS_COUNT_FILE}: bad magic {magic:?}",
            )));
        }

        let version = u32::from_le_bytes(bytes[4..8].try_into().expect("slice length mismatch"));
        if version != POINT_TO_TOKENS_COUNT_VERSION {
            return Err(OperationError::service_error(format!(
                "Unsupported {POINT_TO_TOKENS_COUNT_FILE} version: {version}",
            )));
        }

        let len_u64 = u64::from_le_bytes(bytes[8..16].try_into().expect("slice length mismatch"));
        let len = usize::try_from(len_u64).map_err(|_| {
            OperationError::service_error(format!(
                "Corrupted {POINT_TO_TOKENS_COUNT_FILE}: len too large ({len_u64})",
            ))
        })?;

        let expected = POINT_TO_TOKENS_COUNT_HEADER_SIZE
            .checked_add(len.checked_mul(std::mem::size_of::<u32>()).ok_or_else(|| {
                OperationError::service_error(format!(
                    "Corrupted {POINT_TO_TOKENS_COUNT_FILE}: len overflow ({len})",
                ))
            })?)
            .ok_or_else(|| {
                OperationError::service_error(format!(
                    "Corrupted {POINT_TO_TOKENS_COUNT_FILE}: size overflow ({len})",
                ))
            })?;

        if bytes.len() != expected {
            return Err(OperationError::service_error(format!(
                "Corrupted {POINT_TO_TOKENS_COUNT_FILE}: expected {expected} bytes, got {}",
                bytes.len()
            )));
        }

        Ok(len)
    }

    pub fn create(path: &std::path::Path, mut iter: impl ExactSizeIterator<Item = usize>) -> OperationResult<()> {
        let len = iter.len();
        let file_len = POINT_TO_TOKENS_COUNT_HEADER_SIZE + len * std::mem::size_of::<u32>();

        let _file = mmap_ops::create_and_ensure_length(path, file_len)?;
        let mut mmap = mmap_ops::open_write_mmap(
            path,
            AdviceSetting::Advice(memory::madvise::Advice::Normal), // sequential write
            false,
        )?;

        let bytes = mmap.as_mut();
        bytes[0..4].copy_from_slice(POINT_TO_TOKENS_COUNT_MAGIC);
        bytes[4..8].copy_from_slice(&POINT_TO_TOKENS_COUNT_VERSION.to_le_bytes());
        bytes[8..16].copy_from_slice(&(len as u64).to_le_bytes());

        let counts_bytes = &mut bytes[POINT_TO_TOKENS_COUNT_HEADER_SIZE..];
        debug_assert_eq!(counts_bytes.len(), len * std::mem::size_of::<u32>());

        // SAFETY: header size is 16 (multiple of 4), and the mmap is page-aligned. We also
        // validated the buffer length is exactly len * 4.
        let counts: &mut [u32] = unsafe {
            std::slice::from_raw_parts_mut(counts_bytes.as_mut_ptr().cast::<u32>(), len)
        };
        for dst in counts.iter_mut() {
            let value = iter
                .next()
                .expect("iterator size mismatch while writing point_to_tokens_count");
            let value_u32: u32 = value.try_into().map_err(|_| {
                OperationError::service_error(format!(
                    "{POINT_TO_TOKENS_COUNT_FILE}: token count overflows u32 ({value})",
                ))
            })?;
            *dst = value_u32.to_le();
        }

        // Ensure no trailing elements (ExactSizeIterator contract).
        debug_assert!(iter.next().is_none());

        if !mmap.is_empty() {
            mmap.flush()?;
        }
        Ok(())
    }

    fn migrate_legacy(path: &std::path::Path, bytes: &[u8]) -> OperationResult<()> {
        let word = std::mem::size_of::<usize>();
        if word != 4 && word != 8 {
            return Err(OperationError::service_error(format!(
                "Unsupported usize size for legacy migration: {word}",
            )));
        }
        if !bytes.len().is_multiple_of(word) {
            return Err(OperationError::service_error(format!(
                "Corrupted legacy {POINT_TO_TOKENS_COUNT_FILE}: size {} not multiple of {word}",
                bytes.len()
            )));
        }

        let len = bytes.len() / word;
        let detected = detect_legacy_counts_endian(bytes);

        atomic_save::<OperationError, _>(path, |writer| {
            writer.write_all(POINT_TO_TOKENS_COUNT_MAGIC)?;
            writer.write_all(&POINT_TO_TOKENS_COUNT_VERSION.to_le_bytes())?;
            writer.write_all(&(len as u64).to_le_bytes())?;

            for i in 0..len {
                let chunk = &bytes[i * word..(i + 1) * word];
                let value = match detected {
                    LegacyEndian::Little => legacy_usize_from_le_bytes(chunk),
                    LegacyEndian::Big => legacy_usize_from_be_bytes(chunk),
                };
                let value_u32: u32 = value.try_into().map_err(|_| {
                    OperationError::service_error(format!(
                        "legacy {POINT_TO_TOKENS_COUNT_FILE}: token count overflows u32 ({value})",
                    ))
                })?;
                writer.write_all(&value_u32.to_le_bytes())?;
            }
            Ok(())
        })?;

        Ok(())
    }

    pub fn open(path: &std::path::Path, populate: bool) -> OperationResult<Self> {
        // Fast header check without mmap first; if legacy, migrate with streaming rewrite.
        let meta = std::fs::metadata(path).map_err(|err| {
            OperationError::service_error(format!(
                "Failed to stat {POINT_TO_TOKENS_COUNT_FILE}: {err}"
            ))
        })?;
        let file_len = usize::try_from(meta.len()).unwrap_or(usize::MAX);

        let is_new = if file_len >= POINT_TO_TOKENS_COUNT_HEADER_SIZE {
            let mut header = [0u8; 4];
            std::fs::File::open(path)
                .and_then(|mut f| std::io::Read::read_exact(&mut f, &mut header))
                .is_ok()
                && &header == POINT_TO_TOKENS_COUNT_MAGIC
        } else {
            false
        };

        if !is_new {
            // Legacy file: mmap-read it to avoid copying large files.
            let file = std::fs::File::open(path).map_err(|err| {
                OperationError::service_error(format!(
                    "Failed to open legacy {POINT_TO_TOKENS_COUNT_FILE}: {err}"
                ))
            })?;
            let legacy_mmap = unsafe { memmap2::Mmap::map(&file)? };
            Self::migrate_legacy(path, &legacy_mmap)?;
        }

        let mmap = mmap_ops::open_write_mmap(path, AdviceSetting::Global, populate)?;
        let len = Self::validate_header(&mmap)?;
        Ok(Self { mmap, len })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    fn counts(&self) -> &[u32] {
        let bytes = &self.mmap[POINT_TO_TOKENS_COUNT_HEADER_SIZE..];
        // SAFETY: header size is multiple of 4 and mmap is page-aligned.
        unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast::<u32>(), self.len) }
    }

    fn counts_mut(&mut self) -> &mut [u32] {
        let bytes = &mut self.mmap[POINT_TO_TOKENS_COUNT_HEADER_SIZE..];
        // SAFETY: header size is multiple of 4 and mmap is page-aligned.
        unsafe { std::slice::from_raw_parts_mut(bytes.as_mut_ptr().cast::<u32>(), self.len) }
    }

    pub fn get(&self, idx: usize) -> Option<usize> {
        self.counts()
            .get(idx)
            .copied()
            .map(u32::from_le)
            .map(|v| v as usize)
    }

    pub fn set_zero(&mut self, idx: usize) -> bool {
        let Some(slot) = self.counts_mut().get_mut(idx) else {
            return false;
        };
        *slot = 0u32.to_le();
        true
    }

    pub fn to_vec(&self) -> Vec<usize> {
        self.counts()
            .iter()
            .copied()
            .map(u32::from_le)
            .map(|v| v as usize)
            .collect()
    }

    pub fn populate(&self) -> std::io::Result<()> {
        use memory::madvise::Madviseable as _;
        self.mmap.populate();
        Ok(())
    }
}

pub struct MmapInvertedIndex {
    pub(in crate::index::field_index::full_text_index) path: PathBuf,
    pub(in crate::index::field_index::full_text_index) storage: Storage,
    /// Number of points which are not deleted
    pub(in crate::index::field_index::full_text_index) active_points_count: usize,
    is_on_disk: bool,
}

pub(in crate::index::field_index::full_text_index) struct Storage {
    pub(in crate::index::field_index::full_text_index) postings: MmapPostingsEnum,
    pub(in crate::index::field_index::full_text_index) vocab: MmapHashMap<str, TokenId>,
    pub(in crate::index::field_index::full_text_index) point_to_tokens_count: PointToTokensCount,
    pub(in crate::index::field_index::full_text_index) deleted_points:
        MmapBitSliceBufferedUpdateWrapper,
}

impl MmapInvertedIndex {
    pub fn create(path: PathBuf, inverted_index: &ImmutableInvertedIndex) -> OperationResult<()> {
        let ImmutableInvertedIndex {
            postings,
            vocab,
            point_to_tokens_count,
            points_count: _,
        } = inverted_index;

        debug_assert_eq!(vocab.len(), postings.len());

        let postings_path = path.join(POSTINGS_FILE);
        let vocab_path = path.join(VOCAB_FILE);
        let point_to_tokens_count_path = path.join(POINT_TO_TOKENS_COUNT_FILE);
        let deleted_points_path = path.join(DELETED_POINTS_FILE);

        match postings {
            ImmutablePostings::Ids(postings) => MmapPostings::create(postings_path, postings)?,
            ImmutablePostings::WithPositions(postings) => {
                MmapPostings::create(postings_path, postings)?
            }
        }

        // Currently MmapHashMap maps str -> [u32], but we only need to map str -> u32.
        // TODO: Consider making another mmap structure for this case.
        MmapHashMap::<str, TokenId>::create(
            &vocab_path,
            vocab.iter().map(|(k, v)| (k.as_str(), std::iter::once(*v))),
        )?;

        // Save point_to_tokens_count, separated into a bitslice for None values and a slice for actual values
        //
        // None values are represented as deleted in the bitslice
        let deleted_bitslice: BitVec = point_to_tokens_count
            .iter()
            .map(|count| *count == 0)
            .collect();
        MmapBitSlice::create(&deleted_points_path, &deleted_bitslice)?;

        // The actual values go in the slice
        let point_to_tokens_count_iter = point_to_tokens_count.iter().copied();
        PointToTokensCount::create(&point_to_tokens_count_path, point_to_tokens_count_iter)?;

        Ok(())
    }

    pub fn open(
        path: PathBuf,
        populate: bool,
        has_positions: bool,
    ) -> OperationResult<Option<Self>> {
        let postings_path = path.join(POSTINGS_FILE);
        let vocab_path = path.join(VOCAB_FILE);
        let point_to_tokens_count_path = path.join(POINT_TO_TOKENS_COUNT_FILE);
        let deleted_points_path = path.join(DELETED_POINTS_FILE);

        // If postings don't exist, assume the index doesn't exist on disk
        if !postings_path.is_file() {
            return Ok(None);
        }

        let postings = match has_positions {
            false => MmapPostingsEnum::Ids(MmapPostings::<()>::open(&postings_path, populate)?),
            true => MmapPostingsEnum::WithPositions(MmapPostings::<Positions>::open(
                &postings_path,
                populate,
            )?),
        };
        let vocab = MmapHashMap::<str, TokenId>::open(&vocab_path, false)?;

        let point_to_tokens_count = PointToTokensCount::open(&point_to_tokens_count_path, populate)?;

        let deleted =
            mmap_ops::open_write_mmap(&deleted_points_path, AdviceSetting::Global, populate)?;
        let deleted = MmapBitSlice::from(deleted, 0);

        let num_deleted_points = deleted.count_ones();
        let deleted_points = MmapBitSliceBufferedUpdateWrapper::new(deleted);
        let points_count = point_to_tokens_count.len() - num_deleted_points;

        Ok(Some(Self {
            path,
            storage: Storage {
                postings,
                vocab,
                point_to_tokens_count,
                deleted_points,
            },
            active_points_count: points_count,
            is_on_disk: !populate,
        }))
    }

    pub(super) fn iter_vocab(&self) -> impl Iterator<Item = (&str, TokenId)> + '_ {
        // unwrap safety: we know that each token points to a token id.
        self.storage.vocab.iter().filter_map(|(k, v)| {
            v.first()
                .copied()
                .map(TokenId::from_le)
                .map(|token_id| (k, token_id))
        })
    }

    /// Returns whether the point id is valid and active.
    pub fn is_active(&self, point_id: PointOffsetType) -> bool {
        let is_deleted = self
            .storage
            .deleted_points
            .get(point_id as usize)
            .unwrap_or(true);
        !is_deleted
    }

    /// Iterate over point ids whose documents contain all given tokens
    pub fn filter_has_all<'a>(
        &'a self,
        tokens: TokenSet,
    ) -> Box<dyn Iterator<Item = PointOffsetType> + 'a> {
        // in case of mmap immutable index, deleted points are still in the postings
        let filter = move |idx| self.is_active(idx);

        fn intersection<'a, V: MmapPostingValue>(
            postings: &'a MmapPostings<V>,
            tokens: TokenSet,
            filter: impl Fn(u32) -> bool + 'a,
        ) -> Box<dyn Iterator<Item = PointOffsetType> + 'a> {
            let postings_opt: Option<Vec<_>> = tokens
                .tokens()
                .iter()
                .map(|&token_id| postings.get(token_id))
                .collect();

            let Some(posting_readers) = postings_opt else {
                // There are unseen tokens -> no matches
                return Box::new(std::iter::empty());
            };

            if posting_readers.is_empty() {
                // Empty request -> no matches
                return Box::new(std::iter::empty());
            }

            Box::new(intersect_compressed_postings_iterator(
                posting_readers,
                filter,
            ))
        }

        match &self.storage.postings {
            MmapPostingsEnum::Ids(postings) => intersection(postings, tokens, filter),
            MmapPostingsEnum::WithPositions(postings) => intersection(postings, tokens, filter),
        }
    }

    /// Iterate over point ids whose documents contain at least one of the given tokens
    fn filter_has_any<'a>(
        &'a self,
        tokens: TokenSet,
    ) -> impl Iterator<Item = PointOffsetType> + 'a {
        // in case of immutable index, deleted documents are still in the postings
        let is_active = move |idx| self.is_active(idx);

        fn merge<'a, V: MmapPostingValue>(
            postings: &'a MmapPostings<V>,
            tokens: TokenSet,
            is_active: impl Fn(PointOffsetType) -> bool + 'a,
        ) -> impl Iterator<Item = PointOffsetType> + 'a {
            let postings: Vec<_> = tokens
                .tokens()
                .iter()
                .filter_map(|&token_id| postings.get(token_id))
                .collect();

            // Query must not be empty
            if postings.is_empty() {
                return Either::Left(std::iter::empty());
            };

            Either::Right(merge_compressed_postings_iterator(postings, is_active))
        }

        match &self.storage.postings {
            MmapPostingsEnum::Ids(postings) => Either::Left(merge(postings, tokens, is_active)),
            MmapPostingsEnum::WithPositions(postings) => {
                Either::Right(merge(postings, tokens, is_active))
            }
        }
    }

    fn check_has_subset(&self, tokens: &TokenSet, point_id: PointOffsetType) -> bool {
        // check non-empty query
        if tokens.is_empty() {
            return false;
        }

        // check presence of the document
        if self.values_is_empty(point_id) {
            return false;
        }

        fn check_intersection<V: MmapPostingValue>(
            postings: &MmapPostings<V>,
            tokens: &TokenSet,
            point_id: PointOffsetType,
        ) -> bool {
            // Check that all tokens are in document
            tokens.tokens().iter().all(|query_token| {
                postings
                    .get(*query_token)
                    // unwrap safety: all tokens exist in the vocabulary, otherwise there'd be no query tokens
                    .unwrap()
                    .visitor()
                    .contains(point_id)
            })
        }

        match &self.storage.postings {
            MmapPostingsEnum::Ids(postings) => check_intersection(postings, tokens, point_id),
            MmapPostingsEnum::WithPositions(postings) => {
                check_intersection(postings, tokens, point_id)
            }
        }
    }

    fn check_has_any(&self, tokens: &TokenSet, point_id: PointOffsetType) -> bool {
        if tokens.is_empty() {
            return false;
        }

        // check presence of the document
        if self.values_is_empty(point_id) {
            return false;
        }

        fn check_any<V: MmapPostingValue>(
            postings: &MmapPostings<V>,
            tokens: &TokenSet,
            point_id: PointOffsetType,
        ) -> bool {
            // Check that at least one token is in document
            tokens.tokens().iter().any(|token_id| {
                let posting_list = postings.get(*token_id).unwrap();
                posting_list.visitor().contains(point_id)
            })
        }

        match &self.storage.postings {
            MmapPostingsEnum::Ids(postings) => check_any(postings, tokens, point_id),
            MmapPostingsEnum::WithPositions(postings) => check_any(postings, tokens, point_id),
        }
    }

    /// Iterate over point ids whose documents contain all given tokens in the same order they are provided
    pub fn filter_has_phrase<'a>(
        &'a self,
        phrase: Document,
    ) -> impl Iterator<Item = PointOffsetType> + 'a {
        // in case of mmap immutable index, deleted points are still in the postings
        let is_active = move |idx| self.is_active(idx);

        match &self.storage.postings {
            MmapPostingsEnum::WithPositions(postings) => {
                Either::Right(intersect_compressed_postings_phrase_iterator(
                    phrase,
                    |token_id| postings.get(*token_id),
                    is_active,
                ))
            }
            // cannot do phrase matching if there's no positional information
            MmapPostingsEnum::Ids(_postings) => Either::Left(std::iter::empty()),
        }
    }

    pub fn check_has_phrase(&self, phrase: &Document, point_id: PointOffsetType) -> bool {
        // in case of mmap immutable index, deleted points are still in the postings
        if !self.is_active(point_id) {
            return false;
        }

        match &self.storage.postings {
            MmapPostingsEnum::WithPositions(postings) => {
                check_compressed_postings_phrase(phrase, point_id, |token_id| {
                    postings.get(*token_id)
                })
            }
            // cannot do phrase matching if there's no positional information
            MmapPostingsEnum::Ids(_postings) => false,
        }
    }

    pub fn files(&self) -> Vec<PathBuf> {
        vec![
            self.path.join(POSTINGS_FILE),
            self.path.join(VOCAB_FILE),
            self.path.join(POINT_TO_TOKENS_COUNT_FILE),
            self.path.join(DELETED_POINTS_FILE),
        ]
    }

    pub fn immutable_files(&self) -> Vec<PathBuf> {
        vec![
            self.path.join(POSTINGS_FILE),
            self.path.join(VOCAB_FILE),
            self.path.join(POINT_TO_TOKENS_COUNT_FILE),
        ]
    }

    pub fn flusher(&self) -> Flusher {
        self.storage.deleted_points.flusher()
    }

    pub fn is_on_disk(&self) -> bool {
        self.is_on_disk
    }

    /// Populate all pages in the mmap.
    /// Block until all pages are populated.
    pub fn populate(&self) -> OperationResult<()> {
        self.storage.postings.populate();
        self.storage.vocab.populate()?;
        self.storage.point_to_tokens_count.populate()?;
        Ok(())
    }

    /// Drop disk cache.
    pub fn clear_cache(&self) -> OperationResult<()> {
        let files = self.files();
        for file in files {
            clear_disk_cache(&file)?;
        }

        Ok(())
    }
}

impl InvertedIndex for MmapInvertedIndex {
    fn get_vocab_mut(&mut self) -> &mut HashMap<String, TokenId> {
        unreachable!("MmapInvertedIndex does not support mutable operations")
    }

    fn index_tokens(
        &mut self,
        _idx: PointOffsetType,
        _tokens: super::TokenSet,
        _hw_counter: &HardwareCounterCell,
    ) -> OperationResult<()> {
        Err(OperationError::service_error(
            "Can't add values to mmap immutable text index",
        ))
    }

    fn index_document(
        &mut self,
        _idx: PointOffsetType,
        _document: Document,
        _hw_counter: &HardwareCounterCell,
    ) -> OperationResult<()> {
        Err(OperationError::service_error(
            "Can't add values to mmap immutable text index",
        ))
    }

    fn remove(&mut self, idx: PointOffsetType) -> bool {
        let Some(is_deleted) = self.storage.deleted_points.get(idx as usize) else {
            return false; // Never existed
        };

        if is_deleted {
            return false; // Already removed
        }

        self.storage.deleted_points.set(idx as usize, true);
        if self
            .storage
            .point_to_tokens_count
            .set_zero(idx as usize)
        {
            // `deleted_points`'s length can be larger than `point_to_tokens_count`'s length.
            // Only if the index is within bounds of `point_to_tokens_count`, we decrement the active points count.
            self.active_points_count -= 1;
        }

        true
    }

    fn filter<'a>(
        &'a self,
        query: ParsedQuery,
        _hw_counter: &HardwareCounterCell,
    ) -> Box<dyn Iterator<Item = PointOffsetType> + 'a> {
        match query {
            ParsedQuery::AllTokens(tokens) => self.filter_has_all(tokens),
            ParsedQuery::Phrase(phrase) => Box::new(self.filter_has_phrase(phrase)),
            ParsedQuery::AnyTokens(tokens) => Box::new(self.filter_has_any(tokens)),
        }
    }

    fn get_posting_len(
        &self,
        token_id: TokenId,
        _hw_counter: &HardwareCounterCell,
    ) -> Option<usize> {
        self.storage.postings.posting_len(token_id)
    }

    fn vocab_with_postings_len_iter(&self) -> impl Iterator<Item = (&str, usize)> + '_ {
        self.iter_vocab().filter_map(move |(token, token_id)| {
            self.storage
                .postings
                .posting_len(token_id)
                .map(|posting_len| (token, posting_len))
        })
    }

    fn check_match(&self, parsed_query: &ParsedQuery, point_id: PointOffsetType) -> bool {
        match parsed_query {
            ParsedQuery::AllTokens(tokens) => self.check_has_subset(tokens, point_id),
            ParsedQuery::Phrase(phrase) => self.check_has_phrase(phrase, point_id),
            ParsedQuery::AnyTokens(tokens) => self.check_has_any(tokens, point_id),
        }
    }

    fn values_is_empty(&self, point_id: PointOffsetType) -> bool {
        if self
            .storage
            .deleted_points
            .get(point_id as usize)
            .unwrap_or(true)
        {
            return true;
        }
        self.storage
            .point_to_tokens_count
            .get(point_id as usize)
            .map(|count| count == 0)
            // if the point does not exist, it is considered empty
            .unwrap_or(true)
    }

    fn values_count(&self, point_id: PointOffsetType) -> usize {
        if self
            .storage
            .deleted_points
            .get(point_id as usize)
            .unwrap_or(true)
        {
            return 0;
        }

        self.storage
            .point_to_tokens_count
            .get(point_id as usize)
            // if the point does not exist, it is considered empty
            .unwrap_or(0)
    }

    fn points_count(&self) -> usize {
        self.active_points_count
    }

    fn get_token_id(&self, token: &str, hw_counter: &HardwareCounterCell) -> Option<TokenId> {
        if self.is_on_disk {
            hw_counter.payload_index_io_read_counter().incr_delta(
                READ_ENTRY_OVERHEAD + size_of::<TokenId>(), // Avoid check overhead and assume token is always read
            );
        }

        self.storage
            .vocab
            .get(token)
            .ok()
            .flatten()
            .and_then(<[TokenId]>::first)
            .copied()
            .map(TokenId::from_le)
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write as _;

    use super::{LegacyEndian, PointToTokensCount, POINT_TO_TOKENS_COUNT_HEADER_SIZE};

    #[test]
    fn test_point_to_tokens_count_migrates_legacy_le_and_be() {
        fn write_legacy(path: &std::path::Path, endian: LegacyEndian, values: &[usize]) {
            let mut f = std::fs::File::create(path).expect("create legacy file");
            for &v in values {
                match std::mem::size_of::<usize>() {
                    8 => {
                        let raw = v as u64;
                        let bytes = match endian {
                            LegacyEndian::Little => raw.to_le_bytes(),
                            LegacyEndian::Big => raw.to_be_bytes(),
                        };
                        f.write_all(&bytes).expect("write legacy word");
                    }
                    4 => {
                        let raw = v as u32;
                        let bytes = match endian {
                            LegacyEndian::Little => raw.to_le_bytes(),
                            LegacyEndian::Big => raw.to_be_bytes(),
                        };
                        f.write_all(&bytes).expect("write legacy word");
                    }
                    other => panic!("unsupported usize size: {other}"),
                }
            }
        }

        let values: Vec<usize> = vec![0, 1, 5, 42, 255, 1024, 65_535];

        for endian in [LegacyEndian::Little, LegacyEndian::Big] {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("point_to_tokens_count.dat");

            write_legacy(&path, endian, &values);

            let opened = PointToTokensCount::open(&path, false).expect("open migrated");
            assert_eq!(opened.len(), values.len());
            for (i, &expected) in values.iter().enumerate() {
                assert_eq!(opened.get(i), Some(expected));
            }

            let bytes = std::fs::read(&path).expect("read migrated file");
            assert!(bytes.starts_with(b"pttc"), "missing new-format magic");
            assert_eq!(
                bytes.len(),
                POINT_TO_TOKENS_COUNT_HEADER_SIZE + values.len() * std::mem::size_of::<u32>()
            );

            // Verify canonical u32 LE encoding on disk.
            for (i, &expected) in values.iter().enumerate() {
                let off = POINT_TO_TOKENS_COUNT_HEADER_SIZE + i * std::mem::size_of::<u32>();
                let got = u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()) as usize;
                assert_eq!(got, expected);
            }
        }
    }
}
