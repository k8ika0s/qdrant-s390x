use std::path::{Path, PathBuf};

use common::counter::conditioned_counter::ConditionedCounter;
use common::types::PointOffsetType;
use memmap2::Mmap;
use memory::fadvise::clear_disk_cache;
use memory::madvise::{AdviceSetting, Madviseable};
use memory::mmap_ops::{create_and_ensure_length, open_write_mmap};
use ordered_float::OrderedFloat;
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

use crate::common::operation_error::{OperationError, OperationResult};
use crate::types::{FloatPayloadType, GeoPoint, IntPayloadType, UuidIntType};

const POINT_TO_VALUES_PATH: &str = "point_to_values.bin";
const NOT_ENOUGHT_BYTES_ERROR_MESSAGE: &str = "Not enough bytes to operate with memmapped file `point_to_values.bin`. Is the storage corrupted?";
const PADDING_SIZE: usize = 4096;

/// Trait for values that can be stored in memmapped file. It's used in `MmapPointToValues` to store values.
pub trait MmapValue {
    /// Lifetime `'a` is required to define lifetime for `&'a str` case
    type Referenced<'a>: Sized + Clone;

    fn mmapped_size(value: Self::Referenced<'_>) -> usize;

    fn read_from_mmap(bytes: &[u8]) -> Option<Self::Referenced<'_>>;

    fn write_to_mmap(value: Self::Referenced<'_>, bytes: &mut [u8]) -> Option<()>;

    /// In-place migration helper: byteswap one legacy big-endian encoded value into canonical LE.
    /// Returns the size (in bytes) of the encoded value so the caller can advance.
    fn swap_legacy_be_value_in_place(bytes: &mut [u8]) -> Option<usize>;

    fn from_referenced<'a>(value: &'a Self::Referenced<'_>) -> &'a Self;

    fn as_referenced(&self) -> Self::Referenced<'_>;
}

#[cfg(target_endian = "little")]
impl MmapValue for IntPayloadType {
    type Referenced<'a> = &'a Self;

    fn mmapped_size(_value: Self::Referenced<'_>) -> usize {
        std::mem::size_of::<Self>()
    }

    fn read_from_mmap(bytes: &[u8]) -> Option<Self::Referenced<'_>> {
        Some(Self::ref_from_prefix(bytes).ok()?.0)
    }

    fn write_to_mmap(value: Self::Referenced<'_>, bytes: &mut [u8]) -> Option<()> {
        bytes
            .get_mut(..std::mem::size_of::<Self>())?
            .copy_from_slice(&value.to_le_bytes());
        Some(())
    }

    fn swap_legacy_be_value_in_place(bytes: &mut [u8]) -> Option<usize> {
        let size = std::mem::size_of::<Self>();
        bytes.get_mut(..size)?.reverse();
        Some(size)
    }

    fn from_referenced<'a>(value: &'a Self::Referenced<'_>) -> &'a Self {
        value
    }

    fn as_referenced(&self) -> Self::Referenced<'_> {
        self
    }
}

#[cfg(target_endian = "big")]
impl MmapValue for IntPayloadType {
    type Referenced<'a> = Self;

    fn mmapped_size(_value: Self) -> usize {
        std::mem::size_of::<Self>()
    }

    fn read_from_mmap(bytes: &[u8]) -> Option<Self> {
        let raw: [u8; 8] = bytes.get(..8)?.try_into().ok()?;
        Some(Self::from_le_bytes(raw))
    }

    fn write_to_mmap(value: Self, bytes: &mut [u8]) -> Option<()> {
        bytes
            .get_mut(..std::mem::size_of::<Self>())?
            .copy_from_slice(&value.to_le_bytes());
        Some(())
    }

    fn swap_legacy_be_value_in_place(bytes: &mut [u8]) -> Option<usize> {
        let size = std::mem::size_of::<Self>();
        bytes.get_mut(..size)?.reverse();
        Some(size)
    }

    fn from_referenced<'a>(value: &'a Self) -> &'a Self {
        value
    }

    fn as_referenced(&self) -> Self::Referenced<'_> {
        *self
    }
}

impl MmapValue for FloatPayloadType {
    type Referenced<'a> = Self;

    fn mmapped_size(_value: Self) -> usize {
        std::mem::size_of::<Self>()
    }

    fn read_from_mmap(bytes: &[u8]) -> Option<Self> {
        let raw: [u8; 8] = bytes.get(..8)?.try_into().ok()?;
        Some(Self::from_bits(u64::from_le_bytes(raw)))
    }

    fn write_to_mmap(value: Self, bytes: &mut [u8]) -> Option<()> {
        bytes
            .get_mut(..std::mem::size_of::<Self>())?
            .copy_from_slice(&value.to_bits().to_le_bytes());
        Some(())
    }

    fn swap_legacy_be_value_in_place(bytes: &mut [u8]) -> Option<usize> {
        let size = std::mem::size_of::<Self>();
        bytes.get_mut(..size)?.reverse();
        Some(size)
    }

    fn from_referenced<'a>(value: &'a Self::Referenced<'_>) -> &'a Self {
        value
    }

    fn as_referenced(&self) -> Self::Referenced<'_> {
        *self
    }
}

#[cfg(target_endian = "little")]
impl MmapValue for UuidIntType {
    type Referenced<'a> = &'a Self;

    fn mmapped_size(_value: Self::Referenced<'_>) -> usize {
        std::mem::size_of::<Self>()
    }

    fn read_from_mmap(bytes: &[u8]) -> Option<Self::Referenced<'_>> {
        Some(Self::ref_from_prefix(bytes).ok()?.0)
    }

    fn write_to_mmap(value: Self::Referenced<'_>, bytes: &mut [u8]) -> Option<()> {
        bytes
            .get_mut(..std::mem::size_of::<Self>())?
            .copy_from_slice(&value.to_le_bytes());
        Some(())
    }

    fn swap_legacy_be_value_in_place(bytes: &mut [u8]) -> Option<usize> {
        let size = std::mem::size_of::<Self>();
        bytes.get_mut(..size)?.reverse();
        Some(size)
    }

    fn from_referenced<'a>(value: &'a Self::Referenced<'_>) -> &'a Self {
        value
    }

    fn as_referenced(&self) -> Self::Referenced<'_> {
        self
    }
}

#[cfg(target_endian = "big")]
impl MmapValue for UuidIntType {
    type Referenced<'a> = Self;

    fn mmapped_size(_value: Self) -> usize {
        std::mem::size_of::<Self>()
    }

    fn read_from_mmap(bytes: &[u8]) -> Option<Self> {
        let raw: [u8; 16] = bytes.get(..16)?.try_into().ok()?;
        Some(Self::from_le_bytes(raw))
    }

    fn write_to_mmap(value: Self, bytes: &mut [u8]) -> Option<()> {
        bytes
            .get_mut(..std::mem::size_of::<Self>())?
            .copy_from_slice(&value.to_le_bytes());
        Some(())
    }

    fn swap_legacy_be_value_in_place(bytes: &mut [u8]) -> Option<usize> {
        let size = std::mem::size_of::<Self>();
        bytes.get_mut(..size)?.reverse();
        Some(size)
    }

    fn from_referenced<'a>(value: &'a Self) -> &'a Self {
        value
    }

    fn as_referenced(&self) -> Self::Referenced<'_> {
        *self
    }
}

impl MmapValue for GeoPoint {
    type Referenced<'a> = Self;

    fn mmapped_size(_value: Self) -> usize {
        2 * std::mem::size_of::<f64>()
    }

    fn read_from_mmap(bytes: &[u8]) -> Option<Self> {
        let lon_raw: [u8; 8] = bytes.get(..8)?.try_into().ok()?;
        let lat_raw: [u8; 8] = bytes.get(8..16)?.try_into().ok()?;
        let lon = f64::from_bits(u64::from_le_bytes(lon_raw));
        let lat = f64::from_bits(u64::from_le_bytes(lat_raw));

        Some(Self {
            lon: OrderedFloat(lon),
            lat: OrderedFloat(lat),
        })
    }

    fn write_to_mmap(value: Self, bytes: &mut [u8]) -> Option<()> {
        bytes
            .get_mut(..8)?
            .copy_from_slice(&value.lon.0.to_bits().to_le_bytes());
        bytes
            .get_mut(8..16)?
            .copy_from_slice(&value.lat.0.to_bits().to_le_bytes());
        Some(())
    }

    fn swap_legacy_be_value_in_place(bytes: &mut [u8]) -> Option<usize> {
        bytes.get_mut(..8)?.reverse();
        bytes.get_mut(8..16)?.reverse();
        Some(2 * std::mem::size_of::<f64>())
    }

    fn from_referenced<'a>(value: &'a Self::Referenced<'_>) -> &'a Self {
        value
    }

    fn as_referenced(&self) -> Self::Referenced<'_> {
        *self
    }
}

impl MmapValue for str {
    type Referenced<'a> = &'a str;

    fn mmapped_size(value: &str) -> usize {
        value.len() + std::mem::size_of::<u32>()
    }

    fn read_from_mmap(bytes: &[u8]) -> Option<&str> {
        let size_bytes: [u8; 4] = bytes.get(..4)?.try_into().ok()?;
        let size = u32::from_le_bytes(size_bytes);
        let bytes = bytes.get(std::mem::size_of::<u32>()..)?;
        let bytes = bytes.get(..size as usize)?;
        std::str::from_utf8(bytes).ok()
    }

    fn write_to_mmap(value: &str, bytes: &mut [u8]) -> Option<()> {
        bytes
            .get_mut(..4)?
            .copy_from_slice(&(value.len() as u32).to_le_bytes());
        bytes
            .get_mut(std::mem::size_of::<u32>()..std::mem::size_of::<u32>() + value.len())?
            .copy_from_slice(value.as_bytes());
        Some(())
    }

    fn swap_legacy_be_value_in_place(bytes: &mut [u8]) -> Option<usize> {
        let len_bytes: [u8; 4] = bytes.get(..4)?.try_into().ok()?;
        let len = u32::from_be_bytes(len_bytes) as usize;
        bytes.get_mut(..4)?.reverse();
        Some(std::mem::size_of::<u32>() + len)
    }

    fn from_referenced<'a>(value: &'a Self::Referenced<'_>) -> &'a Self {
        value
    }

    fn as_referenced(&self) -> Self::Referenced<'_> {
        self
    }
}

/// Flattened memmapped points-to-values map
/// It's an analogue of `Vec<Vec<N>>` but in memmapped file.
/// This structure doesn't support adding new values, only removing.
/// It's used in mmap field indices like `MmapMapIndex`, `MmapNumericIndex`, etc to store points-to-values map.
/// This structure is not generic to avoid boxing lifetimes for `&str` values.
pub struct MmapPointToValues<T: MmapValue + ?Sized> {
    file_name: PathBuf,
    mmap: Mmap,
    header: Header,
    phantom: std::marker::PhantomData<T>,
}

/// Memory and IO overhead of accessing mmap index.
pub const MMAP_PTV_ACCESS_OVERHEAD: usize = size_of::<MmapRangeDisk>();

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, FromBytes, Immutable, IntoBytes, KnownLayout)]
struct MmapRangeDisk {
    start: u64,
    count: u64,
}

#[derive(Copy, Clone, Debug, Default)]
struct MmapRange {
    start: u64,
    count: u64,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, FromBytes, Immutable, IntoBytes, KnownLayout)]
struct HeaderDisk {
    ranges_start: u64,
    points_count: u64,
}

#[derive(Copy, Clone, Debug)]
struct Header {
    ranges_start: u64,
    points_count: u64,
}

impl HeaderDisk {
    fn decode_le(&self) -> Header {
        Header {
            ranges_start: u64::from_le(self.ranges_start),
            points_count: u64::from_le(self.points_count),
        }
    }

    fn decode_be(&self) -> Header {
        Header {
            ranges_start: u64::from_be(self.ranges_start),
            points_count: u64::from_be(self.points_count),
        }
    }
}

impl MmapRangeDisk {
    fn decode_le(&self) -> MmapRange {
        MmapRange {
            start: u64::from_le(self.start),
            count: u64::from_le(self.count),
        }
    }

    fn decode_be(&self) -> MmapRange {
        MmapRange {
            start: u64::from_be(self.start),
            count: u64::from_be(self.count),
        }
    }
}
impl<T: MmapValue + ?Sized> MmapPointToValues<T> {
    pub fn from_iter<'a>(
        path: &Path,
        iter: impl Iterator<Item = (PointOffsetType, impl Iterator<Item = T::Referenced<'a>>)> + Clone,
    ) -> OperationResult<Self> {
        // calculate file size
        let points_count = iter
            .clone()
            .map(|(point_id, _)| (point_id + 1) as usize)
            .max()
            .unwrap_or(0);
        let ranges_size = points_count * std::mem::size_of::<MmapRangeDisk>();
        let values_size = iter
            .clone()
            .map(|v| v.1.map(|v| T::mmapped_size(v)).sum::<usize>())
            .sum::<usize>();
        let file_size = PADDING_SIZE + ranges_size + values_size;

        // create new file and mmap
        let file_name = path.join(POINT_TO_VALUES_PATH);
        create_and_ensure_length(&file_name, file_size)?;
        let mut mmap = open_write_mmap(&file_name, AdviceSetting::Global, false)?;

        // fill mmap file data
        let header = Header {
            ranges_start: PADDING_SIZE as u64,
            points_count: points_count as u64,
        };
        let header_disk = HeaderDisk {
            ranges_start: header.ranges_start.to_le(),
            points_count: header.points_count.to_le(),
        };
        header_disk
            .write_to_prefix(mmap.as_mut())
            .map_err(|_| OperationError::service_error(NOT_ENOUGHT_BYTES_ERROR_MESSAGE))?;

        // counter for values offset
        let mut point_values_offset = header.ranges_start as usize + ranges_size;
        for (point_id, values) in iter {
            let start = point_values_offset;
            let mut values_count = 0;
            for value in values {
                values_count += 1;
                let bytes = mmap.get_mut(point_values_offset..).ok_or_else(|| {
                    OperationError::service_error(NOT_ENOUGHT_BYTES_ERROR_MESSAGE)
                })?;
                T::write_to_mmap(value.clone(), bytes).ok_or_else(|| {
                    OperationError::service_error(NOT_ENOUGHT_BYTES_ERROR_MESSAGE)
                })?;
                point_values_offset += T::mmapped_size(value);
            }

            let range = MmapRange {
                start: start as u64,
                count: values_count as u64,
            };
            let range_disk = MmapRangeDisk {
                start: range.start.to_le(),
                count: range.count.to_le(),
            };
            mmap.get_mut(
                header.ranges_start as usize
                    + point_id as usize * std::mem::size_of::<MmapRangeDisk>()..,
            )
            .and_then(|bytes| range_disk.write_to_prefix(bytes).ok())
            .ok_or_else(|| OperationError::service_error(NOT_ENOUGHT_BYTES_ERROR_MESSAGE))?;
        }

        mmap.flush()?;
        Ok(Self {
            file_name,
            mmap: mmap.make_read_only()?,
            header,
            phantom: std::marker::PhantomData,
        })
    }

    pub fn open(path: &Path, populate: bool) -> OperationResult<Self> {
        let file_name = path.join(POINT_TO_VALUES_PATH);
        let mut mmap = open_write_mmap(&file_name, AdviceSetting::Global, populate)?;

        let (header_disk, _) = HeaderDisk::read_from_prefix(mmap.as_ref()).map_err(|_| {
            OperationError::InconsistentStorage {
                description: NOT_ENOUGHT_BYTES_ERROR_MESSAGE.to_owned(),
            }
        })?;

        // Canonical encoding is little-endian. Legacy BE files (created on s390x before
        // canonicalization) are migrated in-place by byte-swapping all multi-byte fields.
        let header = {
            let header_le = header_disk.decode_le();
            if header_le.ranges_start == PADDING_SIZE as u64 {
                header_le
            } else {
                let header_be = header_disk.decode_be();
                if header_be.ranges_start != PADDING_SIZE as u64 {
                    return Err(OperationError::InconsistentStorage {
                        description: NOT_ENOUGHT_BYTES_ERROR_MESSAGE.to_owned(),
                    });
                }

                migrate_legacy_be_in_place::<T>(mmap.as_mut(), header_be)?;
                mmap.flush()?;

                let (header_disk, _) =
                    HeaderDisk::read_from_prefix(mmap.as_ref()).map_err(|_| {
                        OperationError::InconsistentStorage {
                            description: NOT_ENOUGHT_BYTES_ERROR_MESSAGE.to_owned(),
                        }
                    })?;

                let header_le = header_disk.decode_le();
                if header_le.ranges_start != PADDING_SIZE as u64 {
                    return Err(OperationError::InconsistentStorage {
                        description: NOT_ENOUGHT_BYTES_ERROR_MESSAGE.to_owned(),
                    });
                }
                header_le
            }
        };

        Ok(Self {
            file_name,
            mmap: mmap.make_read_only()?,
            header,
            phantom: std::marker::PhantomData,
        })
    }

    pub fn files(&self) -> Vec<PathBuf> {
        vec![self.file_name.clone()]
    }

    pub fn immutable_files(&self) -> Vec<PathBuf> {
        // `MmapPointToValues` is immutable
        vec![self.file_name.clone()]
    }

    pub fn check_values_any(
        &self,
        point_id: PointOffsetType,
        check_fn: impl Fn(T::Referenced<'_>) -> bool,
        hw_counter: &ConditionedCounter,
    ) -> bool {
        let hw_cell = hw_counter.payload_index_io_read_counter();

        // Measure IO overhead of `self.get_range()`
        hw_cell.incr_delta(MMAP_PTV_ACCESS_OVERHEAD);

        self.get_range(point_id)
            .map(|range| {
                let mut value_offset = range.start as usize;
                for _ in 0..range.count {
                    let bytes = self.mmap.get(value_offset..).unwrap();
                    let value = T::read_from_mmap(bytes).unwrap();
                    let mmap_size = T::mmapped_size(value.clone());
                    hw_cell.incr_delta(mmap_size);
                    if check_fn(value) {
                        return true;
                    }
                    value_offset += mmap_size;
                }
                false
            })
            .unwrap_or(false)
    }

    pub fn get_values<'a>(
        &'a self,
        point_id: PointOffsetType,
    ) -> Option<impl Iterator<Item = T::Referenced<'a>> + 'a> {
        // first, get range of values for point
        let range = self.get_range(point_id)?;

        // second, define iteration step for values
        // iteration step gets remainder range from memmapped file and returns left range
        let bytes: &[u8] = self.mmap.as_ref();
        let read_value = move |range: MmapRange| -> Option<(T::Referenced<'a>, MmapRange)> {
            if range.count > 0 {
                let bytes = bytes.get(range.start as usize..)?;
                T::read_from_mmap(bytes).map(|value| {
                    let range = MmapRange {
                        start: range.start + T::mmapped_size(value.clone()) as u64,
                        count: range.count - 1,
                    };
                    (value, range)
                })
            } else {
                None
            }
        };

        // finally, return iterator
        Some(
            std::iter::successors(read_value(range), move |range| read_value(range.1))
                .map(|(value, _)| value),
        )
    }

    pub fn get_values_count(&self, point_id: PointOffsetType) -> Option<usize> {
        self.get_range(point_id).map(|range| range.count as usize)
    }

    pub fn len(&self) -> usize {
        self.header.points_count as usize
    }

    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.header.points_count == 0
    }

    fn get_range(&self, point_id: PointOffsetType) -> Option<MmapRange> {
        if point_id < self.header.points_count as PointOffsetType {
            let range_offset = (self.header.ranges_start as usize)
                + (point_id as usize) * std::mem::size_of::<MmapRangeDisk>();
            let (range_disk, _) =
                MmapRangeDisk::read_from_prefix(self.mmap.get(range_offset..)?).ok()?;
            Some(range_disk.decode_le())
        } else {
            None
        }
    }

    /// Populate all pages in the mmap.
    /// Block until all pages are populated.
    pub fn populate(&self) {
        self.mmap.populate();
    }

    /// Drop disk cache.
    pub fn clear_cache(&self) -> OperationResult<()> {
        clear_disk_cache(&self.file_name)?;
        Ok(())
    }

    pub fn iter(
        &self,
    ) -> impl Iterator<
        Item = (
            PointOffsetType,
            Option<impl Iterator<Item = T::Referenced<'_>> + '_>,
        ),
    > + Clone {
        (0..self.len() as PointOffsetType).map(|idx| (idx, self.get_values(idx)))
    }
}

fn migrate_legacy_be_in_place<T: MmapValue + ?Sized>(
    mmap: &mut [u8],
    header_be: Header,
) -> OperationResult<()> {
    if header_be.ranges_start != PADDING_SIZE as u64 {
        return Err(OperationError::InconsistentStorage {
            description: NOT_ENOUGHT_BYTES_ERROR_MESSAGE.to_owned(),
        });
    }

    let header_size = std::mem::size_of::<HeaderDisk>();
    if mmap.len() < header_size {
        return Err(OperationError::InconsistentStorage {
            description: NOT_ENOUGHT_BYTES_ERROR_MESSAGE.to_owned(),
        });
    }

    // Swap the header fields (two u64s).
    mmap.get_mut(..8)
        .ok_or_else(|| OperationError::InconsistentStorage {
            description: NOT_ENOUGHT_BYTES_ERROR_MESSAGE.to_owned(),
        })?
        .reverse();
    mmap.get_mut(8..16)
        .ok_or_else(|| OperationError::InconsistentStorage {
            description: NOT_ENOUGHT_BYTES_ERROR_MESSAGE.to_owned(),
        })?
        .reverse();

    let points_count: usize =
        header_be
            .points_count
            .try_into()
            .map_err(|_| OperationError::InconsistentStorage {
                description: NOT_ENOUGHT_BYTES_ERROR_MESSAGE.to_owned(),
            })?;
    let ranges_start: usize =
        header_be
            .ranges_start
            .try_into()
            .map_err(|_| OperationError::InconsistentStorage {
                description: NOT_ENOUGHT_BYTES_ERROR_MESSAGE.to_owned(),
            })?;

    let range_size = std::mem::size_of::<MmapRangeDisk>();
    for point_id in 0..points_count {
        let range_offset = ranges_start + point_id * range_size;
        let range_bytes = mmap
            .get(range_offset..range_offset + range_size)
            .ok_or_else(|| OperationError::InconsistentStorage {
                description: NOT_ENOUGHT_BYTES_ERROR_MESSAGE.to_owned(),
            })?;

        let (range_disk, _) = MmapRangeDisk::read_from_prefix(range_bytes).map_err(|_| {
            OperationError::InconsistentStorage {
                description: NOT_ENOUGHT_BYTES_ERROR_MESSAGE.to_owned(),
            }
        })?;
        let range = range_disk.decode_be();
        let start = range.start;
        let count = range.count;

        // Swap the range fields (two u64s) in-place.
        mmap.get_mut(range_offset..range_offset + 8)
            .ok_or_else(|| OperationError::InconsistentStorage {
                description: NOT_ENOUGHT_BYTES_ERROR_MESSAGE.to_owned(),
            })?
            .reverse();
        mmap.get_mut(range_offset + 8..range_offset + 16)
            .ok_or_else(|| OperationError::InconsistentStorage {
                description: NOT_ENOUGHT_BYTES_ERROR_MESSAGE.to_owned(),
            })?
            .reverse();

        let mut value_offset: usize =
            start
                .try_into()
                .map_err(|_| OperationError::InconsistentStorage {
                    description: NOT_ENOUGHT_BYTES_ERROR_MESSAGE.to_owned(),
                })?;

        for _ in 0..count {
            let tail = mmap.get_mut(value_offset..).ok_or_else(|| {
                OperationError::InconsistentStorage {
                    description: NOT_ENOUGHT_BYTES_ERROR_MESSAGE.to_owned(),
                }
            })?;
            let written = T::swap_legacy_be_value_in_place(tail).ok_or_else(|| {
                OperationError::InconsistentStorage {
                    description: NOT_ENOUGHT_BYTES_ERROR_MESSAGE.to_owned(),
                }
            })?;
            value_offset = value_offset.checked_add(written).ok_or_else(|| {
                OperationError::InconsistentStorage {
                    description: NOT_ENOUGHT_BYTES_ERROR_MESSAGE.to_owned(),
                }
            })?;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use tempfile::Builder;

    use super::*;

    #[test]
    fn test_mmap_point_to_values_int_roundtrip() {
        let values: Vec<Vec<IntPayloadType>> =
            vec![vec![1, 2, 3, -4], vec![], vec![i64::MIN, i64::MAX], vec![0]];

        let dir = Builder::new()
            .prefix("mmap_point_to_values_int")
            .tempdir()
            .unwrap();
        MmapPointToValues::<IntPayloadType>::from_iter(
            dir.path(),
            values.iter().enumerate().map(|(id, values)| {
                (
                    id as PointOffsetType,
                    values.iter().map(|v| v.as_referenced()),
                )
            }),
        )
        .unwrap();
        let point_to_values = MmapPointToValues::<IntPayloadType>::open(dir.path(), false).unwrap();

        for (idx, expected) in values.iter().enumerate() {
            let got: Vec<IntPayloadType> = point_to_values
                .get_values(idx as PointOffsetType)
                .map(|it| {
                    it.map(|v| *IntPayloadType::from_referenced(&v))
                        .collect_vec()
                })
                .unwrap_or_default();
            assert_eq!(got, *expected);
        }
    }

    #[test]
    fn test_mmap_point_to_values_int_legacy_be_migrates() {
        let dir = Builder::new()
            .prefix("mmap_point_to_values_int_legacy_be")
            .tempdir()
            .unwrap();
        let path = dir.path().join(POINT_TO_VALUES_PATH);

        // points_count=2:
        // point 0 -> [11, 22]
        // point 1 -> [33]
        let points_count = 2u64;
        let ranges_start = PADDING_SIZE as u64;
        let ranges_size = (points_count as usize) * std::mem::size_of::<MmapRangeDisk>();
        let values_size = 3usize * std::mem::size_of::<IntPayloadType>();
        let file_size = PADDING_SIZE + ranges_size + values_size;

        let mut bytes = vec![0u8; file_size];

        // Header (legacy BE)
        bytes[0..8].copy_from_slice(&ranges_start.to_be_bytes());
        bytes[8..16].copy_from_slice(&points_count.to_be_bytes());

        let values_start = ranges_start as usize + ranges_size;

        // Ranges (legacy BE)
        let r0_start = values_start as u64;
        let r0_count = 2u64;
        let r1_start = (values_start + 2 * std::mem::size_of::<IntPayloadType>()) as u64;
        let r1_count = 1u64;

        let ranges_off = ranges_start as usize;
        bytes[ranges_off..ranges_off + 8].copy_from_slice(&r0_start.to_be_bytes());
        bytes[ranges_off + 8..ranges_off + 16].copy_from_slice(&r0_count.to_be_bytes());
        bytes[ranges_off + 16..ranges_off + 24].copy_from_slice(&r1_start.to_be_bytes());
        bytes[ranges_off + 24..ranges_off + 32].copy_from_slice(&r1_count.to_be_bytes());

        // Values (legacy BE)
        let mut off = values_start;
        for v in [11i64, 22, 33] {
            bytes[off..off + 8].copy_from_slice(&v.to_be_bytes());
            off += 8;
        }

        std::fs::write(&path, &bytes).unwrap();

        let point_to_values = MmapPointToValues::<IntPayloadType>::open(dir.path(), false).unwrap();
        let got0: Vec<i64> = point_to_values
            .get_values(0)
            .map(|it| {
                it.map(|v| *IntPayloadType::from_referenced(&v))
                    .collect_vec()
            })
            .unwrap_or_default();
        let got1: Vec<i64> = point_to_values
            .get_values(1)
            .map(|it| {
                it.map(|v| *IntPayloadType::from_referenced(&v))
                    .collect_vec()
            })
            .unwrap_or_default();

        assert_eq!(got0, vec![11, 22]);
        assert_eq!(got1, vec![33]);

        // Header should have been migrated in-place to canonical LE.
        let after = std::fs::read(&path).unwrap();
        assert_eq!(&after[0..8], &ranges_start.to_le_bytes());
        assert_eq!(&after[8..16], &points_count.to_le_bytes());
    }

    #[test]
    fn test_mmap_point_to_values_string_legacy_be_migrates() {
        let dir = Builder::new()
            .prefix("mmap_point_to_values_string_legacy_be")
            .tempdir()
            .unwrap();
        let path = dir.path().join(POINT_TO_VALUES_PATH);

        // points_count=2:
        // point 0 -> ["ab", "c"]
        // point 1 -> ["xyz"]
        let points_count = 2u64;
        let ranges_start = PADDING_SIZE as u64;
        let ranges_size = (points_count as usize) * std::mem::size_of::<MmapRangeDisk>();
        let values_size = (4 + 2) + (4 + 1) + (4 + 3);
        let file_size = PADDING_SIZE + ranges_size + values_size;

        let mut bytes = vec![0u8; file_size];

        // Header (legacy BE)
        bytes[0..8].copy_from_slice(&ranges_start.to_be_bytes());
        bytes[8..16].copy_from_slice(&points_count.to_be_bytes());

        let values_start = ranges_start as usize + ranges_size;

        // Ranges (legacy BE)
        let r0_start = values_start as u64;
        let r0_count = 2u64;
        let r1_start = (values_start + (4 + 2) + (4 + 1)) as u64;
        let r1_count = 1u64;

        let ranges_off = ranges_start as usize;
        bytes[ranges_off..ranges_off + 8].copy_from_slice(&r0_start.to_be_bytes());
        bytes[ranges_off + 8..ranges_off + 16].copy_from_slice(&r0_count.to_be_bytes());
        bytes[ranges_off + 16..ranges_off + 24].copy_from_slice(&r1_start.to_be_bytes());
        bytes[ranges_off + 24..ranges_off + 32].copy_from_slice(&r1_count.to_be_bytes());

        // Values (legacy BE)
        let mut off = values_start;
        for s in ["ab", "c", "xyz"] {
            let len = s.len() as u32;
            bytes[off..off + 4].copy_from_slice(&len.to_be_bytes());
            off += 4;
            bytes[off..off + s.len()].copy_from_slice(s.as_bytes());
            off += s.len();
        }

        std::fs::write(&path, &bytes).unwrap();

        let point_to_values = MmapPointToValues::<str>::open(dir.path(), false).unwrap();
        let got0: Vec<String> = point_to_values
            .get_values(0)
            .map(|it| it.map(|s| s.to_owned()).collect_vec())
            .unwrap_or_default();
        let got1: Vec<String> = point_to_values
            .get_values(1)
            .map(|it| it.map(|s| s.to_owned()).collect_vec())
            .unwrap_or_default();

        assert_eq!(got0, vec!["ab".to_owned(), "c".to_owned()]);
        assert_eq!(got1, vec!["xyz".to_owned()]);

        // Header should have been migrated in-place to canonical LE.
        let after = std::fs::read(&path).unwrap();
        assert_eq!(&after[0..8], &ranges_start.to_le_bytes());
        assert_eq!(&after[8..16], &points_count.to_le_bytes());
    }

    #[test]
    fn test_mmap_point_to_values_string() {
        let values: Vec<Vec<String>> = vec![
            vec![
                "fox".to_owned(),
                "driver".to_owned(),
                "point".to_owned(),
                "it".to_owned(),
                "box".to_owned(),
            ],
            vec![
                "alice".to_owned(),
                "red".to_owned(),
                "yellow".to_owned(),
                "blue".to_owned(),
                "apple".to_owned(),
            ],
            vec![
                "box".to_owned(),
                "qdrant".to_owned(),
                "line".to_owned(),
                "bash".to_owned(),
                "reproduction".to_owned(),
            ],
            vec![
                "list".to_owned(),
                "vitamin".to_owned(),
                "one".to_owned(),
                "two".to_owned(),
                "three".to_owned(),
            ],
            vec![
                "tree".to_owned(),
                "metallic".to_owned(),
                "ownership".to_owned(),
            ],
            vec![],
            vec!["slice".to_owned()],
            vec!["red".to_owned(), "pink".to_owned()],
        ];

        let dir = Builder::new()
            .prefix("mmap_point_to_values")
            .tempdir()
            .unwrap();
        MmapPointToValues::<str>::from_iter(
            dir.path(),
            values
                .iter()
                .enumerate()
                .map(|(id, values)| (id as PointOffsetType, values.iter().map(|s| s.as_str()))),
        )
        .unwrap();
        let point_to_values = MmapPointToValues::<str>::open(dir.path(), false).unwrap();

        for (idx, values) in values.iter().enumerate() {
            let iter = point_to_values.get_values(idx as PointOffsetType);
            let v: Vec<String> = iter
                .map(|iter| iter.map(|s: &str| s.to_owned()).collect_vec())
                .unwrap_or_default();
            assert_eq!(&v, values);
        }
    }

    #[test]
    fn test_mmap_point_to_values_geo() {
        let values: Vec<Vec<GeoPoint>> = vec![
            vec![
                GeoPoint::new_unchecked(6.0, 2.0),
                GeoPoint::new_unchecked(4.0, 3.0),
                GeoPoint::new_unchecked(2.0, 5.0),
                GeoPoint::new_unchecked(8.0, 7.0),
                GeoPoint::new_unchecked(1.0, 9.0),
            ],
            vec![
                GeoPoint::new_unchecked(8.0, 1.0),
                GeoPoint::new_unchecked(3.0, 3.0),
                GeoPoint::new_unchecked(5.0, 9.0),
                GeoPoint::new_unchecked(1.0, 8.0),
                GeoPoint::new_unchecked(7.0, 2.0),
            ],
            vec![
                GeoPoint::new_unchecked(6.0, 3.0),
                GeoPoint::new_unchecked(4.0, 4.0),
                GeoPoint::new_unchecked(3.0, 7.0),
                GeoPoint::new_unchecked(1.0, 2.0),
                GeoPoint::new_unchecked(4.0, 8.0),
            ],
            vec![
                GeoPoint::new_unchecked(1.0, 3.0),
                GeoPoint::new_unchecked(3.0, 9.0),
                GeoPoint::new_unchecked(7.0, 0.0),
            ],
            vec![],
            vec![GeoPoint::new_unchecked(8.0, 5.0)],
            vec![GeoPoint::new_unchecked(9.0, 4.0)],
        ];

        let dir = Builder::new()
            .prefix("mmap_point_to_values")
            .tempdir()
            .unwrap();
        MmapPointToValues::<GeoPoint>::from_iter(
            dir.path(),
            values
                .iter()
                .enumerate()
                .map(|(id, values)| (id as PointOffsetType, values.iter().cloned())),
        )
        .unwrap();
        let point_to_values = MmapPointToValues::<GeoPoint>::open(dir.path(), false).unwrap();

        for (idx, values) in values.iter().enumerate() {
            let iter = point_to_values.get_values(idx as PointOffsetType);
            let v: Vec<GeoPoint> = iter.map(|iter| iter.collect_vec()).unwrap_or_default();
            assert_eq!(&v, values);
        }
    }
}
