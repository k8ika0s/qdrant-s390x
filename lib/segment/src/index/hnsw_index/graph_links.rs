use std::alloc::Layout;
use std::io::Cursor;
use std::path::Path;
use std::sync::Arc;

use common::types::PointOffsetType;
use memmap2::Mmap;
use memory::madvise::{Advice, AdviceSetting, Madviseable};
use memory::mmap_ops::open_read_mmap;

use crate::common::operation_error::{OperationError, OperationResult};
use crate::index::hnsw_index::HnswM;
use crate::vector_storage::quantized::quantized_vectors::QuantizedVectors;
use crate::vector_storage::{Sequential, VectorStorageEnum};

mod header;
mod serializer;
mod view;

pub use serializer::serialize_graph_links;
pub use view::LinksIterator;
use view::{CompressionInfo, GraphLinksView, LinksWithVectorsIterator};

/*
Links data for whole graph layers.

                                    sorted
                     points:        points:
points to lvl        012345         142350
     0 -> 0
     1 -> 4    lvl4:  7       lvl4: 7
     2 -> 2    lvl3:  Z  Y    lvl3: ZY
     3 -> 2    lvl2:  abcd    lvl2: adbc
     4 -> 3    lvl1:  ABCDE   lvl1: ADBCE
     5 -> 1    lvl0: 123456   lvl0: 123456  <- lvl 0 is not sorted


lvl offset:        6       11     15     17
                   │       │      │      │
                   │       │      │      │
                   ▼       ▼      ▼      ▼
indexes:  012345   6789A   BCDE   FG     H

flatten:  123456   ADBCE   adbc   ZY     7
                   ▲ ▲ ▲   ▲ ▲    ▲      ▲
                   │ │ │   │ │    │      │
                   │ │ │   │ │    │      │
                   │ │ │   │ │    │      │
reindex:           142350  142350 142350 142350  (same for each level)


for lvl > 0:
links offset = level_offsets[level] + offsets[reindex[point_id]]
*/

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum GraphLinksFormat {
    Plain,
    Compressed,
    CompressedWithVectors,
}

/// Current persisted-format versions and legacy decode fallback counters for graph links.
#[derive(Debug, Clone, Copy, Default)]
pub struct GraphLinksCompatibilityTelemetry {
    pub plain_version: u64,
    pub compressed_version: u64,
    pub compressed_legacy_version: u64,
    pub compressed_with_vectors_version: u64,
    pub compressed_with_vectors_legacy_version: u64,
    pub fallback_decode: GraphLinksFallbackDecodeTelemetry,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct GraphLinksFallbackDecodeTelemetry {
    pub legacy_plain_big_endian_fallback_loads: u64,
    pub legacy_compressed_big_endian_fallback_loads: u64,
    pub legacy_compressed_with_vectors_big_endian_fallback_loads: u64,
}

pub fn graph_links_compatibility_telemetry() -> GraphLinksCompatibilityTelemetry {
    let fallback = view::fallback_decode_telemetry();
    GraphLinksCompatibilityTelemetry {
        plain_version: header::HEADER_VERSION_PLAIN,
        compressed_version: header::HEADER_VERSION_COMPRESSED,
        compressed_legacy_version: header::HEADER_VERSION_COMPRESSED_LEGACY,
        compressed_with_vectors_version: header::HEADER_VERSION_COMPRESSED_WITH_VECTORS,
        compressed_with_vectors_legacy_version:
            header::HEADER_VERSION_COMPRESSED_WITH_VECTORS_LEGACY,
        fallback_decode: fallback,
    }
}

/// Similar to [`GraphLinksFormat`], won't let you use `CompressedWithVectors`
/// without providing the vectors.
#[derive(Clone, Copy)]
pub enum GraphLinksFormatParam<'a> {
    Plain,
    Compressed,
    CompressedWithVectors(&'a dyn GraphLinksVectors),
}

/// This trait lets the [`serialize_graph_links`] to access vector values.
pub trait GraphLinksVectors {
    /// Base vectors will be included once per point on level 0.
    /// The layout of each vector must correspond to [`VectorLayout::base`].
    fn get_base_vector(&self, point_id: PointOffsetType) -> OperationResult<&[u8]>;

    /// Link vectors will be included for each link per point.
    /// The layout of each vector must correspond to [`VectorLayout::link`].
    fn get_link_vector(&self, point_id: PointOffsetType) -> OperationResult<&[u8]>;

    /// Get the layout of base and link vectors.
    fn vectors_layout(&self) -> GraphLinksVectorsLayout;
}

/// Layout of base and link vectors, returned by [`GraphLinksVectors::vectors_layout`].
#[derive(Copy, Clone)]
pub struct GraphLinksVectorsLayout {
    pub base: Layout,
    pub link: Layout,
}

/// A [`GraphLinksVectors`] implementation that uses real storage.
pub struct StorageGraphLinksVectors<'a> {
    vector_storage: &'a VectorStorageEnum,   // base vectors
    quantized_vectors: &'a QuantizedVectors, // link vectors
    vectors_layout: GraphLinksVectorsLayout,
}

impl<'a> StorageGraphLinksVectors<'a> {
    pub fn try_new(
        vector_storage: &'a VectorStorageEnum,
        quantized_vectors: Option<&'a QuantizedVectors>,
    ) -> Option<Self> {
        let quantized_vectors = quantized_vectors?;
        Some(Self {
            vector_storage,
            quantized_vectors,
            vectors_layout: GraphLinksVectorsLayout {
                base: vector_storage.get_vector_layout().ok()?,
                link: quantized_vectors.get_quantized_vector_layout().ok()?,
            },
        })
    }
}

impl<'a> GraphLinksVectors for StorageGraphLinksVectors<'a> {
    /// Note: uses [`Sequential`] because [`serializer::serialize_graph_links`]
    /// traverses base vectors in a sequential order.
    fn get_base_vector(&self, point_id: PointOffsetType) -> OperationResult<&[u8]> {
        self.vector_storage
            .get_vector_bytes_opt::<Sequential>(point_id)
            .ok_or_else(|| {
                OperationError::service_error(format!(
                    "Point {point_id} not found in vector storage"
                ))
            })
    }

    /// Note: unlike base vectors, link vectors are written in a random order.
    fn get_link_vector(&self, point_id: PointOffsetType) -> OperationResult<&[u8]> {
        Ok(self.quantized_vectors.get_quantized_vector(point_id))
    }

    fn vectors_layout(&self) -> GraphLinksVectorsLayout {
        self.vectors_layout
    }
}

impl GraphLinksFormat {
    /// Create the corresponding [`GraphLinksFormatParam`].
    ///
    /// # Panics
    ///
    /// Panics if `CompressedWithVectors` is selected, but `vectors` is `None`.
    #[cfg(test)]
    pub fn with_param_for_tests<'a, Q: GraphLinksVectors>(
        &self,
        vectors: Option<&'a Q>,
    ) -> GraphLinksFormatParam<'a> {
        match self {
            GraphLinksFormat::Plain => GraphLinksFormatParam::Plain,
            GraphLinksFormat::Compressed => GraphLinksFormatParam::Compressed,
            GraphLinksFormat::CompressedWithVectors => match vectors {
                Some(v) => GraphLinksFormatParam::CompressedWithVectors(v),
                None => panic!(),
            },
        }
    }

    /// Create the corresponding [`GraphLinksFormatParam`].
    ///
    /// When vectors are not available, `CompressedWithVectors` is downgraded to
    /// `Compressed`.
    pub fn with_param<'a, V: GraphLinksVectors>(
        &self,
        vectors: Option<&'a V>,
    ) -> GraphLinksFormatParam<'a> {
        match self {
            GraphLinksFormat::Plain => GraphLinksFormatParam::Plain,
            GraphLinksFormat::Compressed => GraphLinksFormatParam::Compressed,
            GraphLinksFormat::CompressedWithVectors => match vectors {
                Some(v) => GraphLinksFormatParam::CompressedWithVectors(v),
                None => GraphLinksFormatParam::Compressed,
            },
        }
    }

    pub fn is_with_vectors(&self) -> bool {
        match self {
            GraphLinksFormat::Plain | GraphLinksFormat::Compressed => false,
            GraphLinksFormat::CompressedWithVectors => true,
        }
    }
}

impl<'a> GraphLinksFormatParam<'a> {
    pub fn as_format(&self) -> GraphLinksFormat {
        match self {
            GraphLinksFormatParam::Plain => GraphLinksFormat::Plain,
            GraphLinksFormatParam::Compressed => GraphLinksFormat::Compressed,
            GraphLinksFormatParam::CompressedWithVectors(_) => {
                GraphLinksFormat::CompressedWithVectors
            }
        }
    }
}

self_cell::self_cell! {
    pub struct GraphLinks {
        owner: GraphLinksEnum,
        #[covariant]
        dependent: GraphLinksView,
    }

    impl {Debug}
}

#[derive(Debug)]
enum GraphLinksEnum {
    Ram(Vec<u8>),
    Mmap(Arc<Mmap>),
}

impl GraphLinksEnum {
    fn as_bytes(&self) -> &[u8] {
        match self {
            GraphLinksEnum::Ram(data) => data.as_slice(),
            GraphLinksEnum::Mmap(mmap) => &mmap[..],
        }
    }
}

impl GraphLinks {
    pub fn load_from_file(
        path: &Path,
        on_disk: bool,
        format: GraphLinksFormat,
    ) -> OperationResult<Self> {
        let populate = !on_disk;
        let mmap = open_read_mmap(path, AdviceSetting::Advice(Advice::Random), populate)?;
        Self::try_new(GraphLinksEnum::Mmap(Arc::new(mmap)), |x| {
            GraphLinksView::load(x.as_bytes(), format)
        })
    }

    pub fn new_from_edges(
        edges: Vec<Vec<Vec<PointOffsetType>>>,
        format_param: GraphLinksFormatParam<'_>,
        hnsw_m: HnswM,
    ) -> OperationResult<Self> {
        let mut cursor = Cursor::new(Vec::<u8>::new());
        serialize_graph_links(edges, format_param, hnsw_m, &mut cursor)?;
        let mut bytes = cursor.into_inner();
        bytes.shrink_to_fit();
        Self::try_new(GraphLinksEnum::Ram(bytes), |x| {
            GraphLinksView::load(x.as_bytes(), format_param.as_format())
        })
    }

    fn view(&self) -> &GraphLinksView<'_> {
        self.borrow_dependent()
    }

    pub fn as_bytes(&self) -> &[u8] {
        self.borrow_owner().as_bytes()
    }

    pub fn format(&self) -> GraphLinksFormat {
        match self.view().compression {
            CompressionInfo::Uncompressed { .. } => GraphLinksFormat::Plain,
            CompressionInfo::Compressed { .. } => GraphLinksFormat::Compressed,
            CompressionInfo::CompressedWithVectors { .. } => {
                GraphLinksFormat::CompressedWithVectors
            }
        }
    }

    pub fn num_points(&self) -> usize {
        self.view().reindex.len()
    }

    pub fn for_each_link(
        &self,
        point_id: PointOffsetType,
        level: usize,
        f: impl FnMut(PointOffsetType),
    ) {
        self.links(point_id, level).for_each(f);
    }

    #[inline]
    pub fn links(&self, point_id: PointOffsetType, level: usize) -> LinksIterator<'_> {
        self.view().links(point_id, level)
    }

    #[inline]
    pub fn links_empty(&self, point_id: PointOffsetType, level: usize) -> bool {
        self.view().links_empty(point_id, level)
    }

    #[inline]
    pub fn links_with_vectors(
        &self,
        point_id: PointOffsetType,
        level: usize,
    ) -> (&[u8], LinksWithVectorsIterator<'_>) {
        let (base_vector, links, vectors) = self.view().links_with_vectors(point_id, level);
        (base_vector, links.zip(vectors))
    }

    pub fn point_level(&self, point_id: PointOffsetType) -> usize {
        self.view().point_level(point_id)
    }

    /// Convert the graph links to a vector of edges, suitable for passing into
    /// [`serialize_graph_links`] or using in tests.
    pub fn to_edges(&self) -> Vec<Vec<Vec<PointOffsetType>>> {
        self.to_edges_impl(|point_id, level| self.links(point_id, level).collect())
    }

    /// Convert the graph links to a vector of edges, generic over the container type.
    pub fn to_edges_impl<Container>(
        &self,
        mut f: impl FnMut(PointOffsetType, usize) -> Container,
    ) -> Vec<Vec<Container>> {
        let mut edges = Vec::with_capacity(self.num_points());
        for point_id in 0..self.num_points() {
            let num_levels = self.point_level(point_id as PointOffsetType) + 1;
            let mut levels = Vec::with_capacity(num_levels);
            levels.extend((0..num_levels).map(|level| f(point_id as PointOffsetType, level)));
            edges.push(levels);
        }
        edges
    }

    /// Populate the disk cache with data, if applicable.
    /// This is a blocking operation.
    pub fn populate(&self) -> OperationResult<()> {
        match self.borrow_owner() {
            GraphLinksEnum::Mmap(mmap) => mmap.populate(),
            GraphLinksEnum::Ram(_) => {}
        };
        Ok(())
    }
}

/// Sort the first `m` values in `links` and return them. Used to compare stored
/// links where the order of the first `m` links is not preserved.
#[cfg(test)]
pub(super) fn normalize_links(m: usize, mut links: Vec<PointOffsetType>) -> Vec<PointOffsetType> {
    let first = links.len().min(m);
    links[..first].sort_unstable();
    links
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;
    use std::mem::size_of;

    use io::file_operations::atomic_save;
    use rand::Rng;
    use rstest::rstest;
    use tempfile::Builder;

    use super::*;
    use crate::index::hnsw_index::HnswM;

    struct TestGraphLinksVectors {
        base_vectors: Vec<Vec<u8>>,
        link_vectors: Vec<Vec<u8>>,
        vectors_layout: GraphLinksVectorsLayout,
    }

    impl TestGraphLinksVectors {
        fn new(count: usize, base_align: usize, link_align: usize) -> Self {
            let mut rng = rand::rng();
            let base_len = base_align * 7;
            let link_len = link_align * 5;
            Self {
                base_vectors: (0..count)
                    .map(|_| (0..base_len).map(|_| rng.random()).collect())
                    .collect(),
                link_vectors: (0..count)
                    .map(|_| (0..link_len).map(|_| rng.random()).collect())
                    .collect(),
                vectors_layout: GraphLinksVectorsLayout {
                    base: Layout::from_size_align(base_len, base_align).unwrap(),
                    link: Layout::from_size_align(link_len, link_align).unwrap(),
                },
            }
        }
    }

    impl GraphLinksVectors for TestGraphLinksVectors {
        fn get_base_vector(&self, point_id: PointOffsetType) -> OperationResult<&[u8]> {
            Ok(&self.base_vectors[point_id as usize])
        }

        fn get_link_vector(&self, point_id: PointOffsetType) -> OperationResult<&[u8]> {
            Ok(&self.link_vectors[point_id as usize])
        }

        fn vectors_layout(&self) -> GraphLinksVectorsLayout {
            self.vectors_layout
        }
    }

    fn random_links(
        points_count: usize,
        max_levels_count: usize,
        hnsw_m: &HnswM,
    ) -> Vec<Vec<Vec<PointOffsetType>>> {
        let mut rng = rand::rng();
        (0..points_count)
            .map(|_| {
                let levels_count = rng.random_range(1..max_levels_count);
                (0..levels_count)
                    .map(|level| {
                        let mut max_links_count = hnsw_m.level_m(level);
                        max_links_count *= 2; // Simulate additional payload links.
                        let links_count = rng.random_range(0..max_links_count);
                        (0..links_count)
                            .map(|_| rng.random_range(0..points_count) as PointOffsetType)
                            .collect()
                    })
                    .collect()
            })
            .collect()
    }

    fn check_links(
        mut left: Vec<Vec<Vec<PointOffsetType>>>,
        right: &GraphLinks,
        vectors: &Option<TestGraphLinksVectors>,
    ) {
        let mut right_links = right.to_edges_impl(|point_id, level| {
            let links: Vec<_> = if let Some(vectors) = vectors {
                let (base_vector, iter) = right.links_with_vectors(point_id, level);
                if level == 0 {
                    assert_eq!(base_vector, vectors.get_base_vector(point_id).unwrap());
                } else {
                    assert!(base_vector.is_empty());
                }
                iter.map(|(link, bytes)| {
                    assert_eq!(bytes, vectors.get_link_vector(link).unwrap());
                    link
                })
                .collect()
            } else {
                right.links(point_id, level).collect()
            };
            assert_eq!(links.is_empty(), right.links_empty(point_id, level));
            links
        });
        for links in [&mut left, &mut right_links].iter_mut() {
            links.iter_mut().for_each(|levels| {
                levels
                    .iter_mut()
                    .enumerate()
                    .for_each(|(level_idx, links)| {
                        *links = normalize_links(
                            right.view().sorted_count(level_idx),
                            std::mem::take(links),
                        );
                    })
            });
        }
        assert_eq!(left, right_links);
    }

    /// Test that random links can be saved by [`serialize_graph_links`] and
    /// loaded correctly by a [`GraphLinks`] impl.
    #[rstest]
    #[case::plain(GraphLinksFormat::Plain, 8, 8)]
    #[case::compressed(GraphLinksFormat::Compressed, 8, 8)]
    #[case::comp_vec_1_16(GraphLinksFormat::CompressedWithVectors, 1, 16)]
    #[case::comp_vec_4_1(GraphLinksFormat::CompressedWithVectors, 4, 1)]
    #[case::comp_vec_4_16(GraphLinksFormat::CompressedWithVectors, 4, 16)]
    fn test_save_load(
        #[case] format: GraphLinksFormat,
        #[case] base_align: usize,
        #[case] link_align: usize,
    ) {
        let points_count = 1000;
        let max_levels_count = 10;
        let hnsw_m = HnswM::new2(8);

        let path = Builder::new().prefix("graph_dir").tempdir().unwrap();
        let links_file = path.path().join("links.bin");
        let links = random_links(points_count, max_levels_count, &hnsw_m);

        let vectors = format
            .is_with_vectors()
            .then(|| TestGraphLinksVectors::new(points_count, base_align, link_align));

        let format_param = format.with_param_for_tests(vectors.as_ref());
        atomic_save(&links_file, |writer| {
            serialize_graph_links(links.clone(), format_param, hnsw_m, writer)
        })
        .unwrap();

        let cmp_links = GraphLinks::load_from_file(&links_file, true, format).unwrap();
        check_links(links, &cmp_links, &vectors);
    }

    #[test]
    fn test_plain_serialization_has_little_endian_versioned_header() {
        let hnsw_m = HnswM::new2(8);
        let links = vec![vec![vec![1]], vec![vec![0]]];
        let mut cursor = Cursor::new(Vec::<u8>::new());
        serialize_graph_links(links, GraphLinksFormatParam::Plain, hnsw_m, &mut cursor).unwrap();
        let bytes = cursor.into_inner();

        let version_offset = 5 * size_of::<u64>();
        let version = u64::from_le_bytes(
            bytes[version_offset..version_offset + size_of::<u64>()]
                .try_into()
                .unwrap(),
        );
        assert_eq!(version, super::header::HEADER_VERSION_PLAIN);
    }

    #[test]
    fn test_compressed_serialization_has_little_endian_versioned_header() {
        let hnsw_m = HnswM::new2(8);
        let links = vec![vec![vec![1]], vec![vec![0]]];
        let mut cursor = Cursor::new(Vec::<u8>::new());
        serialize_graph_links(
            links,
            GraphLinksFormatParam::Compressed,
            hnsw_m,
            &mut cursor,
        )
        .unwrap();
        let bytes = cursor.into_inner();

        let version_offset = size_of::<u64>();
        let version = u64::from_le_bytes(
            bytes[version_offset..version_offset + size_of::<u64>()]
                .try_into()
                .unwrap(),
        );
        assert_eq!(version, super::header::HEADER_VERSION_COMPRESSED);
    }

    #[test]
    fn test_compressed_with_vectors_serialization_has_little_endian_versioned_header() {
        let hnsw_m = HnswM::new2(8);
        let links = vec![vec![vec![1]], vec![vec![0]]];
        let vectors = TestGraphLinksVectors::new(2, 8, 8);
        let mut cursor = Cursor::new(Vec::<u8>::new());
        serialize_graph_links(
            links,
            GraphLinksFormatParam::CompressedWithVectors(&vectors),
            hnsw_m,
            &mut cursor,
        )
        .unwrap();
        let bytes = cursor.into_inner();

        let version_offset = size_of::<u64>();
        let version = u64::from_le_bytes(
            bytes[version_offset..version_offset + size_of::<u64>()]
                .try_into()
                .unwrap(),
        );
        assert_eq!(
            version,
            super::header::HEADER_VERSION_COMPRESSED_WITH_VECTORS
        );
    }

    #[test]
    fn test_load_plain_legacy_big_endian_fixture() {
        let path = Builder::new().prefix("graph_dir").tempdir().unwrap();
        let links_file = path.path().join("legacy_plain_be_links.bin");
        fs_err::write(&links_file, legacy_plain_big_endian_fixture()).unwrap();
        let before = graph_links_compatibility_telemetry()
            .fallback_decode
            .legacy_plain_big_endian_fallback_loads;

        let links = GraphLinks::load_from_file(&links_file, true, GraphLinksFormat::Plain).unwrap();

        assert_eq!(links.format(), GraphLinksFormat::Plain);
        assert_eq!(links.links(0, 0).collect::<Vec<_>>(), vec![1]);
        assert_eq!(links.links(1, 0).collect::<Vec<_>>(), vec![0]);
        let after = graph_links_compatibility_telemetry()
            .fallback_decode
            .legacy_plain_big_endian_fallback_loads;
        assert!(after > before);
    }

    #[test]
    fn test_load_compressed_legacy_big_endian_fixture() {
        let path = Builder::new().prefix("graph_dir").tempdir().unwrap();
        let links_file = path.path().join("legacy_compressed_be_links.bin");
        fs_err::write(&links_file, legacy_compressed_big_endian_fixture()).unwrap();
        let before = graph_links_compatibility_telemetry()
            .fallback_decode
            .legacy_compressed_big_endian_fallback_loads;

        let links =
            GraphLinks::load_from_file(&links_file, true, GraphLinksFormat::Compressed).unwrap();

        assert_eq!(links.format(), GraphLinksFormat::Compressed);
        assert_eq!(links.links(0, 0).collect::<Vec<_>>(), vec![1]);
        assert_eq!(links.links(1, 0).collect::<Vec<_>>(), vec![0]);
        let after = graph_links_compatibility_telemetry()
            .fallback_decode
            .legacy_compressed_big_endian_fallback_loads;
        assert!(after > before);
    }

    #[test]
    fn test_load_compressed_with_vectors_legacy_big_endian_fixture() {
        let path = Builder::new().prefix("graph_dir").tempdir().unwrap();
        let links_file = path
            .path()
            .join("legacy_compressed_with_vectors_be_links.bin");
        let links = vec![vec![vec![1]], vec![vec![0]]];
        let vectors = TestGraphLinksVectors::new(2, 8, 8);
        fs_err::write(
            &links_file,
            legacy_compressed_with_vectors_big_endian_fixture(links.clone(), &vectors),
        )
        .unwrap();
        let before = graph_links_compatibility_telemetry()
            .fallback_decode
            .legacy_compressed_with_vectors_big_endian_fallback_loads;

        let loaded =
            GraphLinks::load_from_file(&links_file, true, GraphLinksFormat::CompressedWithVectors)
                .unwrap();

        assert_eq!(loaded.format(), GraphLinksFormat::CompressedWithVectors);
        check_links(links, &loaded, &Some(vectors));
        let after = graph_links_compatibility_telemetry()
            .fallback_decode
            .legacy_compressed_with_vectors_big_endian_fallback_loads;
        assert!(after > before);
    }

    #[test]
    fn test_load_plain_rejects_zero_levels_with_points() {
        let hnsw_m = HnswM::new2(8);
        let links = vec![vec![vec![1]], vec![vec![0]]];
        let mut cursor = Cursor::new(Vec::<u8>::new());
        serialize_graph_links(links, GraphLinksFormatParam::Plain, hnsw_m, &mut cursor).unwrap();
        let mut bytes = cursor.into_inner();

        write_u64_le_at(&mut bytes, 1, 0); // levels_count

        let path = Builder::new().prefix("graph_dir").tempdir().unwrap();
        let links_file = path.path().join("invalid_plain_links.bin");
        fs_err::write(&links_file, bytes).unwrap();

        let err = GraphLinks::load_from_file(&links_file, true, GraphLinksFormat::Plain)
            .err()
            .expect("invalid plain layout must fail");
        assert!(
            err.to_string()
                .contains("Invalid plain GraphLinks level/point counts")
        );
    }

    #[test]
    fn test_load_compressed_rejects_zero_levels_with_points() {
        let hnsw_m = HnswM::new2(8);
        let links = vec![vec![vec![1]], vec![vec![0]]];
        let mut cursor = Cursor::new(Vec::<u8>::new());
        serialize_graph_links(
            links,
            GraphLinksFormatParam::Compressed,
            hnsw_m,
            &mut cursor,
        )
        .unwrap();
        let mut bytes = cursor.into_inner();

        write_u64_le_at(&mut bytes, 2, 0); // levels_count

        let path = Builder::new().prefix("graph_dir").tempdir().unwrap();
        let links_file = path.path().join("invalid_compressed_links.bin");
        fs_err::write(&links_file, bytes).unwrap();

        let err = GraphLinks::load_from_file(&links_file, true, GraphLinksFormat::Compressed)
            .err()
            .expect("invalid compressed layout must fail");
        assert!(
            err.to_string()
                .contains("Invalid compressed GraphLinks level/point counts")
        );
    }

    fn legacy_plain_big_endian_fixture() -> Vec<u8> {
        let mut bytes = Vec::new();

        // Legacy plain header (native-endian fields, no version marker).
        bytes.extend_from_slice(&2_u64.to_be_bytes()); // point_count
        bytes.extend_from_slice(&1_u64.to_be_bytes()); // levels_count
        bytes.extend_from_slice(&2_u64.to_be_bytes()); // total_neighbors_count
        bytes.extend_from_slice(&3_u64.to_be_bytes()); // total_offset_count
        bytes.extend_from_slice(&0_u64.to_be_bytes()); // offsets_padding_bytes
        bytes.extend_from_slice(&[0_u8; 24]); // legacy zero padding

        // level_offsets
        bytes.extend_from_slice(&0_u64.to_be_bytes());
        // reindex
        bytes.extend_from_slice(&0_u32.to_be_bytes());
        bytes.extend_from_slice(&1_u32.to_be_bytes());
        // neighbors
        bytes.extend_from_slice(&1_u32.to_be_bytes());
        bytes.extend_from_slice(&0_u32.to_be_bytes());
        // offsets
        bytes.extend_from_slice(&0_u64.to_be_bytes());
        bytes.extend_from_slice(&1_u64.to_be_bytes());
        bytes.extend_from_slice(&2_u64.to_be_bytes());

        bytes
    }

    fn legacy_compressed_big_endian_fixture() -> Vec<u8> {
        let hnsw_m = HnswM::new2(8);
        let links = vec![vec![vec![1]], vec![vec![0]]];
        let mut cursor = Cursor::new(Vec::<u8>::new());
        serialize_graph_links(
            links,
            GraphLinksFormatParam::Compressed,
            hnsw_m,
            &mut cursor,
        )
        .unwrap();
        let mut bytes = cursor.into_inner();

        // Mark as legacy compressed format.
        write_u64_le_at(
            &mut bytes,
            1,
            super::header::HEADER_VERSION_COMPRESSED_LEGACY,
        );

        let point_count = read_u64_le_at(&bytes, 0) as usize;
        let levels_count = read_u64_le_at(&bytes, 2) as usize;
        let header_size = size_of::<super::header::HeaderCompressed>();

        // Legacy compressed level offsets/reindex were native-endian.
        let mut pos = header_size;
        for _ in 0..levels_count {
            let value = u64::from_le_bytes(bytes[pos..pos + size_of::<u64>()].try_into().unwrap());
            bytes[pos..pos + size_of::<u64>()].copy_from_slice(&value.to_be_bytes());
            pos += size_of::<u64>();
        }
        for _ in 0..point_count {
            let value = u32::from_le_bytes(bytes[pos..pos + size_of::<u32>()].try_into().unwrap());
            bytes[pos..pos + size_of::<u32>()].copy_from_slice(&value.to_be_bytes());
            pos += size_of::<u32>();
        }

        bytes
    }

    fn legacy_compressed_with_vectors_big_endian_fixture(
        links: Vec<Vec<Vec<PointOffsetType>>>,
        vectors: &TestGraphLinksVectors,
    ) -> Vec<u8> {
        let hnsw_m = HnswM::new2(8);
        let mut cursor = Cursor::new(Vec::<u8>::new());
        serialize_graph_links(
            links,
            GraphLinksFormatParam::CompressedWithVectors(vectors),
            hnsw_m,
            &mut cursor,
        )
        .unwrap();
        let mut bytes = cursor.into_inner();

        // Mark as legacy compressed-with-vectors format.
        write_u64_le_at(
            &mut bytes,
            1,
            super::header::HEADER_VERSION_COMPRESSED_WITH_VECTORS_LEGACY,
        );

        let point_count = read_u64_le_at(&bytes, 0) as usize;
        let levels_count = read_u64_le_at(&bytes, 2) as usize;
        let header_size = size_of::<super::header::HeaderCompressedWithVectors>();

        // Legacy compressed-with-vectors level offsets/reindex were native-endian.
        let mut pos = header_size;
        for _ in 0..levels_count {
            rewrite_u64_le_to_be_at(&mut bytes, pos);
            pos += size_of::<u64>();
        }
        for _ in 0..point_count {
            rewrite_u32_le_to_be_at(&mut bytes, pos);
            pos += size_of::<u32>();
        }

        bytes
    }

    fn read_u64_le_at(bytes: &[u8], field_idx: usize) -> u64 {
        let start = field_idx * size_of::<u64>();
        let end = start + size_of::<u64>();
        u64::from_le_bytes(bytes[start..end].try_into().unwrap())
    }

    fn write_u64_le_at(bytes: &mut [u8], field_idx: usize, value: u64) {
        let start = field_idx * size_of::<u64>();
        let end = start + size_of::<u64>();
        bytes[start..end].copy_from_slice(&value.to_le_bytes());
    }

    fn rewrite_u64_le_to_be_at(bytes: &mut [u8], start: usize) {
        let end = start + size_of::<u64>();
        let value = u64::from_le_bytes(bytes[start..end].try_into().unwrap());
        bytes[start..end].copy_from_slice(&value.to_be_bytes());
    }

    fn rewrite_u32_le_to_be_at(bytes: &mut [u8], start: usize) {
        let end = start + size_of::<u32>();
        let value = u32::from_le_bytes(bytes[start..end].try_into().unwrap());
        bytes[start..end].copy_from_slice(&value.to_be_bytes());
    }

    #[rstest]
    #[case::uncompressed(GraphLinksFormat::Plain)]
    #[case::compressed(GraphLinksFormat::Compressed)]
    #[case::compressed_with_vectors(GraphLinksFormat::CompressedWithVectors)]
    fn test_graph_links_construction(#[case] format: GraphLinksFormat) {
        let hnsw_m = HnswM::new2(8);

        let vectors = format
            .is_with_vectors()
            .then(|| TestGraphLinksVectors::new(100, 8, 8));

        let check = |links: Vec<Vec<Vec<PointOffsetType>>>| {
            let format_param = format.with_param_for_tests(vectors.as_ref());
            let cmp_links =
                GraphLinks::new_from_edges(links.clone(), format_param, hnsw_m).unwrap();
            check_links(links, &cmp_links, &vectors);
        };

        // no points
        check(vec![]);

        // 2 points without any links
        check(vec![vec![vec![]], vec![vec![]]]);

        // one link at level 0
        check(vec![vec![vec![1]], vec![vec![0]]]);

        // 3 levels with no links at second level
        check(vec![
            vec![vec![1, 2]],
            vec![vec![0, 2], vec![], vec![2]],
            vec![vec![0, 1], vec![], vec![1]],
        ]);

        // 3 levels with no links at last level
        check(vec![
            vec![vec![1, 2], vec![2], vec![]],
            vec![vec![0, 2], vec![1], vec![]],
            vec![vec![0, 1]],
        ]);

        // 4 levels with random nonexistent links
        check(vec![
            vec![vec![1, 2, 5, 6]],
            vec![vec![0, 2, 7, 8], vec![], vec![34, 45, 10]],
            vec![vec![0, 1, 1, 2], vec![3, 5, 9], vec![9, 8], vec![9], vec![]],
            vec![vec![0, 1, 5, 6], vec![1, 5, 0]],
            vec![vec![0, 1, 9, 18], vec![1, 5, 6], vec![5], vec![9]],
        ]);

        // fully random links
        check(random_links(100, 10, &hnsw_m));
    }
}
