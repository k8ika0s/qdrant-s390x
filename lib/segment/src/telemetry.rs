use schemars::JsonSchema;
use serde::Serialize;

use crate::common::anonymize::Anonymize;
use crate::common::operation_time_statistics::OperationDurationStatistics;
use crate::types::{SegmentConfig, SegmentInfo, VectorNameBuf};

#[derive(Serialize, Clone, Debug, JsonSchema, Anonymize)]
pub struct SegmentTelemetry {
    pub info: SegmentInfo,
    pub config: SegmentConfig,
    pub vector_index_searches: Vec<VectorIndexSearchesTelemetry>,
    pub payload_field_indices: Vec<PayloadIndexTelemetry>,
}

#[derive(Serialize, Clone, Debug, JsonSchema, Anonymize)]
pub struct PayloadIndexTelemetry {
    #[anonymize(value = None)]
    pub field_name: Option<String>,

    #[anonymize(false)]
    pub index_type: &'static str,

    /// The amount of values indexed for all points.
    pub points_values_count: usize,

    /// The amount of points that have at least one value indexed.
    pub points_count: usize,

    #[serde(skip_serializing_if = "Option::is_none")]
    #[anonymize(false)]
    pub histogram_bucket_size: Option<usize>,
}

impl PayloadIndexTelemetry {
    pub fn set_name(mut self, name: String) -> Self {
        self.field_name = Some(name);
        self
    }
}

#[derive(Serialize, Clone, Debug, JsonSchema, Anonymize, Default)]
pub struct VectorIndexSearchesTelemetry {
    #[serde(skip_serializing_if = "Option::is_none")]
    #[anonymize(value = None)]
    pub index_name: Option<VectorNameBuf>,

    #[serde(skip_serializing_if = "OperationDurationStatistics::is_empty")]
    pub unfiltered_plain: OperationDurationStatistics,

    #[serde(skip_serializing_if = "OperationDurationStatistics::is_empty")]
    pub unfiltered_hnsw: OperationDurationStatistics,

    #[serde(skip_serializing_if = "OperationDurationStatistics::is_empty")]
    pub unfiltered_sparse: OperationDurationStatistics,

    #[serde(skip_serializing_if = "OperationDurationStatistics::is_empty")]
    pub filtered_plain: OperationDurationStatistics,

    #[serde(skip_serializing_if = "OperationDurationStatistics::is_empty")]
    pub filtered_small_cardinality: OperationDurationStatistics,

    #[serde(skip_serializing_if = "OperationDurationStatistics::is_empty")]
    pub filtered_large_cardinality: OperationDurationStatistics,

    #[serde(skip_serializing_if = "OperationDurationStatistics::is_empty")]
    pub filtered_exact: OperationDurationStatistics,

    #[serde(skip_serializing_if = "OperationDurationStatistics::is_empty")]
    pub filtered_sparse: OperationDurationStatistics,

    #[serde(skip_serializing_if = "OperationDurationStatistics::is_empty")]
    pub unfiltered_exact: OperationDurationStatistics,
}

#[derive(Serialize, Clone, Debug, JsonSchema, Anonymize)]
pub struct PersistenceCompatibilityTelemetry {
    pub format_versions: PersistenceFormatVersionsTelemetry,
    #[serde(skip_serializing_if = "PersistenceMigrationCountersTelemetry::is_empty")]
    pub migration_counters: PersistenceMigrationCountersTelemetry,
}

#[derive(Serialize, Clone, Debug, JsonSchema, Anonymize)]
pub struct PersistenceFormatVersionsTelemetry {
    #[anonymize(false)]
    pub hnsw_graph_links_plain: u64,
    #[anonymize(false)]
    pub hnsw_graph_links_compressed: u64,
    #[anonymize(false)]
    pub hnsw_graph_links_compressed_legacy: u64,
    #[anonymize(false)]
    pub hnsw_graph_links_compressed_with_vectors: u64,
    #[anonymize(false)]
    pub hnsw_graph_links_compressed_with_vectors_legacy: u64,
    #[anonymize(false)]
    pub quantization_scalar_u8_metadata: u32,
    #[anonymize(false)]
    pub quantization_binary_metadata: u32,
}

#[derive(Serialize, Clone, Debug, Default, JsonSchema, Anonymize)]
pub struct PersistenceMigrationCountersTelemetry {
    #[anonymize(false)]
    pub hnsw_legacy_plain_big_endian_fallback_loads: u64,
    #[anonymize(false)]
    pub hnsw_legacy_compressed_big_endian_fallback_loads: u64,
    #[anonymize(false)]
    pub hnsw_legacy_compressed_with_vectors_big_endian_fallback_loads: u64,
    #[anonymize(false)]
    pub sparse_legacy_index_filename_migrations: u64,
}

impl PersistenceMigrationCountersTelemetry {
    fn is_empty(&self) -> bool {
        self.hnsw_legacy_plain_big_endian_fallback_loads == 0
            && self.hnsw_legacy_compressed_big_endian_fallback_loads == 0
            && self.hnsw_legacy_compressed_with_vectors_big_endian_fallback_loads == 0
            && self.sparse_legacy_index_filename_migrations == 0
    }
}

pub fn collect_persistence_compatibility_telemetry() -> PersistenceCompatibilityTelemetry {
    PersistenceCompatibilityTelemetry {
        format_versions: PersistenceFormatVersionsTelemetry {
            // Versions are reported by dedicated format slices (U11/U12). Keep a stable
            // shape here so telemetry consumers can safely parse this struct.
            hnsw_graph_links_plain: 0,
            hnsw_graph_links_compressed: 0,
            hnsw_graph_links_compressed_legacy: 0,
            hnsw_graph_links_compressed_with_vectors: 0,
            hnsw_graph_links_compressed_with_vectors_legacy: 0,
            quantization_scalar_u8_metadata: 0,
            quantization_binary_metadata: 0,
        },
        migration_counters: PersistenceMigrationCountersTelemetry::default(),
    }
}
