#[cfg(not(target_os = "windows"))]
mod prof;

mod fixture;

use std::hint::black_box;
use std::time::Duration;

use common::types::PointOffsetType;
use criterion::{criterion_group, criterion_main, Criterion};
use rand::rngs::StdRng;
use rand::SeedableRng;
use segment::fixtures::index_fixtures::{random_vector, TestRawScorerProducer};
use segment::index::hnsw_index::graph_layers::SearchAlgorithm;
use segment::index::hnsw_index::graph_layers_builder::GraphLayersBuilder;
use segment::index::hnsw_index::HnswM;
use segment::spaces::simple::CosineMetric;
use segment::types::Distance;
use segment::vector_storage::DEFAULT_STOPPED;

const BUILD_NUM_VECTORS: usize = 20_000;
const SEARCH_NUM_VECTORS: usize = 100_000;
const DIM: usize = 32;
const M: usize = 16;
const TOP: usize = 10;
const EF_CONSTRUCT: usize = 64;
const EF: usize = 64;
const USE_HEURISTIC: bool = true;
const QUERY_COUNT: usize = 32;

fn benchmark_hnsw_build_smoke(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(42);
    let vector_holder =
        TestRawScorerProducer::new(DIM, Distance::Cosine, BUILD_NUM_VECTORS, false, &mut rng);

    let mut group = c.benchmark_group("hnsw-smoke-build");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(10));
    group.bench_function("plain-build", |b| {
        b.iter(|| {
            let mut rng = StdRng::seed_from_u64(42);
            let mut graph_layers_builder = GraphLayersBuilder::new(
                BUILD_NUM_VECTORS,
                HnswM::new2(M),
                EF_CONSTRUCT,
                10,
                USE_HEURISTIC,
            );

            for idx in 0..(BUILD_NUM_VECTORS as PointOffsetType) {
                let level = graph_layers_builder.get_random_layer(&mut rng);
                graph_layers_builder.set_levels(idx, level);
                graph_layers_builder.link_new_point(idx, vector_holder.internal_scorer(idx));
            }
            black_box(graph_layers_builder);
        })
    });
    group.finish();
}

fn benchmark_hnsw_search_smoke(c: &mut Criterion) {
    let (vector_holder, mut graph_layers) = fixture::make_cached_graph::<CosineMetric>(
        SEARCH_NUM_VECTORS,
        DIM,
        M,
        EF_CONSTRUCT,
        USE_HEURISTIC,
    );

    let mut rng = StdRng::seed_from_u64(24);
    let queries: Vec<_> = (0..QUERY_COUNT)
        .map(|_| random_vector(&mut rng, DIM))
        .collect();

    let mut group = c.benchmark_group("hnsw-smoke-search");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(10));

    let mut query_idx = 0usize;
    group.bench_function("plain-search", |b| {
        b.iter(|| {
            let query = queries[query_idx % queries.len()].clone();
            query_idx += 1;

            let scorer = vector_holder.scorer(query);
            black_box(
                graph_layers
                    .search(
                        TOP,
                        EF,
                        SearchAlgorithm::Hnsw,
                        scorer,
                        None,
                        &DEFAULT_STOPPED,
                    )
                    .unwrap(),
            );
        })
    });

    graph_layers.compress_ram();

    let mut query_idx = 0usize;
    group.bench_function("compressed-search", |b| {
        b.iter(|| {
            let query = queries[query_idx % queries.len()].clone();
            query_idx += 1;

            let scorer = vector_holder.scorer(query);
            black_box(
                graph_layers
                    .search(
                        TOP,
                        EF,
                        SearchAlgorithm::Hnsw,
                        scorer,
                        None,
                        &DEFAULT_STOPPED,
                    )
                    .unwrap(),
            );
        })
    });

    group.finish();
}

#[cfg(not(target_os = "windows"))]
criterion_group! {
    name = benches;
    config = Criterion::default().with_profiler(prof::FlamegraphProfiler::new(100));
    targets = benchmark_hnsw_build_smoke, benchmark_hnsw_search_smoke
}

#[cfg(target_os = "windows")]
criterion_group! {
    name = benches;
    config = Criterion::default();
    targets = benchmark_hnsw_build_smoke, benchmark_hnsw_search_smoke
}

criterion_main!(benches);
