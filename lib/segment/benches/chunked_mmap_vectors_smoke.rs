#[cfg(not(target_os = "windows"))]
mod prof;

use std::hint::black_box;
use std::path::Path;
use std::time::Duration;

use common::counter::hardware_counter::HardwareCounterCell;
use criterion::{Criterion, criterion_group, criterion_main};
use memory::madvise::AdviceSetting;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use segment::data_types::vectors::VectorElementType;
use segment::vector_storage::{ChunkedMmapVectors, Random};
use tempfile::TempDir;

const DIM: usize = 32;
const NUM_VECTORS: usize = 200_000;
const QUERY_INDICES: usize = 65_536;
const READS_PER_ITER: usize = 1_024;
const BATCH_SIZE: usize = 64;
const BATCHES_PER_ITER: usize = 64;

fn build_storage(path: &Path) -> ChunkedMmapVectors<VectorElementType> {
    let hw_counter = HardwareCounterCell::new();

    {
        let mut storage: ChunkedMmapVectors<VectorElementType> =
            ChunkedMmapVectors::open(path, DIM, AdviceSetting::Global, Some(false))
                .expect("open chunked mmap vectors (write)");

        let mut vector = vec![0.0f32; DIM];
        for i in 0..NUM_VECTORS {
            vector[0] = i as f32;
            storage.push(&vector, &hw_counter).expect("push vector");
        }
        storage.flusher()().expect("flush vectors");
    }

    ChunkedMmapVectors::open(path, DIM, AdviceSetting::Global, Some(false))
        .expect("open chunked mmap vectors (read)")
}

fn make_query_indices() -> Vec<usize> {
    let mut rng = StdRng::seed_from_u64(42);
    (0..QUERY_INDICES)
        .map(|_| rng.random_range(0..NUM_VECTORS))
        .collect()
}

fn make_batches(indices: &[usize]) -> Vec<[usize; BATCH_SIZE]> {
    indices
        .chunks_exact(BATCH_SIZE)
        .map(|chunk| {
            let mut batch = [0usize; BATCH_SIZE];
            batch.copy_from_slice(chunk);
            batch
        })
        .collect()
}

fn benchmark_chunked_mmap_vectors_read_smoke(c: &mut Criterion) {
    let tmp = TempDir::new().expect("create temp dir");
    let storage_path = tmp.path().join("vectors");
    let storage = build_storage(&storage_path);
    assert_eq!(storage.len(), NUM_VECTORS);

    let indices = make_query_indices();
    let batches = make_batches(&indices);
    assert!(!batches.is_empty());

    let mut group = c.benchmark_group("chunked-mmap-vectors-smoke");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(10));

    let mut cursor = 0usize;
    group.bench_function("random-get", |b| {
        b.iter(|| {
            let mut sum = 0.0f32;
            for _ in 0..READS_PER_ITER {
                let idx = indices[cursor % indices.len()];
                cursor += 1;
                let v = storage.get::<Random>(idx).expect("vector exists");
                sum += v[0];
            }
            black_box(sum);
        })
    });

    let mut cursor = 0usize;
    group.bench_function("batch-for_each_in_batch", |b| {
        b.iter(|| {
            let mut sum = 0.0f32;
            for _ in 0..BATCHES_PER_ITER {
                let batch = &batches[cursor % batches.len()];
                cursor += 1;
                storage.for_each_in_batch(batch, |_, v| {
                    sum += v[0];
                });
            }
            black_box(sum);
        })
    });

    group.finish();
}

#[cfg(not(target_os = "windows"))]
criterion_group! {
    name = benches;
    config = Criterion::default().with_profiler(prof::FlamegraphProfiler::new(100));
    targets = benchmark_chunked_mmap_vectors_read_smoke
}

#[cfg(target_os = "windows")]
criterion_group! {
    name = benches;
    config = Criterion::default();
    targets = benchmark_chunked_mmap_vectors_read_smoke
}

criterion_main!(benches);
