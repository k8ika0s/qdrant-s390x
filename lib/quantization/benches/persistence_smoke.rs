use std::hint::black_box;
use std::sync::atomic::AtomicBool;
use std::time::Duration;

use common::counter::hardware_counter::HardwareCounterCell;
use criterion::{Criterion, criterion_group, criterion_main};
use quantization::encoded_storage::{TestEncodedStorage, TestEncodedStorageBuilder};
use quantization::encoded_vectors::{DistanceType, EncodedVectors, VectorParameters};
use quantization::encoded_vectors_binary::{EncodedVectorsBin, Encoding, QueryEncoding};
use quantization::encoded_vectors_u8::{EncodedVectorsU8, ScalarQuantizationMethod};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const DEFAULT_VECTOR_COUNT: usize = 8_192;
const DEFAULT_DIM: usize = 128;
const DEFAULT_SAMPLE_SIZE: usize = 10;
const DEFAULT_WARMUP_SECS: u64 = 1;
const DEFAULT_MEASUREMENT_SECS: u64 = 5;

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(default)
}

fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(default)
}

fn build_vectors(count: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..count)
        .map(|_| {
            (0..dim)
                .map(|_| rng.random_range(-1.0f32..1.0f32))
                .collect::<Vec<f32>>()
        })
        .collect()
}

fn configure_group(group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>) {
    let sample_size = env_usize("QDRANT_QBENCH_SAMPLE_SIZE", DEFAULT_SAMPLE_SIZE).max(10);
    let warmup_secs = env_u64("QDRANT_QBENCH_WARMUP_SECS", DEFAULT_WARMUP_SECS).max(1);
    let measurement_secs =
        env_u64("QDRANT_QBENCH_MEASUREMENT_SECS", DEFAULT_MEASUREMENT_SECS).max(1);

    group.sample_size(sample_size);
    group.warm_up_time(Duration::from_secs(warmup_secs));
    group.measurement_time(Duration::from_secs(measurement_secs));
}

fn scalar_u8_persistence_smoke(c: &mut Criterion) {
    let vectors_count = env_usize("QDRANT_QBENCH_VECTORS", DEFAULT_VECTOR_COUNT);
    let dim = env_usize("QDRANT_QBENCH_DIM", DEFAULT_DIM);
    let vectors = build_vectors(vectors_count, dim, 42);

    let vector_params = VectorParameters {
        dim,
        deprecated_count: None,
        distance_type: DistanceType::Dot,
        invert: false,
    };
    let quantized_vector_size =
        EncodedVectorsU8::<TestEncodedStorage>::get_quantized_vector_size(&vector_params);

    let mut group = c.benchmark_group("quantization_persistence_smoke/scalar_u8");
    configure_group(&mut group);

    group.bench_function("encode", |b| {
        b.iter(|| {
            let encoded = EncodedVectorsU8::encode(
                vectors.iter().map(Vec::as_slice),
                TestEncodedStorageBuilder::new(None, quantized_vector_size),
                &vector_params,
                vectors_count,
                None,
                ScalarQuantizationMethod::Int8,
                None,
                &AtomicBool::new(false),
            )
            .expect("scalar quantization encode should succeed");
            black_box(encoded.quantized_vector_size());
        });
    });

    let encoded = EncodedVectorsU8::encode(
        vectors.iter().map(Vec::as_slice),
        TestEncodedStorageBuilder::new(None, quantized_vector_size),
        &vector_params,
        vectors_count,
        None,
        ScalarQuantizationMethod::Int8,
        None,
        &AtomicBool::new(false),
    )
    .expect("scalar quantization encode should succeed");
    let query = vectors[vectors_count / 2].clone();
    let encoded_query = encoded.encode_query(&query);
    let hw_counter = HardwareCounterCell::new();

    group.bench_function("score_scan", |b| {
        b.iter(|| {
            let mut best_score = f32::NEG_INFINITY;
            let mut best_id = 0u32;
            for i in 0..vectors_count as u32 {
                let score = encoded.score_point(&encoded_query, i, &hw_counter);
                if score > best_score {
                    best_score = score;
                    best_id = i;
                }
            }
            black_box((best_score, best_id));
        });
    });

    group.finish();
}

fn binary_u8_persistence_smoke(c: &mut Criterion) {
    let vectors_count = env_usize("QDRANT_QBENCH_VECTORS", DEFAULT_VECTOR_COUNT);
    let dim = env_usize("QDRANT_QBENCH_DIM", DEFAULT_DIM);
    let vectors = build_vectors(vectors_count, dim, 84);

    let vector_params = VectorParameters {
        dim,
        deprecated_count: None,
        distance_type: DistanceType::Dot,
        invert: false,
    };

    let quantized_vector_size =
        EncodedVectorsBin::<u8, TestEncodedStorage>::get_quantized_vector_size_from_params(
            dim,
            Encoding::OneBit,
        );

    let mut group = c.benchmark_group("quantization_persistence_smoke/binary_u8");
    configure_group(&mut group);

    group.bench_function("encode", |b| {
        b.iter(|| {
            let encoded = EncodedVectorsBin::<u8, _>::encode(
                vectors.iter().map(Vec::as_slice),
                TestEncodedStorageBuilder::new(None, quantized_vector_size),
                &vector_params,
                Encoding::OneBit,
                QueryEncoding::SameAsStorage,
                None,
                &AtomicBool::new(false),
            )
            .expect("binary quantization encode should succeed");
            black_box(encoded.quantized_vector_size());
        });
    });

    let encoded = EncodedVectorsBin::<u8, _>::encode(
        vectors.iter().map(Vec::as_slice),
        TestEncodedStorageBuilder::new(None, quantized_vector_size),
        &vector_params,
        Encoding::OneBit,
        QueryEncoding::SameAsStorage,
        None,
        &AtomicBool::new(false),
    )
    .expect("binary quantization encode should succeed");
    let query = vectors[vectors_count / 3].clone();
    let encoded_query = encoded.encode_query(&query);
    let hw_counter = HardwareCounterCell::new();

    group.bench_function("score_scan", |b| {
        b.iter(|| {
            let mut best_score = f32::NEG_INFINITY;
            let mut best_id = 0u32;
            for i in 0..vectors_count as u32 {
                let score = encoded.score_point(&encoded_query, i, &hw_counter);
                if score > best_score {
                    best_score = score;
                    best_id = i;
                }
            }
            black_box((best_score, best_id));
        });
    });

    group.finish();
}

criterion_group!(benches, scalar_u8_persistence_smoke, binary_u8_persistence_smoke);
criterion_main!(benches);
