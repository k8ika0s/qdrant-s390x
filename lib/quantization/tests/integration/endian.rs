#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicBool;

    use common::counter::hardware_counter::HardwareCounterCell;
    use quantization::encoded_storage::EncodedStorage;
    use quantization::encoded_storage::{TestEncodedStorage, TestEncodedStorageBuilder};
    use quantization::encoded_vectors::{DistanceType, EncodedVectors, VectorParameters};
    use quantization::encoded_vectors_binary::{EncodedVectorsBin, Encoding, QueryEncoding};
    use rand::{Rng, SeedableRng};

    #[test]
    fn test_binary_u128_storage_words_are_little_endian_on_disk() {
        let vector_dim = 1usize;
        let vectors_count = 1usize;

        let data: Vec<Vec<f32>> = vec![vec![1.0f32]; vectors_count];

        let quantized_vector_size =
            EncodedVectorsBin::<u128, TestEncodedStorage>::get_quantized_vector_size_from_params(
                vector_dim,
                Encoding::OneBit,
            );
        let encoded = EncodedVectorsBin::<u128, _>::encode(
            data.iter(),
            TestEncodedStorageBuilder::new(None, quantized_vector_size),
            &VectorParameters {
                dim: vector_dim,
                deprecated_count: None,
                distance_type: DistanceType::Dot,
                invert: false,
            },
            Encoding::OneBit,
            QueryEncoding::SameAsStorage,
            None,
            &AtomicBool::new(false),
        )
        .unwrap();

        // For dim=1 and OneBit encoding, the first word is exactly `1` (bit 0 set).
        let stored = encoded.storage().get_vector_data(0);
        assert_eq!(stored, 1u128.to_le_bytes().as_slice());

        // Also ensure scoring stays correct with the canonical on-disk representation.
        let query = vec![1.0f32];
        let query_encoded = encoded.encode_query(&query);
        let counter = HardwareCounterCell::new();
        let score = encoded.score_point(&query_encoded, 0, &counter);
        assert!((score - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    #[cfg(target_endian = "big")]
    fn test_binary_v1_legacy_native_words_load_and_score_on_be() {
        use std::fs;

        use tempfile::Builder;

        let vector_dim = 1usize;

        let quantized_vector_size =
            EncodedVectorsBin::<u128, TestEncodedStorage>::get_quantized_vector_size_from_params(
                vector_dim,
                Encoding::OneBit,
            );

        let dir = Builder::new().prefix("bq_v1_legacy").tempdir().unwrap();
        let data_path = dir.path().join("data.bin");
        let meta_path = dir.path().join("meta.json");

        // v1 legacy on BE stored words in native endian; for u128=1 this is `to_be_bytes()`.
        fs::write(&data_path, 1u128.to_be_bytes()).unwrap();

        let meta = serde_json::json!({
            "format_version": 1,
            "vector_parameters": {
                "dim": vector_dim,
                "distance_type": "Dot",
                "invert": false
            },
            "encoding": "OneBit",
            "query_encoding": "SameAsStorage"
        });
        fs::write(&meta_path, serde_json::to_vec(&meta).unwrap()).unwrap();

        let storage = TestEncodedStorage::from_file(&data_path, quantized_vector_size).unwrap();
        let encoded = EncodedVectorsBin::<u128, _>::load(storage, &meta_path).unwrap();

        let stored = encoded.storage().get_vector_data(0);
        assert_eq!(stored, 1u128.to_be_bytes().as_slice());

        let query = vec![1.0f32];
        let query_encoded = encoded.encode_query(&query);
        let counter = HardwareCounterCell::new();
        let score = encoded.score_point(&query_encoded, 0, &counter);
        assert!((score - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_binary_xor_popcnt_invariant_under_byte_swaps() {
        fn xor_popcnt_u128(v1: &[u128], v2: &[u128]) -> usize {
            v1.iter()
                .zip(v2)
                .map(|(a, b)| (a ^ b).count_ones() as usize)
                .sum()
        }

        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        for _ in 0..1024 {
            let mut v1: Vec<u128> = (0..8).map(|_| rng.random()).collect();
            let mut v2: Vec<u128> = (0..8).map(|_| rng.random()).collect();

            let base = xor_popcnt_u128(&v1, &v2);

            for w in &mut v1 {
                *w = w.swap_bytes();
            }
            for w in &mut v2 {
                *w = w.swap_bytes();
            }

            let swapped = xor_popcnt_u128(&v1, &v2);
            assert_eq!(base, swapped);
        }
    }
}
