#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicBool;

    use common::counter::hardware_counter::HardwareCounterCell;
    use quantization::encoded_storage::EncodedStorage;
    use quantization::encoded_storage::{TestEncodedStorage, TestEncodedStorageBuilder};
    use quantization::encoded_vectors::{DistanceType, EncodedVectors, VectorParameters};
    use quantization::encoded_vectors_binary::{EncodedVectorsBin, Encoding, QueryEncoding};

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
}
