use half::f16;

pub trait MmapEndianConvertible: Copy + Sized {
    fn to_le_storage(self) -> Self;
    fn from_le_storage(stored: Self) -> Self;
}

macro_rules! impl_identity_mmap_endian {
    ($ty:ty) => {
        impl MmapEndianConvertible for $ty {
            #[inline]
            fn to_le_storage(self) -> Self {
                self
            }

            #[inline]
            fn from_le_storage(stored: Self) -> Self {
                stored
            }
        }
    };
}

macro_rules! impl_int_mmap_endian {
    ($ty:ty) => {
        impl MmapEndianConvertible for $ty {
            #[inline]
            fn to_le_storage(self) -> Self {
                <$ty>::to_le(self)
            }

            #[inline]
            fn from_le_storage(stored: Self) -> Self {
                <$ty>::from_le(stored)
            }
        }
    };
}

impl_identity_mmap_endian!(u8);
impl_int_mmap_endian!(u16);
impl_int_mmap_endian!(u32);
impl_int_mmap_endian!(u64);

impl MmapEndianConvertible for f32 {
    #[inline]
    fn to_le_storage(self) -> Self {
        f32::from_bits(self.to_bits().to_le())
    }

    #[inline]
    fn from_le_storage(stored: Self) -> Self {
        f32::from_bits(u32::from_le(stored.to_bits()))
    }
}

impl MmapEndianConvertible for f16 {
    #[inline]
    fn to_le_storage(self) -> Self {
        f16::from_bits(self.to_bits().to_le())
    }

    #[inline]
    fn from_le_storage(stored: Self) -> Self {
        f16::from_bits(u16::from_le(stored.to_bits()))
    }
}

#[cfg(test)]
mod tests {
    use super::MmapEndianConvertible;

    #[test]
    fn test_u32_roundtrip() {
        let value = 0xAABB_CCDD_u32;
        let stored = value.to_le_storage();
        let decoded = u32::from_le(stored);
        assert_eq!(decoded, value);
        assert_eq!(u32::from_le_storage(stored), value);
    }

    #[test]
    fn test_f32_roundtrip() {
        let value = 123.456f32;
        let stored = value.to_le_storage();
        let decoded = f32::from_le_storage(stored);
        assert_eq!(decoded.to_bits(), value.to_bits());
    }
}
