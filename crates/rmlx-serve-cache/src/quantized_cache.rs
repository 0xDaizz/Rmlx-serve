//! Quantized KV cache with 4-bit or 8-bit storage.
//!
//! Ported from mlx-lm's `cache.py` `QuantizedKVCache`. This cache stores
//! key/value data in quantized form to reduce memory usage, with per-group
//! scale factors for dequantization.

/// KV cache with 4-bit or 8-bit quantized storage.
///
/// Data is stored as packed bytes with per-group scale factors.
/// Groups of `group_size` elements share a single scale factor,
/// enabling asymmetric min-max quantization.
///
/// Supported bit widths: 4 (2 values per byte) and 8 (1 value per byte).
pub struct QuantizedKVCache {
    /// `[layer]` -- quantized key bytes for each layer.
    keys: Vec<Vec<u8>>,
    /// `[layer]` -- quantized value bytes for each layer.
    values: Vec<Vec<u8>>,
    /// `[layer]` -- per-group scale factors for keys.
    key_scales: Vec<Vec<f32>>,
    /// `[layer]` -- per-group scale factors for values.
    value_scales: Vec<Vec<f32>>,
    /// `[layer]` -- per-group zero-point (minimum) for keys.
    key_zeros: Vec<Vec<f32>>,
    /// `[layer]` -- per-group zero-point (minimum) for values.
    value_zeros: Vec<Vec<f32>>,
    num_layers: usize,
    num_kv_heads: usize,
    head_dim: usize,
    /// Quantization bit width (4 or 8).
    bits: usize,
    /// Number of elements per quantization group.
    group_size: usize,
    /// Number of tokens currently cached.
    seq_len: usize,
}

impl QuantizedKVCache {
    /// Create a new empty quantized KV cache.
    ///
    /// # Arguments
    /// * `num_layers` - Number of transformer layers.
    /// * `num_kv_heads` - Number of KV heads per layer.
    /// * `head_dim` - Dimension of each head.
    /// * `bits` - Quantization bit width (4 or 8).
    /// * `group_size` - Number of elements per quantization group (e.g. 32, 64).
    ///
    /// # Panics
    /// Panics if `bits` is not 4 or 8, or if `head_dim` is not divisible by `group_size`.
    pub fn new(
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        bits: usize,
        group_size: usize,
    ) -> Self {
        assert!(
            bits == 4 || bits == 8,
            "bits must be 4 or 8, got {bits}"
        );
        assert!(
            head_dim % group_size == 0 || group_size > head_dim,
            "head_dim ({head_dim}) should be divisible by group_size ({group_size}) \
             for optimal quantization"
        );

        Self {
            keys: vec![Vec::new(); num_layers],
            values: vec![Vec::new(); num_layers],
            key_scales: vec![Vec::new(); num_layers],
            value_scales: vec![Vec::new(); num_layers],
            key_zeros: vec![Vec::new(); num_layers],
            value_zeros: vec![Vec::new(); num_layers],
            num_layers,
            num_kv_heads,
            head_dim,
            bits,
            group_size,
            seq_len: 0,
        }
    }

    /// Append new key/value data for a single layer.
    ///
    /// `keys` and `values` are flat f32 arrays of length
    /// `num_kv_heads * new_tokens * head_dim`.
    ///
    /// The data is quantized and appended to the stored cache.
    pub fn append(
        &mut self,
        layer: usize,
        keys: &[f32],
        values: &[f32],
        new_tokens: usize,
    ) {
        let expected_len = self.num_kv_heads * new_tokens * self.head_dim;
        assert_eq!(
            keys.len(),
            expected_len,
            "keys length mismatch: expected {expected_len}, got {}",
            keys.len()
        );
        assert_eq!(
            values.len(),
            expected_len,
            "values length mismatch: expected {expected_len}, got {}",
            values.len()
        );

        // Quantize the new data in groups.
        let (k_bytes, k_scales, k_zeros) = self.quantize_data(keys);
        let (v_bytes, v_scales, v_zeros) = self.quantize_data(values);

        self.keys[layer].extend_from_slice(&k_bytes);
        self.values[layer].extend_from_slice(&v_bytes);
        self.key_scales[layer].extend_from_slice(&k_scales);
        self.value_scales[layer].extend_from_slice(&v_scales);
        self.key_zeros[layer].extend_from_slice(&k_zeros);
        self.value_zeros[layer].extend_from_slice(&v_zeros);

        // Only update seq_len once (the first layer to be appended for a given
        // step drives the seq_len update). In practice, all layers are appended
        // in sequence for each token batch.
        if layer == 0 {
            self.seq_len += new_tokens;
        }
    }

    /// Dequantize and return all cached keys for a layer.
    ///
    /// Returns a flat `Vec<f32>` of length
    /// `num_kv_heads * seq_len * head_dim`.
    pub fn get_keys_dequantized(&self, layer: usize) -> Vec<f32> {
        self.dequantize_data(
            &self.keys[layer],
            &self.key_scales[layer],
            &self.key_zeros[layer],
        )
    }

    /// Dequantize and return all cached values for a layer.
    ///
    /// Returns a flat `Vec<f32>` of length
    /// `num_kv_heads * seq_len * head_dim`.
    pub fn get_values_dequantized(&self, layer: usize) -> Vec<f32> {
        self.dequantize_data(
            &self.values[layer],
            &self.value_scales[layer],
            &self.value_zeros[layer],
        )
    }

    /// Current sequence length (number of cached tokens).
    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    /// Number of layers.
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    /// Quantization bit width.
    pub fn bits(&self) -> usize {
        self.bits
    }

    /// Group size used for quantization.
    pub fn group_size(&self) -> usize {
        self.group_size
    }

    /// Quantize a flat f32 data slice into packed bytes, scales, and zeros.
    ///
    /// Data is divided into groups of `group_size`. Each group gets its own
    /// scale and zero-point using min-max quantization.
    fn quantize_data(&self, data: &[f32]) -> (Vec<u8>, Vec<f32>, Vec<f32>) {
        let gs = self.group_size;
        // Pad data length to be a multiple of group_size if needed.
        let num_groups = (data.len() + gs - 1) / gs;
        let mut all_bytes = Vec::new();
        let mut all_scales = Vec::with_capacity(num_groups);
        let mut all_zeros = Vec::with_capacity(num_groups);

        for g in 0..num_groups {
            let start = g * gs;
            let end = (start + gs).min(data.len());
            let group = &data[start..end];

            let (bytes, scale, zero) = Self::quantize_group(group, self.bits);
            all_bytes.extend_from_slice(&bytes);
            all_scales.push(scale);
            all_zeros.push(zero);
        }

        (all_bytes, all_scales, all_zeros)
    }

    /// Dequantize packed bytes back into f32 values.
    fn dequantize_data(&self, bytes: &[u8], scales: &[f32], zeros: &[f32]) -> Vec<f32> {
        if bytes.is_empty() {
            return Vec::new();
        }

        let gs = self.group_size;
        let bytes_per_group = match self.bits {
            4 => (gs + 1) / 2, // 2 values per byte
            8 => gs,
            _ => unreachable!(),
        };

        let mut result = Vec::new();
        for (g, (&scale, &zero)) in scales.iter().zip(zeros.iter()).enumerate() {
            let start = g * bytes_per_group;
            let end = (start + bytes_per_group).min(bytes.len());
            let group_bytes = &bytes[start..end];

            let group_values = Self::dequantize_group(group_bytes, scale, zero, self.bits);
            result.extend_from_slice(&group_values);
        }

        result
    }

    /// Quantize a single group of f32 values into packed bytes.
    ///
    /// Uses min-max (asymmetric) quantization:
    ///   quantized = round((value - min) / scale)
    ///   scale = (max - min) / (2^bits - 1)
    ///
    /// Returns (packed_bytes, scale, zero_point/min).
    fn quantize_group(data: &[f32], bits: usize) -> (Vec<u8>, f32, f32) {
        if data.is_empty() {
            return (Vec::new(), 0.0, 0.0);
        }

        let max_val = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let min_val = data.iter().copied().fold(f32::INFINITY, f32::min);

        let qmax = ((1u32 << bits) - 1) as f32;
        let range = max_val - min_val;
        let scale = if range == 0.0 { 1.0 } else { range / qmax };

        match bits {
            8 => {
                let bytes: Vec<u8> = data
                    .iter()
                    .map(|&v| {
                        let q = ((v - min_val) / scale).round().clamp(0.0, qmax) as u8;
                        q
                    })
                    .collect();
                (bytes, scale, min_val)
            }
            4 => {
                // Pack 2 values per byte: low nibble first, high nibble second.
                let mut bytes = Vec::with_capacity((data.len() + 1) / 2);
                let mut i = 0;
                while i < data.len() {
                    let low = ((data[i] - min_val) / scale).round().clamp(0.0, qmax) as u8;
                    let high = if i + 1 < data.len() {
                        ((data[i + 1] - min_val) / scale).round().clamp(0.0, qmax) as u8
                    } else {
                        0
                    };
                    bytes.push(low | (high << 4));
                    i += 2;
                }
                (bytes, scale, min_val)
            }
            _ => unreachable!("unsupported bit width: {bits}"),
        }
    }

    /// Dequantize a single group of packed bytes back into f32 values.
    ///
    /// Reverses the quantization: value = quantized * scale + zero_point.
    fn dequantize_group(data: &[u8], scale: f32, zero: f32, bits: usize) -> Vec<f32> {
        match bits {
            8 => data
                .iter()
                .map(|&b| (b as f32) * scale + zero)
                .collect(),
            4 => {
                let mut result = Vec::with_capacity(data.len() * 2);
                for &byte in data {
                    let low = (byte & 0x0F) as f32;
                    let high = ((byte >> 4) & 0x0F) as f32;
                    result.push(low * scale + zero);
                    result.push(high * scale + zero);
                }
                result
            }
            _ => unreachable!("unsupported bit width: {bits}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_quantized_cache() {
        let cache = QuantizedKVCache::new(32, 8, 128, 4, 32);
        assert_eq!(cache.seq_len(), 0);
        assert_eq!(cache.num_layers(), 32);
        assert_eq!(cache.bits(), 4);
        assert_eq!(cache.group_size(), 32);
    }

    #[test]
    fn test_quantize_dequantize_8bit_roundtrip() {
        let data = vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5];
        let (bytes, scale, zero) = QuantizedKVCache::quantize_group(&data, 8);
        let recovered = QuantizedKVCache::dequantize_group(&bytes, scale, zero, 8);

        assert_eq!(recovered.len(), data.len());
        for (orig, rec) in data.iter().zip(recovered.iter()) {
            assert!(
                (orig - rec).abs() < 0.02,
                "8-bit roundtrip: {orig} -> {rec}"
            );
        }
    }

    #[test]
    fn test_quantize_dequantize_4bit_roundtrip() {
        let data = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let (bytes, scale, zero) = QuantizedKVCache::quantize_group(&data, 4);
        let recovered = QuantizedKVCache::dequantize_group(&bytes, scale, zero, 4);

        // 4-bit only has 16 levels so some quantization error is expected.
        assert_eq!(recovered.len(), data.len());
        for (orig, rec) in data.iter().zip(recovered.iter()) {
            assert!(
                (orig - rec).abs() < 0.6,
                "4-bit roundtrip: {orig} -> {rec}"
            );
        }
    }

    #[test]
    fn test_append_and_retrieve() {
        // 1 layer, 1 head, head_dim=8, 8-bit, group_size=8
        let mut cache = QuantizedKVCache::new(1, 1, 8, 8, 8);

        // Append 1 token (8 floats for keys, 8 for values)
        let keys: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let values: Vec<f32> = (0..8).map(|i| (i as f32) * 10.0).collect();

        cache.append(0, &keys, &values, 1);
        assert_eq!(cache.seq_len(), 1);

        let dk = cache.get_keys_dequantized(0);
        assert_eq!(dk.len(), 8);

        // Check approximate roundtrip
        for (orig, rec) in keys.iter().zip(dk.iter()) {
            assert!(
                (orig - rec).abs() < 0.1,
                "key roundtrip: {orig} -> {rec}"
            );
        }
    }

    #[test]
    fn test_constant_values() {
        // All same value -- scale should handle this gracefully (range = 0).
        let data = vec![42.0; 8];
        let (bytes, scale, zero) = QuantizedKVCache::quantize_group(&data, 8);
        let recovered = QuantizedKVCache::dequantize_group(&bytes, scale, zero, 8);

        for &v in &recovered {
            assert!(
                (v - 42.0).abs() < 0.01,
                "constant roundtrip: 42.0 -> {v}"
            );
        }
    }

    #[test]
    #[should_panic(expected = "bits must be 4 or 8")]
    fn test_invalid_bits() {
        let _ = QuantizedKVCache::new(1, 1, 8, 3, 8);
    }
}
