//! Hardware detection and optimization hints for Apple Silicon.
//!
//! This module detects the Apple Silicon chip family, GPU core count, memory
//! size, and estimated bandwidth, then provides tuning recommendations for
//! the inference engine (prefill chunk size, decode batch size, KV-cache
//! budget, and quantisation).

use std::fmt;
use std::process::Command;

// ---------------------------------------------------------------------------
// AppleChip
// ---------------------------------------------------------------------------

/// Apple Silicon chip family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AppleChip {
    M1,
    M1Pro,
    M1Max,
    M1Ultra,
    M2,
    M2Pro,
    M2Max,
    M2Ultra,
    M3,
    M3Pro,
    M3Max,
    M3Ultra,
    M4,
    M4Pro,
    M4Max,
    M4Ultra,
    /// Fallback -- stores the GPU core count so we can still make reasonable
    /// tuning decisions even on unrecognised hardware.
    Unknown(u32),
}

impl fmt::Display for AppleChip {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AppleChip::M1 => write!(f, "Apple M1"),
            AppleChip::M1Pro => write!(f, "Apple M1 Pro"),
            AppleChip::M1Max => write!(f, "Apple M1 Max"),
            AppleChip::M1Ultra => write!(f, "Apple M1 Ultra"),
            AppleChip::M2 => write!(f, "Apple M2"),
            AppleChip::M2Pro => write!(f, "Apple M2 Pro"),
            AppleChip::M2Max => write!(f, "Apple M2 Max"),
            AppleChip::M2Ultra => write!(f, "Apple M2 Ultra"),
            AppleChip::M3 => write!(f, "Apple M3"),
            AppleChip::M3Pro => write!(f, "Apple M3 Pro"),
            AppleChip::M3Max => write!(f, "Apple M3 Max"),
            AppleChip::M3Ultra => write!(f, "Apple M3 Ultra"),
            AppleChip::M4 => write!(f, "Apple M4"),
            AppleChip::M4Pro => write!(f, "Apple M4 Pro"),
            AppleChip::M4Max => write!(f, "Apple M4 Max"),
            AppleChip::M4Ultra => write!(f, "Apple M4 Ultra"),
            AppleChip::Unknown(cores) => write!(f, "Unknown Apple Silicon ({cores} GPU cores)"),
        }
    }
}

// ---------------------------------------------------------------------------
// HardwareProfile
// ---------------------------------------------------------------------------

/// Detected hardware profile for the current machine.
pub struct HardwareProfile {
    /// Identified chip family.
    pub chip: AppleChip,
    /// Number of GPU cores.
    pub gpu_cores: u32,
    /// Total system memory in bytes.
    pub memory_bytes: u64,
    /// Estimated memory bandwidth in GB/s.
    pub memory_bandwidth_gbps: f64,
    /// Whether the system uses unified memory (always `true` on Apple Silicon).
    pub unified_memory: bool,
}

impl HardwareProfile {
    /// Detect the hardware profile of the current machine by querying `sysctl`.
    pub fn detect() -> Self {
        let brand = sysctl_string("machdep.cpu.brand_string");
        let memory_bytes = sysctl_u64("hw.memsize").unwrap_or(0);
        let gpu_cores = detect_gpu_cores();

        let chip = parse_chip(&brand, gpu_cores);
        let memory_bandwidth_gbps = estimate_bandwidth(chip);

        Self {
            chip,
            gpu_cores,
            memory_bytes,
            memory_bandwidth_gbps,
            unified_memory: true,
        }
    }

    /// Return optimisation hints tailored to the detected hardware.
    pub fn optimization_hints(&self) -> OptimizationHints {
        // Tier the chip into a rough performance bucket.
        let tier = chip_tier(self.chip);
        let mem_gb = self.memory_bytes as f64 / (1024.0 * 1024.0 * 1024.0);

        let prefill_chunk_size: usize;
        let decode_batch_size: usize;
        let recommended_quantization: Option<String>;

        match tier {
            // Tier 0 -- M1 base: conservative.
            0 => {
                prefill_chunk_size = 512;
                decode_batch_size = 1;
                recommended_quantization = if mem_gb <= 8.0 {
                    Some("q4_0".to_string())
                } else {
                    Some("q4_1".to_string())
                };
            }
            // Tier 1 -- M1 Pro/Max/Ultra, M2 base: moderate.
            1 => {
                prefill_chunk_size = 1024;
                decode_batch_size = 4;
                recommended_quantization = if mem_gb <= 16.0 {
                    Some("q4_1".to_string())
                } else {
                    Some("q8_0".to_string())
                };
            }
            // Tier 2 -- M2 Pro/Max/Ultra, M3 base: aggressive.
            2 => {
                prefill_chunk_size = 2048;
                decode_batch_size = 8;
                recommended_quantization = if mem_gb <= 32.0 {
                    Some("q8_0".to_string())
                } else {
                    None // fp16 is fine with plenty of memory
                };
            }
            // Tier 3 -- M3 Pro/Max/Ultra: most aggressive.
            3 => {
                prefill_chunk_size = 4096;
                decode_batch_size = 16;
                recommended_quantization = if mem_gb <= 36.0 {
                    Some("q8_0".to_string())
                } else {
                    None
                };
            }
            // Tier 4 -- M4 family: latest optimisations.
            _ => {
                prefill_chunk_size = 4096;
                decode_batch_size = 32;
                recommended_quantization = if mem_gb <= 32.0 {
                    Some("q8_0".to_string())
                } else {
                    None
                };
            }
        }

        // Reserve ~75 % of system memory for the KV cache (the model weights
        // and runtime overhead take the remaining ~25 %).
        let max_cache_memory_bytes = (self.memory_bytes as f64 * 0.75) as u64;

        OptimizationHints {
            prefill_chunk_size,
            decode_batch_size,
            max_cache_memory_bytes,
            recommended_quantization,
        }
    }
}

impl fmt::Display for HardwareProfile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mem_gb = self.memory_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
        write!(
            f,
            "{} ({} GPU cores, {:.0} GB unified memory, ~{:.0} GB/s bandwidth)",
            self.chip, self.gpu_cores, mem_gb, self.memory_bandwidth_gbps,
        )
    }
}

// ---------------------------------------------------------------------------
// OptimizationHints
// ---------------------------------------------------------------------------

/// Engine tuning knobs derived from the detected hardware.
#[derive(Debug, Clone)]
pub struct OptimizationHints {
    /// Maximum number of tokens to prefill in a single step.
    pub prefill_chunk_size: usize,
    /// Maximum number of sequences to decode in parallel.
    pub decode_batch_size: usize,
    /// Recommended upper bound for KV-cache memory usage (bytes).
    pub max_cache_memory_bytes: u64,
    /// Suggested quantisation format, or `None` if full precision is feasible.
    pub recommended_quantization: Option<String>,
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Run `sysctl -n <key>` and return the trimmed stdout as a `String`.
fn sysctl_string(key: &str) -> String {
    Command::new("sysctl")
        .args(["-n", key])
        .output()
        .ok()
        .and_then(|o| {
            if o.status.success() {
                String::from_utf8(o.stdout).ok()
            } else {
                None
            }
        })
        .map(|s| s.trim().to_string())
        .unwrap_or_default()
}

/// Run `sysctl -n <key>` and parse the result as `u64`.
fn sysctl_u64(key: &str) -> Option<u64> {
    let s = sysctl_string(key);
    s.parse::<u64>().ok()
}

/// Attempt to detect the number of GPU cores via `sysctl`.
/// Falls back to the IORegistry `gpu-core-count` key, then to a default of 8.
fn detect_gpu_cores() -> u32 {
    // Try the Metal performance-statistics sysctl first (macOS 13+).
    if let Some(n) = sysctl_u64("machdep.gpu.core_count") {
        return n as u32;
    }

    // Fallback: query IORegistry for gpu-core-count.
    if let Ok(output) = Command::new("ioreg")
        .args(["-r", "-c", "AGXAccelerator", "-d", "1"])
        .output()
    {
        if output.status.success() {
            if let Ok(text) = String::from_utf8(output.stdout) {
                for line in text.lines() {
                    if line.contains("gpu-core-count") {
                        // Line format: "gpu-core-count" = 40
                        if let Some(val) = line.split('=').nth(1) {
                            if let Ok(n) = val.trim().parse::<u32>() {
                                return n;
                            }
                        }
                    }
                }
            }
        }
    }

    // Last resort default.
    8
}

/// Parse the CPU brand string (from `machdep.cpu.brand_string`) into an
/// [`AppleChip`] variant.  Falls back to `Unknown(gpu_cores)`.
fn parse_chip(brand: &str, gpu_cores: u32) -> AppleChip {
    let brand_lower = brand.to_lowercase();

    // Order matters -- check longer suffixes first (e.g. "m1 ultra" before
    // "m1") so that we don't match the base variant prematurely.

    if brand_lower.contains("m4 ultra") {
        AppleChip::M4Ultra
    } else if brand_lower.contains("m4 max") {
        AppleChip::M4Max
    } else if brand_lower.contains("m4 pro") {
        AppleChip::M4Pro
    } else if brand_lower.contains("m4") {
        AppleChip::M4
    } else if brand_lower.contains("m3 ultra") {
        AppleChip::M3Ultra
    } else if brand_lower.contains("m3 max") {
        AppleChip::M3Max
    } else if brand_lower.contains("m3 pro") {
        AppleChip::M3Pro
    } else if brand_lower.contains("m3") {
        AppleChip::M3
    } else if brand_lower.contains("m2 ultra") {
        AppleChip::M2Ultra
    } else if brand_lower.contains("m2 max") {
        AppleChip::M2Max
    } else if brand_lower.contains("m2 pro") {
        AppleChip::M2Pro
    } else if brand_lower.contains("m2") {
        AppleChip::M2
    } else if brand_lower.contains("m1 ultra") {
        AppleChip::M1Ultra
    } else if brand_lower.contains("m1 max") {
        AppleChip::M1Max
    } else if brand_lower.contains("m1 pro") {
        AppleChip::M1Pro
    } else if brand_lower.contains("m1") {
        AppleChip::M1
    } else {
        AppleChip::Unknown(gpu_cores)
    }
}

/// Map a chip to a performance tier (0 = lowest, 4 = highest).
fn chip_tier(chip: AppleChip) -> u8 {
    match chip {
        AppleChip::M1 => 0,
        AppleChip::M1Pro | AppleChip::M1Max | AppleChip::M1Ultra | AppleChip::M2 => 1,
        AppleChip::M2Pro | AppleChip::M2Max | AppleChip::M2Ultra | AppleChip::M3 => 2,
        AppleChip::M3Pro | AppleChip::M3Max | AppleChip::M3Ultra => 3,
        AppleChip::M4 | AppleChip::M4Pro | AppleChip::M4Max | AppleChip::M4Ultra => 4,
        AppleChip::Unknown(_) => 1, // conservative default
    }
}

/// Return an estimated memory bandwidth in GB/s for the given chip.
///
/// These are approximate peak theoretical numbers sourced from Apple's
/// published specs.
fn estimate_bandwidth(chip: AppleChip) -> f64 {
    match chip {
        AppleChip::M1 => 68.25,
        AppleChip::M1Pro => 200.0,
        AppleChip::M1Max => 400.0,
        AppleChip::M1Ultra => 800.0,
        AppleChip::M2 => 100.0,
        AppleChip::M2Pro => 200.0,
        AppleChip::M2Max => 400.0,
        AppleChip::M2Ultra => 800.0,
        AppleChip::M3 => 100.0,
        AppleChip::M3Pro => 150.0,
        AppleChip::M3Max => 400.0,
        AppleChip::M3Ultra => 800.0,
        AppleChip::M4 => 120.0,
        AppleChip::M4Pro => 273.0,
        AppleChip::M4Max => 546.0,
        AppleChip::M4Ultra => 819.0,
        AppleChip::Unknown(_) => 100.0, // conservative estimate
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_chip_m3_max() {
        assert_eq!(parse_chip("Apple M3 Max", 40), AppleChip::M3Max);
    }

    #[test]
    fn parse_chip_m1() {
        assert_eq!(parse_chip("Apple M1", 8), AppleChip::M1);
    }

    #[test]
    fn parse_chip_unknown() {
        assert_eq!(parse_chip("Intel Core i9", 0), AppleChip::Unknown(0));
    }

    #[test]
    fn display_hardware_profile() {
        let profile = HardwareProfile {
            chip: AppleChip::M3Max,
            gpu_cores: 40,
            memory_bytes: 96 * 1024 * 1024 * 1024,
            memory_bandwidth_gbps: 400.0,
            unified_memory: true,
        };
        let display = format!("{profile}");
        assert!(display.contains("Apple M3 Max"));
        assert!(display.contains("40 GPU cores"));
        assert!(display.contains("96 GB"));
        assert!(display.contains("400 GB/s"));
    }

    #[test]
    fn optimization_hints_m1() {
        let profile = HardwareProfile {
            chip: AppleChip::M1,
            gpu_cores: 8,
            memory_bytes: 8 * 1024 * 1024 * 1024,
            memory_bandwidth_gbps: 68.25,
            unified_memory: true,
        };
        let hints = profile.optimization_hints();
        assert_eq!(hints.prefill_chunk_size, 512);
        assert_eq!(hints.decode_batch_size, 1);
        assert_eq!(hints.recommended_quantization.as_deref(), Some("q4_0"));
    }

    #[test]
    fn optimization_hints_m4_max() {
        let profile = HardwareProfile {
            chip: AppleChip::M4Max,
            gpu_cores: 40,
            memory_bytes: 128 * 1024 * 1024 * 1024,
            memory_bandwidth_gbps: 546.0,
            unified_memory: true,
        };
        let hints = profile.optimization_hints();
        assert_eq!(hints.prefill_chunk_size, 4096);
        assert_eq!(hints.decode_batch_size, 32);
        // With 128 GB, no quantisation should be recommended.
        assert!(hints.recommended_quantization.is_none());
    }
}
