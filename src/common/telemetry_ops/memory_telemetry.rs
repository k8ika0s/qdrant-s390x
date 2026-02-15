use schemars::JsonSchema;
use segment::common::anonymize::Anonymize;
use serde::Serialize;
#[cfg(not(target_env = "msvc"))]
use storage::rbac::AccessRequirements;
#[cfg(all(
    not(target_env = "msvc"),
    any(target_arch = "x86_64", target_arch = "aarch64")
))]
use storage::rbac::Auth;
#[cfg(all(
    not(target_env = "msvc"),
    any(target_arch = "x86_64", target_arch = "aarch64")
))]
use tikv_jemalloc_ctl::{epoch, stats};

#[derive(Debug, Clone, Default, JsonSchema, Serialize, Anonymize)]
#[anonymize(false)]
/// Memory telemetry collected from the running process.
///
/// Notes on portability:
/// - On Linux `x86_64`/`aarch64` builds (non-MSVC), values are sourced from jemalloc stats.
/// - On other non-MSVC targets, `resident_bytes`/`retained_bytes` are best-effort from procfs
///   (`/proc/self/status`), and allocator-internal breakdowns are reported as `0`.
pub struct MemoryTelemetry {
    /// Total number of bytes in active pages allocated by the application
    pub active_bytes: usize,
    /// Total number of bytes allocated by the application
    pub allocated_bytes: usize,
    /// Total number of bytes dedicated to metadata
    pub metadata_bytes: usize,
    /// Maximum number of bytes in physically resident data pages mapped
    pub resident_bytes: usize,
    /// Total number of bytes in virtual memory mappings
    pub retained_bytes: usize,
}

impl MemoryTelemetry {
    fn clamp_u64_to_usize(value: u64) -> usize {
        if value > usize::MAX as u64 {
            usize::MAX
        } else {
            value as usize
        }
    }

    fn parse_proc_self_status_kb(status: &str, key: &str) -> Option<u64> {
        // Expected format: `VmRSS:\t  1234 kB`
        for line in status.lines() {
            if !line.starts_with(key) {
                continue;
            }
            let mut parts = line.split_whitespace();
            let found_key = parts.next()?;
            if found_key != key {
                return None;
            }
            return parts.next()?.parse::<u64>().ok();
        }

        None
    }

    fn parse_proc_self_status_bytes(status: &str) -> Option<(usize, usize)> {
        let rss_kb = Self::parse_proc_self_status_kb(status, "VmRSS:");
        let vmsize_kb = Self::parse_proc_self_status_kb(status, "VmSize:");

        if rss_kb.is_none() && vmsize_kb.is_none() {
            return None;
        }

        let rss_bytes = rss_kb
            .and_then(|kb| kb.checked_mul(1024))
            .map(Self::clamp_u64_to_usize)
            .unwrap_or_default();
        let vmsize_bytes = vmsize_kb
            .and_then(|kb| kb.checked_mul(1024))
            .map(Self::clamp_u64_to_usize)
            .unwrap_or_default();

        Some((rss_bytes, vmsize_bytes))
    }

    #[cfg(all(
        not(target_env = "msvc"),
        any(target_arch = "x86_64", target_arch = "aarch64")
    ))]
    pub fn collect(auth: &Auth) -> Option<MemoryTelemetry> {
        let required_access = AccessRequirements::new();
        if epoch::advance().is_ok()
            && auth
                .check_global_access(required_access, "telemetry_memory")
                .is_ok()
        {
            Some(MemoryTelemetry {
                active_bytes: stats::active::read().unwrap_or_default(),
                allocated_bytes: stats::allocated::read().unwrap_or_default(),
                metadata_bytes: stats::metadata::read().unwrap_or_default(),
                resident_bytes: stats::resident::read().unwrap_or_default(),
                retained_bytes: stats::retained::read().unwrap_or_default(),
            })
        } else {
            log::info!("Failed to advance Jemalloc stats epoch");
            None
        }
    }

    #[cfg(target_env = "msvc")]
    pub fn collect(_auth: &Auth) -> Option<MemoryTelemetry> {
        None
    }

    #[cfg(all(
        not(target_env = "msvc"),
        not(any(target_arch = "x86_64", target_arch = "aarch64"))
    ))]
    pub fn collect(auth: &Auth) -> Option<MemoryTelemetry> {
        // Best-effort fallback for targets where jemalloc ctl is not available or not enabled.
        // On Linux, `/proc/self/status` provides resident and virtual memory sizes which are
        // meaningful across allocators. Allocator-internal breakdowns are unavailable, so we
        // leave them as `0` rather than guessing.
        let required_access = AccessRequirements::new();
        auth.check_global_access(required_access, "telemetry_memory")
            .ok()?;

        let status = std::fs::read_to_string("/proc/self/status").ok()?;
        let (resident_bytes, retained_bytes) = Self::parse_proc_self_status_bytes(&status)?;

        Some(MemoryTelemetry {
            active_bytes: 0,
            allocated_bytes: 0,
            metadata_bytes: 0,
            resident_bytes,
            retained_bytes,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::MemoryTelemetry;

    #[test]
    fn parse_proc_self_status_bytes_extracts_rss_and_vmsize() {
        let status = "\
Name:\tqdrant\n\
VmSize:\t    2048 kB\n\
VmRSS:\t     1024 kB\n\
";
        let (rss, vmsize) = MemoryTelemetry::parse_proc_self_status_bytes(status).unwrap();
        assert_eq!(rss, 1024 * 1024);
        assert_eq!(vmsize, 2048 * 1024);
    }

    #[test]
    fn parse_proc_self_status_bytes_is_none_when_keys_missing() {
        let status = "Name:\tqdrant\nState:\tR (running)\n";
        assert!(MemoryTelemetry::parse_proc_self_status_bytes(status).is_none());
    }
}
