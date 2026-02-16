#!/usr/bin/env bash
#
# Native validation gates for s390x (big-endian) and other architectures.
# Writes per-stage logs into an output directory to make failures easy to triage.
#
# Usage:
#   tools/run-s390x-gates.sh [out_dir]
#
# Notes:
# - This script intentionally runs a "high-signal" subset by default.
# - If you want the full workspace test suite, run stage `segment_full` explicitly after
#   ensuring enough free disk space (segments can be very large).
set -euo pipefail

OUT_DIR="${1:-dev-docs/s390x-validation}"
mkdir -p "$OUT_DIR"

ts="$(date -u +%Y%m%dT%H%M%SZ)"
arch="$(uname -m 2>/dev/null || echo unknown)"
endian="$(
  rustc --print cfg 2>/dev/null | grep '^target_endian=' | head -n 1 | cut -d'"' -f2 || true
)"
endian="${endian:-unknown}"

stage() {
  local name="$1"
  shift
  local log="${OUT_DIR}/${name}_${arch}_${endian}_${ts}.log"

  {
    echo "# stage=${name}"
    echo "# start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "# pwd=$(pwd)"
    echo "# uname=$(uname -a || true)"
    echo "# rustc=$(rustc -Vv 2>/dev/null || true)"
    echo "# cargo=$(cargo -Vv 2>/dev/null || true)"
    echo "# cmd=$*"
    echo
    "$@"
    rc=$?
    echo
    echo "# end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "# rc=${rc}"
    exit "${rc}"
  } 2>&1 | tee "$log"
}

# Keep incremental behavior predictable for long runs.
export CARGO_INCREMENTAL="${CARGO_INCREMENTAL:-0}"

stage qdrant_check cargo check -p qdrant --locked
stage qdrant_check_rocksdb cargo check -p qdrant --features rocksdb --locked

# Compile the test graph (often the first place arch-specific link issues show up).
stage workspace_build_tests cargo build --workspace --features rocksdb --tests --locked

# High-signal, fast unit test slices.
stage common_stable_hash cargo test -p common stable_hash --locked
stage common_mmap_hashmap cargo test -p common mmap_hashmap --locked
stage collection_routing cargo test -p collection --locked test_routing_is_stable_across_architectures
stage quantization cargo test -p quantization --locked
stage segment_endian cargo test -p segment endian --locked
stage segment_mmap_point_to_values cargo test -p segment --locked mmap_point_to_values

# Compile qdrant tests without running (stressful link stage on small hosts).
stage qdrant_norun cargo test -p qdrant --features rocksdb --locked --no-run

# End-to-end HTTP smoke test (ignored by default; run explicitly here).
stage qdrant_http_smoke cargo test -p qdrant --features rocksdb --locked --test s390x_http_smoke -- --ignored

# End-to-end snapshot create/restore smoke test (ignored by default; run explicitly here).
stage qdrant_snapshot_smoke cargo test -p qdrant --features rocksdb --locked --test s390x_snapshot_smoke -- --ignored

if [[ -n "${S390X_FIXTURES_DIR:-}" ]]; then
  # Cross-endian fixture consumer: restore LE/BE-produced snapshots on this host.
  stage qdrant_snapshot_fixture_matrix cargo test -p qdrant --features rocksdb --locked --test s390x_snapshot_fixture_matrix -- --ignored
else
  echo "# stage=qdrant_snapshot_fixture_matrix skipped (set S390X_FIXTURES_DIR to enable)"
fi

if [[ -n "${S390X_CONTAINER_SMOKE:-}" ]]; then
  stage container_smoke tools/s390x-container-smoke.sh
else
  echo "# stage=container_smoke skipped (set S390X_CONTAINER_SMOKE=1 to enable)"
fi

echo
echo "All stages completed successfully."
