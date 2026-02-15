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

stage() {
  local name="$1"
  shift
  local log="${OUT_DIR}/${name}_${ts}.log"

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
stage quantization cargo test -p quantization --locked
stage segment_endian cargo test -p segment endian --locked

# Compile qdrant tests without running (stressful link stage on small hosts).
stage qdrant_norun cargo test -p qdrant --features rocksdb --locked --no-run

# End-to-end HTTP smoke test (ignored by default; run explicitly here).
stage qdrant_http_smoke cargo test -p qdrant --features rocksdb --locked --test s390x_http_smoke -- --ignored

echo
echo "All stages completed successfully."
