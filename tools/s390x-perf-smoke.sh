#!/usr/bin/env bash
#
# Lightweight perf/operational smoke suite for s390x workstreams.
# Can also run on other architectures for same-host-class comparisons.
#
# Captures:
# - indexing/search benches (HNSW, sparse, quantization),
# - startup latency and RSS snapshots for qdrant process.
#
# Usage:
#   tools/s390x-perf-smoke.sh [out_dir]
#
# Notes:
# - This is intentionally short-running; it is not a full performance benchmark campaign.
# - Compare results only against runs on the same host class + profile.
set -euo pipefail

OUT_DIR="${1:-dev-docs/s390x-validation}"
mkdir -p "$OUT_DIR"

if ! command -v curl >/dev/null 2>&1; then
  echo "error: curl is required" >&2
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "error: python3 is required" >&2
  exit 1
fi

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

pick_port() {
  python3 - <<'PY'
import socket
s = socket.socket()
s.bind(("127.0.0.1", 0))
print(s.getsockname()[1])
s.close()
PY
}

wait_ready() {
  local base_url="$1"
  local deadline_s="${2:-30}"
  local start
  start="$(date +%s)"
  while true; do
    if curl -sf "${base_url}/collections" >/dev/null 2>&1; then
      return 0
    fi
    if (( "$(date +%s)" - start > deadline_s )); then
      return 1
    fi
    sleep 0.2
  done
}

read_vmrss_kb() {
  local pid="$1"
  if [[ -f "/proc/${pid}/status" ]]; then
    awk '/^VmRSS:/ {print $2; found=1} END {if (!found) print 0}' "/proc/${pid}/status"
    return
  fi
  if command -v ps >/dev/null 2>&1; then
    ps -o rss= -p "$pid" 2>/dev/null | awk 'NF {print $1; found=1} END {if (!found) print 0}'
    return
  fi
  echo 0
}

startup_memory_smoke() {
  local profile="${QDRANT_PERF_PROFILE:-debug}" # debug|release
  local bin_path

  case "$profile" in
    release)
      cargo build -p qdrant --features rocksdb --locked --release
      bin_path="target/release/qdrant"
      ;;
    debug)
      cargo build -p qdrant --features rocksdb --locked
      bin_path="target/debug/qdrant"
      ;;
    *)
      echo "error: unsupported QDRANT_PERF_PROFILE='${profile}' (use debug|release)" >&2
      return 1
      ;;
  esac

  if [[ ! -x "$bin_path" ]]; then
    echo "error: expected qdrant binary at ${bin_path}" >&2
    return 1
  fi

  local tmp_root
  tmp_root="$(mktemp -d)"
  local storage_path="${tmp_root}/storage"
  local snapshots_path="${tmp_root}/snapshots"
  local temp_path="${tmp_root}/tmp"
  local run_log="${tmp_root}/qdrant.log"
  mkdir -p "$storage_path" "$snapshots_path" "$temp_path"

  local http_port
  local grpc_port
  http_port="$(pick_port)"
  grpc_port="$(pick_port)"
  local base_url="http://127.0.0.1:${http_port}"
  local collection="s390x_perf_smoke"

  cleanup_proc() {
    local pid="$1"
    if kill -0 "$pid" >/dev/null 2>&1; then
      kill "$pid" >/dev/null 2>&1 || true
      wait "$pid" >/dev/null 2>&1 || true
    fi
  }

  run_boot() {
    local label="$1"
    local start_ns ready_ns ready_ms rss_ready_kb rss_after_kb
    start_ns="$(date +%s%N)"
    QDRANT__SERVICE__HOST=127.0.0.1 \
      QDRANT__SERVICE__HTTP_PORT="${http_port}" \
      QDRANT__SERVICE__GRPC_PORT="${grpc_port}" \
      QDRANT__STORAGE__STORAGE_PATH="${storage_path}" \
      QDRANT__STORAGE__SNAPSHOTS_PATH="${snapshots_path}" \
      QDRANT__STORAGE__TEMP_PATH="${temp_path}" \
      QDRANT__TELEMETRY_DISABLED=true \
      RUST_LOG=warn \
      "$bin_path" >"$run_log" 2>&1 &
    local pid=$!

    if ! wait_ready "$base_url" 30; then
      echo "error: qdrant did not become ready on ${label}" >&2
      tail -n 80 "$run_log" >&2 || true
      cleanup_proc "$pid"
      return 1
    fi

    ready_ns="$(date +%s%N)"
    ready_ms=$(( (ready_ns - start_ns) / 1000000 ))
    rss_ready_kb="$(read_vmrss_kb "$pid")"
    echo "## ${label}: ready_ms=${ready_ms}"
    echo "## ${label}: rss_ready_kb=${rss_ready_kb}"

    if [[ "$label" == "boot1" ]]; then
      curl -sS -o /dev/null -f -X DELETE "${base_url}/collections/${collection}" || true
      curl -sS -o /dev/null -f -X PUT \
        "${base_url}/collections/${collection}" \
        -H 'Content-Type: application/json' \
        -d '{"vectors":{"size":4,"distance":"Dot"},"optimizers_config":{"default_segment_number":1},"replication_factor":1}'
      curl -sS -o /dev/null -f -X PUT \
        "${base_url}/collections/${collection}/points?wait=true" \
        -H 'Content-Type: application/json' \
        -d '{"points":[{"id":1,"vector":[0.05,0.61,0.76,0.74],"payload":{"city":"Berlin","count":1}},{"id":2,"vector":[0.19,0.81,0.75,0.11],"payload":{"city":"London","count":2}},{"id":3,"vector":[0.36,0.55,0.47,0.94],"payload":{"city":"Moscow","count":3}}]}'
      local resp
      resp="$(curl -sS -X POST "${base_url}/collections/${collection}/points/search" \
        -H 'Content-Type: application/json' \
        -d '{"vector":[0.2,0.1,0.9,0.7],"top":3}')"
      RESP_JSON="$resp" python3 - <<'PY'
import json
import os
v = json.loads(os.environ["RESP_JSON"])
hits = v.get("result")
assert isinstance(hits, list) and len(hits) > 0, f"expected non-empty search result: {v}"
PY
    else
      local info
      info="$(curl -sS -X GET "${base_url}/collections/${collection}")"
      INFO_JSON="$info" python3 - <<'PY'
import json
import os
v = json.loads(os.environ["INFO_JSON"])
points = v.get("result", {}).get("points_count")
assert isinstance(points, int) and points >= 3, f"expected points_count >= 3, got: {v}"
PY
    fi

    rss_after_kb="$(read_vmrss_kb "$pid")"
    echo "## ${label}: rss_after_workload_kb=${rss_after_kb}"

    cleanup_proc "$pid"
  }

  run_boot boot1
  run_boot boot2
  rm -rf "$tmp_root"
}

# Keep runs reproducible and bounded.
export CARGO_INCREMENTAL="${CARGO_INCREMENTAL:-0}"
export QDRANT_QBENCH_VECTORS="${QDRANT_QBENCH_VECTORS:-4096}"
export QDRANT_QBENCH_DIM="${QDRANT_QBENCH_DIM:-64}"
export QDRANT_QBENCH_SAMPLE_SIZE="${QDRANT_QBENCH_SAMPLE_SIZE:-10}"
export QDRANT_QBENCH_WARMUP_SECS="${QDRANT_QBENCH_WARMUP_SECS:-1}"
export QDRANT_QBENCH_MEASUREMENT_SECS="${QDRANT_QBENCH_MEASUREMENT_SECS:-2}"
export S390X_PERF_WARMUP_SECS="${S390X_PERF_WARMUP_SECS:-1}"
export S390X_PERF_MEASUREMENT_SECS="${S390X_PERF_MEASUREMENT_SECS:-2}"
export S390X_PERF_SAMPLE_SIZE="${S390X_PERF_SAMPLE_SIZE:-10}"

if [[ "${S390X_PERF_SKIP_BENCHES:-0}" != "1" ]]; then
  stage hnsw_persistence_smoke cargo bench -p segment --features rocksdb --bench hnsw_persistence_smoke -- \
    --warm-up-time "${S390X_PERF_WARMUP_SECS}" \
    --measurement-time "${S390X_PERF_MEASUREMENT_SECS}" \
    --sample-size "${S390X_PERF_SAMPLE_SIZE}"

  stage sparse_index_build cargo bench -p segment --features rocksdb --bench sparse_index_build -- \
    --warm-up-time "${S390X_PERF_WARMUP_SECS}" \
    --measurement-time "${S390X_PERF_MEASUREMENT_SECS}" \
    --sample-size "${S390X_PERF_SAMPLE_SIZE}"

  stage quantization_persistence_smoke cargo bench -p quantization --bench persistence_smoke -- --nocapture
else
  echo "# benchmark stages skipped (set S390X_PERF_SKIP_BENCHES=0 to run)"
fi

stage startup_memory_smoke startup_memory_smoke

echo
echo "All perf smoke stages completed successfully."
