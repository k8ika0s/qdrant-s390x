#!/usr/bin/env bash
#
# Container smoke test for native s390x (and other architectures).
#
# This is intentionally a "runtime smoke" for an already-built native binary, not a test of the
# multi-arch Docker build pipeline (which uses buildx/xx/cargo-chef on amd64 runners).
#
# What it validates:
# - we can build a local container image that packages the native `qdrant` binary,
# - the container boots and serves HTTP,
# - basic CRUD/search works,
# - persisted data survives a container restart (same mounted storage path).
#
# Default engine: podman (falls back to docker).
set -euo pipefail

ENGINE="${CONTAINER_ENGINE:-}"
if [[ -z "$ENGINE" ]]; then
  if command -v podman >/dev/null 2>&1; then
    ENGINE="podman"
  elif command -v docker >/dev/null 2>&1; then
    ENGINE="docker"
  else
    echo "error: neither podman nor docker is available" >&2
    exit 1
  fi
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "error: curl is required for container smoke" >&2
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "error: python3 is required for container smoke (port selection + json assertions)" >&2
  exit 1
fi

PROFILE="${QDRANT_CONTAINER_PROFILE:-debug}" # debug|release
TAG="${QDRANT_CONTAINER_TAG:-qdrant-s390x-smoke:local}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

pick_port() {
  python3 - <<'PY'
import socket
s = socket.socket()
s.bind(("127.0.0.1", 0))
print(s.getsockname()[1])
s.close()
PY
}

curl_code() {
  # shellcheck disable=SC2005
  echo "$(curl -sS -o /dev/null -w "%{http_code}" "$@")"
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
      echo "error: qdrant did not become ready within ${deadline_s}s" >&2
      return 1
    fi
    sleep 0.2
  done
}

echo "## container_engine=${ENGINE}"
echo "## profile=${PROFILE}"
echo "## tag=${TAG}"

case "$PROFILE" in
  release)
    echo "## build: cargo build -p qdrant --features rocksdb --locked --release"
    cargo build -p qdrant --features rocksdb --locked --release
    BIN_PATH="target/release/qdrant"
    ;;
  debug)
    echo "## build: cargo build -p qdrant --features rocksdb --locked"
    cargo build -p qdrant --features rocksdb --locked
    BIN_PATH="target/debug/qdrant"
    ;;
  *)
    echo "error: unsupported QDRANT_CONTAINER_PROFILE='${PROFILE}' (use debug|release)" >&2
    exit 1
    ;;
esac

if [[ ! -x "$BIN_PATH" ]]; then
  echo "error: expected qdrant binary at ${BIN_PATH}" >&2
  exit 1
fi

ctx="$(mktemp -d)"
storage_dir="$(mktemp -d)"
snapshots_dir="$(mktemp -d)"
tmp_dir="$(mktemp -d)"

cleanup() {
  rm -rf "$ctx" "$storage_dir" "$snapshots_dir" "$tmp_dir"
}
trap cleanup EXIT

cp "$BIN_PATH" "$ctx/qdrant"
cp -R config "$ctx/config"
cp tools/entrypoint.sh "$ctx/entrypoint.sh"
chmod +x "$ctx/qdrant" "$ctx/entrypoint.sh"

cat >"$ctx/Dockerfile" <<'EOF'
FROM debian:13-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates tzdata libunwind8 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /qdrant
COPY qdrant /qdrant/qdrant
COPY config /qdrant/config
COPY entrypoint.sh /qdrant/entrypoint.sh
RUN chmod +x /qdrant/entrypoint.sh

ENV TZ=Etc/UTC \
    RUN_MODE=production

EXPOSE 6333
EXPOSE 6334

CMD ["./entrypoint.sh"]
EOF

echo "## image_build: ${ENGINE} build -t ${TAG} ${ctx}"
"$ENGINE" build -t "$TAG" "$ctx"

if "$ENGINE" image inspect "$TAG" --format '{{.Size}}' >/dev/null 2>&1; then
  size_bytes=$("$ENGINE" image inspect "$TAG" --format '{{.Size}}' | head -n 1)
  echo "## image_size_bytes=${size_bytes}"
else
  echo "## image_size_bytes=unknown"
fi

COLLECTION="s390x_container_smoke"
HTTP_PORT="$(pick_port)"
BASE_URL="http://127.0.0.1:${HTTP_PORT}"

run_once() {
  local label="$1"
  local start_ns ready_ns dur_ms
  start_ns="$(date +%s%N)"

  cid=$(
    "$ENGINE" run -d --rm \
      -p "127.0.0.1:${HTTP_PORT}:6333" \
      -v "${storage_dir}:/qdrant/storage" \
      -v "${snapshots_dir}:/qdrant/snapshots" \
      -v "${tmp_dir}:/qdrant/tmp" \
      -e QDRANT__SERVICE__HOST=0.0.0.0 \
      -e QDRANT__SERVICE__HTTP_PORT=6333 \
      -e QDRANT__SERVICE__GRPC_PORT=6334 \
      -e QDRANT__STORAGE__STORAGE_PATH=/qdrant/storage \
      -e QDRANT__STORAGE__SNAPSHOTS_PATH=/qdrant/snapshots \
      -e QDRANT__STORAGE__TEMP_PATH=/qdrant/tmp \
      -e QDRANT__TELEMETRY_DISABLED=true \
      -e RUST_LOG=warn \
      "$TAG"
  )

  if ! wait_ready "$BASE_URL" 30; then
    echo "## ${label}: container_logs"
    "$ENGINE" logs "$cid" || true
    "$ENGINE" stop "$cid" >/dev/null 2>&1 || true
    return 1
  fi

  ready_ns="$(date +%s%N)"
  dur_ms=$(( (ready_ns - start_ns) / 1000000 ))
  echo "## ${label}: ready_ms=${dur_ms}"

  if [[ "$label" == "boot1" ]]; then
    # Cleanup (200/404), create, upsert, search.
    code="$(curl_code -X DELETE "${BASE_URL}/collections/${COLLECTION}")"
    if [[ "$code" != "200" && "$code" != "404" ]]; then
      echo "error: delete collection returned http=${code}" >&2
      "$ENGINE" logs "$cid" || true
      "$ENGINE" stop "$cid" >/dev/null 2>&1 || true
      return 1
    fi

    code="$(curl_code -X PUT "${BASE_URL}/collections/${COLLECTION}" -H 'Content-Type: application/json' -d '{"vectors":{"size":4,"distance":"Dot"},"optimizers_config":{"default_segment_number":1},"replication_factor":1}')" || true
    if [[ "$code" != "200" ]]; then
      echo "error: create collection returned http=${code}" >&2
      "$ENGINE" logs "$cid" || true
      "$ENGINE" stop "$cid" >/dev/null 2>&1 || true
      return 1
    fi

    code="$(curl_code -X PUT "${BASE_URL}/collections/${COLLECTION}/points?wait=true" -H 'Content-Type: application/json' -d '{"points":[{"id":1,"vector":[0.05,0.61,0.76,0.74],"payload":{"city":"Berlin","count":1}},{"id":2,"vector":[0.19,0.81,0.75,0.11],"payload":{"city":"London","count":2}},{"id":3,"vector":[0.36,0.55,0.47,0.94],"payload":{"city":"Moscow","count":3}}]}')" || true
    if [[ "$code" != "200" ]]; then
      echo "error: upsert returned http=${code}" >&2
      "$ENGINE" logs "$cid" || true
      "$ENGINE" stop "$cid" >/dev/null 2>&1 || true
      return 1
    fi

    resp="$(curl -sS -X POST "${BASE_URL}/collections/${COLLECTION}/points/search" -H 'Content-Type: application/json' -d '{"vector":[0.2,0.1,0.9,0.7],"top":3}')" || true
    python3 - <<PY
import json, sys
v=json.loads(${resp@Q})
hits=v.get("result")
assert isinstance(hits, list) and len(hits) > 0, f"expected non-empty result, got: {v}"
PY
    echo "## ${label}: search_ok=1"
  else
    # On the second boot, validate persisted points exist.
    resp="$(curl -sS -X GET "${BASE_URL}/collections/${COLLECTION}")" || true
    python3 - <<PY
import json
v=json.loads(${resp@Q})
points=v.get("result", {}).get("points_count")
assert isinstance(points, int) and points >= 3, f"expected points_count >= 3, got: {v}"
PY
    echo "## ${label}: points_ok=1"
  fi

  "$ENGINE" stop "$cid" >/dev/null
}

run_once boot1
run_once boot2

echo "container smoke ok"
