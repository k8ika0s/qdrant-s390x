# Persistence Endianness Policy

This document defines the on-disk byte order contract for persisted Qdrant data.

## Scope

This policy applies to all persisted formats, including:

- mmap-backed structures
- segment/index metadata
- quantization files
- graph/link files
- snapshot payload data formats

This policy does not apply to purely in-memory temporary structures.

## Canonical byte order

Persisted numeric fields must use explicit little-endian encoding unless a format explicitly documents another canonical byte order.

Rationale:

- supports portable reads across little-endian and big-endian systems
- removes architecture-dependent behavior from storage formats
- makes format compatibility testing deterministic

## Disallowed for persisted formats

- `to_ne_bytes` / `from_ne_bytes`
- native reinterpretation/transmute of persisted bytes into primitive slices
- `usize` / `isize` in persisted structs
- architecture-dependent `repr(C)` layouts without explicit endian wrappers

## Allowed for persisted formats

- explicit endian codecs (`to_le_bytes`, `from_le_bytes`, `to_be_bytes`, `from_be_bytes`)
- `byteorder` with explicit endian type
- `zerocopy` endian wrappers (`little_endian::*` / `big_endian::*`)
- versioned format headers and compatibility readers

## Migration rules

When changing a persisted format:

1. bump or add a format version marker
2. keep legacy-read compatibility when practical
3. write new data in canonical explicit-endian format
4. add regression tests for:
   - legacy-read compatibility
   - cross-endian portability
   - malformed input handling

## Review checklist for PRs touching persisted formats

- does the change use explicit byte order for all persisted numeric fields?
- does the format avoid `usize`/`isize` on disk?
- is there a versioning or migration path?
- are cross-endian tests added or updated?
- does the PR describe backward compatibility impact?

## Helper tooling

Use `tools/check-persistence-endianness.sh` on touched persistence files for quick heuristic checks.

Example:

```bash
tools/check-persistence-endianness.sh \
  lib/segment/src/index/hnsw_index/graph_links/header.rs \
  lib/segment/src/vector_storage/dense/mmap_dense_vectors.rs
```

This helper is intentionally conservative and should be treated as advisory unless wired into CI with agreed allowlists.
