# s390x PR Notes (What Changed and Why)

This document is a maintainer-facing changelog for the s390x compatibility stream.  
Each slice should keep a narrow scope and include test evidence.

## PR Note Template

- Scope:
- Why:
- Risk:
- Validation:

## Applied Notes

### F4/F5 Policy + Hash Determinism

- Scope: codified canonical little-endian persistence policy and stabilized cross-arch hash inputs.
- Why: prevent architecture-dependent routing and persisted-format drift.
- Risk: low (policy + deterministic encoding at boundaries).
- Validation: unit tests and routing consistency tests on LE/BE.

### F6/F7/F8 Storage Migration Slices

- Scope: sparse, dense/chunked mmap, and HNSW graph persistence hardening with explicit endian-safe decoding.
- Why: remove native-endian assumptions from persisted files while keeping legacy-read compatibility.
- Risk: medium (storage compatibility), mitigated by versioning + fixtures + malformed-input tests.
- Validation: native s390x gate runs, cross-endian fixture producer/consumer tests, persistence smoke tests.

### F9 Quantization Canonicalization + Accuracy/Bench

- Scope: quantization metadata canonicalization (v2 + dual-reader), deterministic quantized fixture assertions,
  and a lightweight persistence/search benchmark (`persistence_smoke`).
- Why: ensure LE/BE portability for quantization files and keep quantized-search checks stable.
- Risk: medium (quantization read/write paths), mitigated by dual-reader support and regression coverage.
- Validation: `quantization` crate tests, snapshot fixture produce/consume on LE+BE, native benchmark logs.

### F10 Common Mmap Utilities Guardrails

- Scope: BE-safe `mmap_hashmap` accessors and call-site migrations away from raw stored-value reinterpretation.
- Why: centralize endian-safe decoding in shared utilities to avoid repeated ad hoc conversions.
- Risk: low-to-medium (utility-level API touchpoints).
- Validation: `common_mmap_hashmap` stage plus full native gate sweep.
