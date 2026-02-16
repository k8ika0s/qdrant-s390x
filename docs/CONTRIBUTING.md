# Contributing to Qdrant
We love your input! We want to make contributing to this project as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features

## We Develop with GitHub
We use github to host code, to track issues and feature requests, as well as accept pull requests.

## We Use [GitHub Flow](https://guides.github.com/introduction/flow/index.html), So All Code Changes Happen Through Pull Requests
Pull requests are the best way to propose changes to the codebase (we use [GitHub Flow](https://docs.github.com/en/get-started/quickstart/github-flow)). We actively welcome your pull requests:

1. Fork the repo and create your branch from `dev`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation and API Schema definitions (see [development docs](https://github.com/qdrant/qdrant/blob/master/docs/DEVELOPMENT.md#api-changes))
4. If you've changed persisted data formats (mmap/index/snapshot/segment files), follow the [persistence endianness policy](https://github.com/qdrant/qdrant/blob/master/docs/PERSISTENCE_ENDIANNESS.md).
5. Ensure the test suite passes.
6. Make sure your code lints (with cargo).
7. Issue that pull request!

### Multi-Arch Safety Checklist

When a PR touches persistence, low-level memory layout, or architecture-specific fast paths:

1. Keep behavior unchanged on existing `x86_64`/`aarch64` paths unless the PR explicitly targets
   them.
2. Use explicit byte order for persisted numeric fields (`to_le_bytes`/`from_le_bytes` or
   equivalent wrappers), never native-endian helpers for on-disk contracts.
3. Avoid persisted `usize`/`isize`; use fixed-width types for serialized metadata.
4. Add/refresh compatibility tests (legacy-read, malformed-input, and cross-endian where relevant).
5. Include architecture-tagged validation evidence (CI logs and/or native host logs) in the PR
   description.

### Multi-Arch Safety Checklist

When a PR touches persistence, low-level memory layout, or architecture-specific fast paths:

1. Keep behavior unchanged on existing `x86_64`/`aarch64` paths unless the PR explicitly targets
   them.
2. Use explicit byte order for persisted numeric fields (`to_le_bytes`/`from_le_bytes` or
   equivalent wrappers), never native-endian helpers for on-disk contracts.
3. Avoid persisted `usize`/`isize`; use fixed-width types for serialized metadata.
4. Add/refresh compatibility tests (legacy-read, malformed-input, and cross-endian where relevant).
5. Include architecture-tagged validation evidence (CI logs and/or native host logs) in the PR
   description.

## Any contributions you make will be under the Apache License 2.0
In short, when you submit code changes, your submissions are understood to be under the same [Apache License 2.0](https://choosealicense.com/licenses/apache-2.0/) that covers the project. Feel free to contact the maintainers if that's a concern.

## Report bugs using GitHub's [issues](https://github.com/qdrant/qdrant/issues)
We use GitHub issues to track public bugs. Report a bug by [opening a new issue](https://github.com/qdrant/qdrant/issues/new/choose); it's that easy!

## Write bug reports with detail, background, and sample code

**Great Bug Reports** tend to have:

- A quick summary and/or background
- Steps to reproduce
  - Be specific!
  - Give sample code if you can.
- What you expected would happen
- What actually happens
- Notes (possibly including why you think this might be happening, or stuff you tried that didn't work)

## Use a Consistent Coding Style

If you are modifying Rust code, make sure it has no warnings from Cargo and follow [Rust code style](https://doc.rust-lang.org/1.0.0/style/).
The project uses [rustfmt](https://github.com/rust-lang/rustfmt) formatter. Please ensure to run it using the
```cargo +nightly fmt --all``` command. The project also use [clippy](https://github.com/rust-lang/rust-clippy) lint collection,
so please ensure running ``cargo clippy --workspace --all-features`` before submitting the PR.

## License
By contributing, you agree that your contributions will be licensed under its Apache License 2.0.
