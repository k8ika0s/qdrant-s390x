#!/usr/bin/env bash

set -euo pipefail

if [ "$#" -eq 0 ]; then
  echo "Usage: $0 <file> [file ...]"
  echo "Pass only touched persistence-format files."
  exit 0
fi

fail=0

for file in "$@"; do
  if [ ! -f "$file" ]; then
    echo "skip: $file (not a file)"
    continue
  fi

  if rg -n 'to_ne_bytes\\(|from_ne_bytes\\(' "$file" >/dev/null; then
    echo "error: native-endian byte conversion in $file"
    rg -n 'to_ne_bytes\\(|from_ne_bytes\\(' "$file"
    fail=1
  fi

  if rg -n 'transmute|from_raw_parts' "$file" >/dev/null; then
    echo "warn: review raw reinterpretation in $file"
    rg -n 'transmute|from_raw_parts' "$file"
  fi

  if rg -n 'pub .*: usize|: usize,' "$file" >/dev/null; then
    echo "warn: review persisted struct fields using usize in $file"
    rg -n 'pub .*: usize|: usize,' "$file"
  fi
done

if [ "$fail" -ne 0 ]; then
  exit 1
fi
