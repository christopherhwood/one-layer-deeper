#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

candidate="submissions/generic_accumulator_program/submission.py"
expected_sha="27d909b6923e1b49e4342401a90bb26b174bc1059a1439d1e99067809568dcfa"
actual_sha="$(shasum -a 256 "$candidate" | awk '{print $1}')"

if [[ "$actual_sha" != "$expected_sha" ]]; then
  echo "refusing preflight: candidate SHA-256 is $actual_sha, expected $expected_sha" >&2
  exit 1
fi

.venv/bin/python -m unittest tests.test_generic_accumulator_program
.venv/bin/one-layer validate "$candidate"

manifest="benchmark/manifests/local_cpu_10s.json"
if [[ "${1:-}" == "--full" ]]; then
  manifest="benchmark/manifests/local_cpu_medium_m5_60s.json"
fi

preflight_tmp="$(mktemp -d "${TMPDIR:-/tmp}/one-layer-preflight.XXXXXX")"
trap 'rm -rf "$preflight_tmp"' EXIT
result_file="$preflight_tmp/result.txt"

.venv/bin/python -m benchmark.runner \
  --manifest "$manifest" \
  --submission-file "$candidate" \
  --include-structured-metrics | tee "$result_file"

.venv/bin/python - "$result_file" <<'PY'
import json
import pathlib
import sys

lines = pathlib.Path(sys.argv[1]).read_text().splitlines()
records = [line.removeprefix("RESULT_JSON=") for line in lines if line.startswith("RESULT_JSON=")]
if len(records) != 1:
    raise SystemExit("preflight failed: runner did not emit exactly one RESULT_JSON record")
result = json.loads(records[0])
score = float(result["score"]["mean_exact_accuracy"])
if score != 1.0:
    raise SystemExit(f"preflight failed: exact score is {score:.6%}, expected 100%")
print(f"preflight passed: exact source {score:.2%}, SHA-256 verified")
PY

echo "No Hard submission was made. Hard remains owner/manual-only."
