#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

candidate="submissions/generic_integer_isa/submission.py"
expected_sha="81e257f755cd14975208accc6e4157d109d2e040c19668b6493c7b19ceff31a3"
actual_sha="$(shasum -a 256 "$candidate" | awk '{print $1}')"

if [[ "$actual_sha" != "$expected_sha" ]]; then
  echo "refusing preflight: candidate SHA-256 is $actual_sha, expected $expected_sha" >&2
  exit 1
fi

.venv/bin/python -m unittest tests.test_generic_integer_isa
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
