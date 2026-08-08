#!/bin/bash
# Pre-submit gate: refuse hosted submission unless the target tier's local
# shape replica has a recorded PASS newer than the current submission.py.
#
# Usage: presubmit_gate.sh <easy|medium|hard> [dataset]
# Reads PASS markers written by the validation runs:
#   submissions/learned_isa_machine/.gate_easy   (affine 60s = 1.0/1.0 + e1-shape clean)
#   submissions/learned_isa_machine/.gate_medium (m5-shape replica clean + plant adoption)
#   submissions/learned_isa_machine/.gate_hard   (hard-shape replica sweep clean)
# Each marker must contain the sha256 of the submission.py it validated.
set -euo pipefail
tier="${1:?usage: presubmit_gate.sh <easy|medium|hard> [dataset]}"
dataset="${2:-}"
dir="$(cd "$(dirname "$0")" && pwd)"
sub="$dir/submission.py"
marker="$dir/.gate_$tier"

if [ ! -f "$marker" ]; then
  echo "GATE FAIL: no recorded local validation for tier '$tier' ($marker missing)."
  echo "Run the tier's local shape replica and record: sha256sum submission.py > $marker"
  exit 1
fi
want=$(sha256sum "$sub" | awk '{print $1}')
have=$(awk '{print $1}' "$marker" | head -1)
if [ "$want" != "$have" ]; then
  echo "GATE FAIL: submission.py changed since tier '$tier' was last validated."
  echo "  validated: $have"
  echo "  current:   $want"
  echo "Re-run the tier's local replica, then: sha256sum submission.py > $marker"
  exit 1
fi
echo "GATE PASS: tier '$tier' validated for current submission.py ($want)"
cd "$dir/../.."
exec .venv/bin/python -m client.cli submit "$sub" --tier "$tier" ${dataset:+--dataset "$dataset"}
