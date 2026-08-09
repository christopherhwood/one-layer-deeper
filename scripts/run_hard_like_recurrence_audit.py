"""Run one submission against generated Hard-like recurrence families."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def _manifest(family_root: Path, budget_seconds: float) -> dict[str, object]:
    return {
        "name": f"hard-like-{family_root.name}-cpu-{budget_seconds:g}s",
        "data": {
            "kind": "squaring_mod",
            "data_root": str(family_root.resolve()),
            "batch_size": 8,
            "eval_batch_size": 512,
            "shuffle_train": True,
            "shuffle_eval": False,
            "num_workers": 0,
            "pin_memory": False,
            "drop_last": True,
            "seed": 45,
        },
        "runtime": {
            "device": "cpu",
            "dtype": "float32",
            "amp": False,
            "compile": False,
            "total_training_time_seconds": budget_seconds,
            "max_steps": 1_000_000,
            "seeds": [74],
            "grad_clip": 1.0,
            "log_every": 100,
        },
        "model_state": {"maximum_elements": 500_000_000},
    }


def _run_one(
    submission: Path,
    family_root: Path,
    budget_seconds: float,
) -> dict[str, object]:
    with tempfile.TemporaryDirectory() as directory:
        manifest_path = Path(directory) / "manifest.json"
        manifest_path.write_text(
            json.dumps(_manifest(family_root, budget_seconds)),
            encoding="utf-8",
        )
        command = [
            sys.executable,
            "-m",
            "benchmark.runner",
            "--manifest",
            str(manifest_path),
            "--submission-file",
            str(submission),
            "--include-structured-metrics",
        ]
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        result_lines: list[str] = []
        for line in process.stdout:
            print(line, end="", flush=True)
            if line.startswith("RESULT_JSON="):
                result_lines.append(line.removeprefix("RESULT_JSON=").strip())
        return_code = process.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, command)
        if len(result_lines) != 1:
            raise RuntimeError(
                f"expected one RESULT_JSON for {family_root.name}, "
                f"received {len(result_lines)}"
            )
        return json.loads(result_lines[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission")
    parser.add_argument(
        "--suite-root",
        default="data/generated/hard_like_recurrence_suite",
    )
    parser.add_argument("--families", nargs="+")
    parser.add_argument("--budget-seconds", type=float, default=10.0)
    parser.add_argument("--output")
    args = parser.parse_args()

    submission = Path(args.submission).resolve()
    suite_root = Path(args.suite_root)
    suite = json.loads((suite_root / "suite_config.json").read_text())
    families = args.families or suite["families"]
    unknown = set(families) - set(suite["families"])
    if unknown:
        raise ValueError(f"families absent from generated suite: {sorted(unknown)}")

    audit: dict[str, object] = {
        "submission": str(submission),
        "budget_seconds": args.budget_seconds,
        "families": {},
    }
    for family in families:
        print(f"\n=== {family} ===", flush=True)
        result = _run_one(
            submission,
            suite_root / family,
            args.budget_seconds,
        )
        profile = result.get("depth_profile", {})
        summary = {
            "exact_accuracy": result["score"]["mean_exact_accuracy"],
            "seen_n_max_t": profile.get("max_certified_time_steps"),
            "ood_n_max_t": profile.get("ood_n_max_certified_time_steps"),
        }
        audit["families"][family] = summary
        print(f"AUDIT_RESULT {family} {json.dumps(summary)}", flush=True)

    output = (
        Path(args.output)
        if args.output
        else suite_root / "latest_audit.json"
    )
    output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(f"\nsaved audit to {output}", flush=True)


if __name__ == "__main__":
    main()
