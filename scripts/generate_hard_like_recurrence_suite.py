"""Generate a compact, evaluator-native suite of changed recurrences.

The suite uses variable semiprime moduli, deep endpoint-only training prompts,
held-out time depth, and matched seen-N/OOD-N certification ladders.  It is a
local hypothesis audit, not a claim about the private Hard recurrence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from data.squaring_mod import (
    SquaringModGenerationConfig,
    generate_squaring_mod_dataset,
)
from scripts.generate_hidden_recurrence_probe import TRANSITIONS


DEFAULT_FAMILIES = (
    "affine",
    "cube",
    "quadratic_plus_two",
    "conditional_threshold",
    "alternating_square_affine",
    "source_residual",
    "second_order_add",
    "time_indexed",
    "modulus_keyed",
    "bitwise_shift_mix",
)
DEPTH_LADDER = (1, 2, 4, 8, 16, 32, 64)


def generate_suite(
    output_root: Path,
    families: list[str],
    *,
    examples_per_setting: int,
    profile_examples: int,
) -> dict[str, object]:
    output_root.mkdir(parents=True, exist_ok=True)
    generated: dict[str, object] = {}
    for family in families:
        family_root = output_root / family
        config = SquaringModGenerationConfig(
            output_dir=str(family_root),
            modulus_bits=[10, 11],
            time_steps=[2, 4, 8],
            examples_per_setting=examples_per_setting,
            seed=45,
            train_fraction=0.8,
            test_fraction=0.2,
            ood_time_steps=[16],
            ood_examples_per_setting=max(32, examples_per_setting // 4),
            generator_family=f"hard_like_{family}",
            separate_input_output=True,
            split_group="prompt",
            depth_evaluation_time_steps=list(DEPTH_LADDER),
            depth_evaluation_examples_per_setting=profile_examples,
            ood_n_depth_evaluation_modulus_bits=[12, 13],
            ood_n_depth_evaluation_examples_per_setting=profile_examples,
        )
        generated[family] = generate_squaring_mod_dataset(
            config, transition=TRANSITIONS[family]
        )
        print(
            f"generated {family}: "
            f"{generated[family]['num_examples']} examples at {family_root}",
            flush=True,
        )

    suite = {
        "purpose": (
            "Local structural audit inspired by the private Hard recurrence "
            "warning; no private recurrence information is encoded."
        ),
        "families": families,
        "training_time_steps": [2, 4, 8],
        "ood_time_steps": [16],
        "depth_ladder": list(DEPTH_LADDER),
        "seen_modulus_bits": [10, 11],
        "ood_modulus_bits": [12, 13],
        "examples_per_setting": examples_per_setting,
        "profile_examples_per_setting": profile_examples,
    }
    (output_root / "suite_config.json").write_text(
        json.dumps(suite, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return suite


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        default="data/generated/hard_like_recurrence_suite",
    )
    parser.add_argument(
        "--families",
        nargs="+",
        choices=tuple(TRANSITIONS),
        default=list(DEFAULT_FAMILIES),
    )
    parser.add_argument("--examples-per-setting", type=int, default=256)
    parser.add_argument("--profile-examples", type=int, default=32)
    args = parser.parse_args()
    generate_suite(
        Path(args.output_root),
        args.families,
        examples_per_setting=args.examples_per_setting,
        profile_examples=args.profile_examples,
    )


if __name__ == "__main__":
    main()
