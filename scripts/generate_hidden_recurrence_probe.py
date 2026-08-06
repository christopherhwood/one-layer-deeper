"""Generate small endpoint-only probes with recurrences unlike public squaring."""

from __future__ import annotations

import argparse

from data.squaring_mod import (
    SquaringModGenerationConfig,
    generate_squaring_mod_dataset,
)


def affine(value: int, time_steps: int, p: int, q: int) -> int:
    modulus = p * q
    for _ in range(time_steps):
        value = (3 * value + 1) % modulus
    return value


def cube(value: int, time_steps: int, p: int, q: int) -> int:
    modulus = p * q
    for _ in range(time_steps):
        value = pow(value, 3, modulus)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("family", choices=("affine", "cube"))
    parser.add_argument("output_dir")
    args = parser.parse_args()
    transition = {"affine": affine, "cube": cube}[args.family]
    config = SquaringModGenerationConfig(
        output_dir=args.output_dir,
        fixed_p=17,
        fixed_q=19,
        time_steps=[1, 2, 3],
        examples_per_setting=250,
        seed=45,
        train_fraction=0.8,
        test_fraction=0.2,
        ood_time_steps=[6],
        ood_examples_per_setting=100,
        generator_family=f"hidden_probe_{args.family}",
        separate_input_output=True,
        split_group="prompt",
    )
    generate_squaring_mod_dataset(config, transition=transition)


if __name__ == "__main__":
    main()
