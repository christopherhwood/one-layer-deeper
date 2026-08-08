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
    parser.add_argument("--time-steps", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--ood-time-steps", type=int, nargs="+", default=[6])
    parser.add_argument("--examples-per-setting", type=int, default=250)
    parser.add_argument("--fixed-p", type=int, default=17)
    parser.add_argument("--fixed-q", type=int, default=19)
    args = parser.parse_args()
    transition = {"affine": affine, "cube": cube}[args.family]
    config = SquaringModGenerationConfig(
        output_dir=args.output_dir,
        fixed_p=args.fixed_p,
        fixed_q=args.fixed_q,
        time_steps=args.time_steps,
        examples_per_setting=args.examples_per_setting,
        seed=45,
        train_fraction=0.8,
        test_fraction=0.2,
        ood_time_steps=args.ood_time_steps,
        ood_examples_per_setting=100,
        generator_family=f"hidden_probe_{args.family}",
        separate_input_output=True,
        split_group="prompt",
    )
    generate_squaring_mod_dataset(config, transition=transition)


if __name__ == "__main__":
    main()
