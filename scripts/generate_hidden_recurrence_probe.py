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


def cubic_polynomial(value: int, time_steps: int, p: int, q: int) -> int:
    modulus = p * q
    for _ in range(time_steps):
        value = (value**3 + value**2 + value + 1) % modulus
    return value


def quotient_mask_mix(value: int, time_steps: int, p: int, q: int) -> int:
    modulus = p * q
    for _ in range(time_steps):
        value = (
            value**2 // modulus + (value & (modulus - 1)) + 1
        ) % modulus
    return value


def quadratic_plus_two(value: int, time_steps: int, p: int, q: int) -> int:
    modulus = p * q
    for _ in range(time_steps):
        value = (value**2 + value + 2) % modulus
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "family",
        choices=(
            "affine",
            "cube",
            "cubic_polynomial",
            "quadratic_plus_two",
            "quotient_mask_mix",
        ),
    )
    parser.add_argument("output_dir")
    parser.add_argument("--time-steps", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--ood-time-steps", type=int, nargs="+", default=[6])
    parser.add_argument("--examples-per-setting", type=int, default=250)
    parser.add_argument("--ood-examples-per-setting", type=int, default=100)
    parser.add_argument("--fixed-p", type=int, default=17)
    parser.add_argument("--fixed-q", type=int, default=19)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--depth-time-steps", type=int, nargs="+", default=[])
    parser.add_argument("--depth-examples", type=int)
    parser.add_argument("--ood-n-bits", type=int, nargs="+", default=[])
    parser.add_argument("--ood-n-examples", type=int)
    args = parser.parse_args()
    transition = {
        "affine": affine,
        "cube": cube,
        "cubic_polynomial": cubic_polynomial,
        "quadratic_plus_two": quadratic_plus_two,
        "quotient_mask_mix": quotient_mask_mix,
    }[args.family]
    config = SquaringModGenerationConfig(
        output_dir=args.output_dir,
        fixed_p=args.fixed_p,
        fixed_q=args.fixed_q,
        time_steps=args.time_steps,
        examples_per_setting=args.examples_per_setting,
        seed=45,
        train_fraction=args.train_fraction,
        test_fraction=args.test_fraction,
        ood_time_steps=args.ood_time_steps,
        ood_examples_per_setting=args.ood_examples_per_setting,
        generator_family=f"hidden_probe_{args.family}",
        separate_input_output=True,
        split_group="prompt",
        depth_evaluation_time_steps=args.depth_time_steps,
        depth_evaluation_examples_per_setting=args.depth_examples,
        ood_n_depth_evaluation_modulus_bits=args.ood_n_bits,
        ood_n_depth_evaluation_examples_per_setting=args.ood_n_examples,
    )
    generate_squaring_mod_dataset(config, transition=transition)


if __name__ == "__main__":
    main()
