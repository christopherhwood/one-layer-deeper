"""Generate endpoint-only probes with recurrences unlike public squaring.

The functions in ``TRANSITIONS`` deliberately isolate different assumptions a
recurrent model may make.  Some are short single-register calculator programs;
others require a branch, an immutable source register, a loop counter, a
modulus-derived constant, or a second latent register.
"""

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


def alternating_square_affine(
    value: int, time_steps: int, p: int, q: int
) -> int:
    """Alternate two transition rules, requiring a persistent phase bit."""

    modulus = p * q
    for step in range(time_steps):
        if step % 2 == 0:
            value = (value**2 + 1) % modulus
        else:
            value = (3 * value + 1) % modulus
    return value


def conditional_threshold(
    value: int, time_steps: int, p: int, q: int
) -> int:
    """Choose the transition from a data-dependent comparison."""

    modulus = p * q
    for _ in range(time_steps):
        if value < modulus // 2:
            value = (3 * value + 1) % modulus
        else:
            value = (value**2 + 2) % modulus
    return value


def source_residual(value: int, time_steps: int, p: int, q: int) -> int:
    """Mix every new state with the immutable original input."""

    modulus = p * q
    source = value
    for _ in range(time_steps):
        value = (value**2 + source + 1) % modulus
    return value


def second_order_add(value: int, time_steps: int, p: int, q: int) -> int:
    """A Fibonacci-like recurrence with two mutable state registers."""

    modulus = p * q
    previous = value
    current = (value + 1) % modulus
    for _ in range(time_steps):
        previous, current = current, (current + previous + 1) % modulus
    return current


def time_indexed(value: int, time_steps: int, p: int, q: int) -> int:
    """Change the transition at every step using an explicit loop counter."""

    modulus = p * q
    for step in range(time_steps):
        value = (value**2 + step + 1) % modulus
    return value


def modulus_keyed(value: int, time_steps: int, p: int, q: int) -> int:
    """Derive an arithmetic constant from the prompt modulus."""

    modulus = p * q
    coefficient = 2 + modulus % 5
    for _ in range(time_steps):
        value = (coefficient * value + 1) % modulus
    return value


def bitwise_shift_mix(value: int, time_steps: int, p: int, q: int) -> int:
    """Use shifts and modulus bits rather than polynomial arithmetic alone."""

    modulus = p * q
    modulus_key = modulus >> 3
    for _ in range(time_steps):
        value = (((value << 1) ^ value ^ modulus_key) + 1) % modulus
    return value


TRANSITIONS = {
    "affine": affine,
    "cube": cube,
    "cubic_polynomial": cubic_polynomial,
    "quadratic_plus_two": quadratic_plus_two,
    "quotient_mask_mix": quotient_mask_mix,
    "alternating_square_affine": alternating_square_affine,
    "conditional_threshold": conditional_threshold,
    "source_residual": source_residual,
    "second_order_add": second_order_add,
    "time_indexed": time_indexed,
    "modulus_keyed": modulus_keyed,
    "bitwise_shift_mix": bitwise_shift_mix,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "family",
        choices=tuple(TRANSITIONS),
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
    transition = TRANSITIONS[args.family]
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
