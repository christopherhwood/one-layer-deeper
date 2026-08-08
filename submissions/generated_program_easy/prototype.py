"""Endpoint-only evolutionary generation of a previously absent program.

This is an offline geometry test for the benchmark submission.  It reuses the
same generic binary-program interpreter as ``binary_relation_hard`` but starts
with a small population from which the canonical complete square program is
mechanically removed.  Selection, program-level mutation, and random restarts
must create it using only final square endpoints.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from benchmark import ModelSpec


ROOT = Path(__file__).parents[2]
BASE_PATH = ROOT / "submissions" / "binary_relation_hard" / "submission.py"
SPEC = importlib.util.spec_from_file_location("generated_program_base", BASE_PATH)
BASE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(BASE)

FIELD_NAMES = (
    "beta_logits",
    "gamma_logits",
    "branch_logits",
    "initial_logits",
    "direction_logits",
    "addend_logits",
    "bit_branch_logits",
    "second_pass_logits",
)
FIELD_SIZES = (5, 7, 2, 2, 2, 2, 2, 2)
# Indices, not semantic values: beta=-1, gamma=-2, reduction branch=1,
# accumulator=0, MSB first, add x, normal bit branch, one pass.
CANONICAL = (1, 1, 1, 0, 0, 1, 0, 0)


@dataclass
class SearchResult:
    seed: int
    initial_contains_target: bool
    created_generation: int | None
    generated_target: bool
    train_exact: float
    unseen_modulus_exact: float
    unique_programs_evaluated: int
    generations: int


def field_parameters(model) -> list[Tensor]:
    return [getattr(model, name) for name in FIELD_NAMES]


def program_indices(model) -> Tensor:
    return torch.stack(
        [parameter.argmax(dim=-1) for parameter in field_parameters(model)],
        dim=1,
    )


def is_target(programs: Tensor) -> Tensor:
    target = programs.new_tensor(CANONICAL)
    return (programs == target[None]).all(dim=1)


def set_program(model, row: int, values: Tensor) -> None:
    with torch.no_grad():
        for parameter, value in zip(field_parameters(model), values, strict=True):
            parameter[row].fill_(-4.0)
            parameter[row, int(value)] = 4.0


def copy_program(model, destination: int, source: int) -> None:
    with torch.no_grad():
        for parameter in field_parameters(model):
            parameter[destination].copy_(parameter[source])


def random_program(generator: torch.Generator) -> Tensor:
    return torch.tensor(
        [
            int(torch.randint(size, (), generator=generator))
            for size in FIELD_SIZES
        ],
        dtype=torch.long,
    )


def force_target_absent(model, generator: torch.Generator) -> None:
    """Experimental intervention only: remove any lucky initial target."""
    programs = program_indices(model)
    for row in is_target(programs).nonzero(as_tuple=False).flatten().tolist():
        replacement = random_program(generator)
        while tuple(replacement.tolist()) == CANONICAL:
            replacement = random_program(generator)
        set_program(model, row, replacement)


def examples(moduli: range, values: range) -> tuple[Tensor, Tensor, Tensor]:
    pairs = [
        (modulus, value)
        for modulus in moduli
        for value in values
        if value < modulus
    ]
    modulus = torch.tensor([pair[0] for pair in pairs])
    value = torch.tensor([pair[1] for pair in pairs])
    target = value.square().remainder(modulus)
    return value, modulus, target


@torch.no_grad()
def endpoint_fitness(
    model,
    value: Tensor,
    modulus: Tensor,
    target: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return bit-match and row-exact fitness for every complete program."""
    width = max(2, int(modulus.max()).bit_length() + 1)
    value_bits = model._bits(value)[:, :width]
    modulus_bits = model._bits(modulus)[:, :width]
    particles = model.particles
    value_program = value_bits[None].expand(particles, -1, -1, -1)
    modulus_program = modulus_bits[None].expand_as(value_program)
    indices = torch.arange(particles)
    table = model._transition_tables(indices)
    orientation = model._programs(indices)[:, 2]
    branch = torch.stack((1.0 - orientation, orientation), dim=-1)
    recurrence = model._recurrence_programs(indices)
    output = model._recur(
        value_program, modulus_program, table, branch, recurrence
    ).argmax(dim=-1)
    target_bits = ((target[:, None] >> torch.arange(width)[None]) & 1).long()
    matched = (output == target_bits[None])
    bit_fitness = matched.float().mean(dim=(1, 2))
    exact_fitness = matched.all(dim=2).float().mean(dim=1)
    return bit_fitness, exact_fitness


def mutate(
    model,
    row: int,
    generator: torch.Generator,
    changes: int,
) -> None:
    values = program_indices(model)[row].clone()
    fields = torch.randperm(len(FIELD_SIZES), generator=generator)[:changes]
    for field in fields.tolist():
        old = int(values[field])
        proposal = int(torch.randint(FIELD_SIZES[field] - 1, (), generator=generator))
        values[field] = proposal + int(proposal >= old)
    set_program(model, row, values)


def search(
    seed: int,
    population: int = 96,
    generations: int = 160,
    elite_fraction: float = 0.25,
    restart_fraction: float = 0.20,
) -> SearchResult:
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(10_000 + seed)
    original_programs = BASE.PROGRAMS
    BASE.PROGRAMS = population
    try:
        model = BASE.build_model(ModelSpec(17, 10, 500_000_000))
    finally:
        BASE.PROGRAMS = original_programs
    model.eval()
    force_target_absent(model, generator)
    initial_contains = bool(is_target(program_indices(model)).any())
    if initial_contains:
        raise AssertionError("target-removal intervention failed")

    # Complete endpoint teaching set over several small moduli.  No carries,
    # residues, products, or intermediate states are supplied as labels.
    train_value, train_modulus, train_target = examples(range(2, 13), range(32))
    test_value, test_modulus, test_target = examples(range(13, 24), range(64))
    elite_count = max(2, int(population * elite_fraction))
    restart_count = max(1, int(population * restart_fraction))
    seen: set[tuple[int, ...]] = set()
    created_generation: int | None = None
    best_row = 0

    for generation in range(generations + 1):
        programs = program_indices(model)
        seen.update(tuple(row.tolist()) for row in programs)
        targets = is_target(programs)
        if bool(targets.any()) and created_generation is None:
            created_generation = generation

        bit_fitness, exact_fitness = endpoint_fitness(
            model, train_value, train_modulus, train_target
        )
        # Endpoint exactness is decisive; bit fitness provides a local slope.
        fitness = 2.0 * exact_fitness + bit_fitness
        order = fitness.argsort(descending=True)
        best_row = int(order[0])
        if float(exact_fitness[best_row]) == 1.0 and bool(targets[best_row]):
            break
        if generation == generations:
            break

        elites = order[:elite_count]
        children = order[elite_count:]
        for child_index, child in enumerate(children.tolist()):
            if child_index < restart_count:
                candidate = random_program(generator)
                set_program(model, child, candidate)
                continue
            parent = int(elites[int(torch.randint(elite_count, (), generator=generator))])
            copy_program(model, child, parent)
            # Mostly one-field refinements, with occasional two-field escapes.
            changes = 1 + int(torch.rand((), generator=generator) < 0.15)
            mutate(model, child, generator, changes)

    programs = program_indices(model)
    targets = is_target(programs)
    generated = bool(targets.any())
    if generated:
        best_row = int(targets.nonzero(as_tuple=False)[0])
    train_bit, train_exact = endpoint_fitness(
        model, train_value, train_modulus, train_target
    )
    test_bit, test_exact = endpoint_fitness(
        model, test_value, test_modulus, test_target
    )
    del train_bit, test_bit
    return SearchResult(
        seed=seed,
        initial_contains_target=initial_contains,
        created_generation=created_generation,
        generated_target=generated,
        train_exact=float(train_exact[best_row]),
        unseen_modulus_exact=float(test_exact[best_row]),
        unique_programs_evaluated=len(seen),
        generations=generation,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--population", type=int, default=96)
    parser.add_argument("--generations", type=int, default=160)
    args = parser.parse_args()
    results = [
        search(seed, args.population, args.generations)
        for seed in range(args.seeds)
    ]
    print(json.dumps([result.__dict__ for result in results], indent=2))


if __name__ == "__main__":
    main()
