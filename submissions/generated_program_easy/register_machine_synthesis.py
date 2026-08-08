"""Endpoint-only synthesis in a compact general-purpose register machine."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from submissions.generated_program_easy.generality_probe import (
    _audit_pairs,
    _numeric_closeness,
    _prefix_fraction,
)


OP_NAMES = ("MOVE", "ADD", "SUB", "MUL", "DIV", "MOD", "AND", "MIN")
REGISTER_NAMES = ("x", "N", "zero", "one")
CAP = (1 << 62) - 1
TARGETS = {
    "identity_mod": (lambda modulus, value: value % modulus, "n"),
    "square_mod": (lambda modulus, value: value**2 % modulus, "n"),
    "cube_mod": (lambda modulus, value: value**3 % modulus, "n"),
    "fourth_power_mod": (lambda modulus, value: value**4 % modulus, "n"),
    "affine_3x_plus_1_mod": (
        lambda modulus, value: (3 * value + 1) % modulus,
        "n",
    ),
    "square_mod_n_plus_1": (
        lambda modulus, value: value**2 % (modulus + 1),
        "n_plus_1",
    ),
    "square_quotient_n": (lambda modulus, value: value**2 // modulus, "linear"),
    "square_saturating_n": (
        lambda modulus, value: min(value**2, modulus - 1),
        "linear",
    ),
    "square_and_n_minus_1": (
        lambda modulus, value: value**2 & (modulus - 1),
        "linear",
    ),
}
TRAIN_MODULI = (5, 7, 8, 9, 11, 13, 16, 17, 19, 23, 29, 31)
OOD_MODULI = (37, 41, 47, 53, 59, 61, 67, 71, 77, 83, 89, 97)


@dataclass
class Population:
    opcode: torch.Tensor
    source_a: torch.Tensor
    source_b: torch.Tensor
    output: torch.Tensor

    @property
    def size(self) -> int:
        return self.opcode.shape[0]

    @property
    def slots(self) -> int:
        return self.opcode.shape[1]


def random_population(
    size: int, slots: int, generator: torch.Generator
) -> Population:
    opcode = torch.randint(len(OP_NAMES), (size, slots), generator=generator)
    source_a = torch.empty(size, slots, dtype=torch.long)
    source_b = torch.empty_like(source_a)
    for slot in range(slots):
        source_a[:, slot] = torch.randint(4 + slot, (size,), generator=generator)
        source_b[:, slot] = torch.randint(4 + slot, (size,), generator=generator)
    output = torch.randint(4 + slots, (size,), generator=generator)
    return Population(opcode, source_a, source_b, output)


def _gather(registers: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    return registers.gather(
        2, source[:, None, None].expand(-1, registers.shape[1], 1)
    ).squeeze(2)


def safe_multiply(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Nonnegative saturating int64 multiply without evaluating overflow."""
    safe_right = torch.minimum(right, CAP // left.clamp_min(1))
    return left * safe_right


def execute(
    population: Population,
    pairs: list[tuple[int, int]],
    *,
    return_registers: bool = False,
) -> torch.Tensor:
    modulus = torch.tensor([row[0] for row in pairs])
    value = torch.tensor([row[1] for row in pairs])
    size = population.size
    registers = torch.stack(
        (
            value[None].expand(size, -1),
            modulus[None].expand(size, -1),
            torch.zeros(size, len(pairs), dtype=torch.long),
            torch.ones(size, len(pairs), dtype=torch.long),
        ),
        dim=2,
    )
    for slot in range(population.slots):
        left = _gather(registers, population.source_a[:, slot])
        right = _gather(registers, population.source_b[:, slot])
        denominator = right.clamp_min(1)
        candidates = torch.stack(
            (
                left,
                (left + right).clamp(max=CAP),
                (left - right).clamp(min=0),
                safe_multiply(left, right),
                torch.div(left, denominator, rounding_mode="floor"),
                left.remainder(denominator),
                torch.bitwise_and(left, right),
                torch.minimum(left, right),
            ),
            dim=2,
        )
        selected = candidates.gather(
            2,
            population.opcode[:, slot, None, None].expand(
                -1, len(pairs), 1
            ),
        ).squeeze(2)
        registers = torch.cat((registers, selected[:, :, None]), dim=2)
    if return_registers:
        return registers
    return _gather(registers, population.output)


def select_output_registers(
    population: Population,
    registers: torch.Tensor,
    target: torch.Tensor,
    pairs: list[tuple[int, int]],
    distance_kind: str,
) -> torch.Tensor:
    """Endpoint-guided coordinate update over every available register."""
    size, examples, register_count = registers.shape
    alternatives = registers.permute(0, 2, 1).reshape(
        size * register_count, examples
    )
    exact = (alternatives == target[None]).float().mean(dim=1)
    prefix = _prefix_fraction(alternatives, target).mean(dim=1)
    closeness = _numeric_closeness(
        alternatives, target, pairs, distance_kind
    ).mean(dim=1)
    score = (8.0 * exact + 2.0 * prefix + closeness).reshape(
        size, register_count
    )
    population.output.copy_(score.argmax(dim=1))
    return _gather(registers, population.output)


def endpoint_score(
    output: torch.Tensor,
    target: torch.Tensor,
    pairs: list[tuple[int, int]],
    distance_kind: str,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    exact = (output == target[None]).float().mean(dim=1)
    prefix = _prefix_fraction(output, target).mean(dim=1)
    closeness = _numeric_closeness(output, target, pairs, distance_kind).mean(dim=1)
    width = max(1, int(max(int(output.max()), int(target.max()))).bit_length())
    positions = torch.arange(width)
    output_bits = (output[:, :, None] >> positions) & 1
    target_bits = (target[None, :, None] >> positions) & 1
    bit_match = (output_bits == target_bits).float().mean(dim=(1, 2))
    # Exactness is decisive; the other endpoint-only views create slopes before
    # a complete program exists.
    score = 8.0 * exact + 2.0 * prefix + closeness + bit_match
    return score, {
        "exact": exact,
        "prefix": prefix,
        "closeness": closeness,
        "bit_match": bit_match,
    }


def posterior_gradient_fitness(score: torch.Tensor) -> torch.Tensor:
    """Obtain candidate fitness through the same endpoint-posterior gradient."""
    logits = torch.zeros(score.numel(), requires_grad=True)
    loss = -(F.softmax(logits, dim=0) * score.detach()).sum()
    loss.backward()
    assert logits.grad is not None
    return -logits.grad


def _copy_rows(
    population: Population, destinations: torch.Tensor, sources: torch.Tensor
) -> None:
    for tensor in (
        population.opcode,
        population.source_a,
        population.source_b,
        population.output,
    ):
        tensor[destinations] = tensor[sources]


def _randomize_rows(
    population: Population, rows: torch.Tensor, generator: torch.Generator
) -> None:
    replacement = random_population(rows.numel(), population.slots, generator)
    population.opcode[rows] = replacement.opcode
    population.source_a[rows] = replacement.source_a
    population.source_b[rows] = replacement.source_b
    population.output[rows] = replacement.output


def mutate_rows(
    population: Population,
    rows: torch.Tensor,
    generator: torch.Generator,
    mutations: int = 1,
) -> None:
    slots = population.slots
    genes = 3 * slots + 1
    for _ in range(mutations):
        chosen = torch.randint(genes, (rows.numel(),), generator=generator)
        for slot in range(slots):
            selected = rows[chosen == slot]
            if selected.numel():
                population.opcode[selected, slot] = torch.randint(
                    len(OP_NAMES), (selected.numel(),), generator=generator
                )
            selected = rows[chosen == slots + slot]
            if selected.numel():
                population.source_a[selected, slot] = torch.randint(
                    4 + slot, (selected.numel(),), generator=generator
                )
            selected = rows[chosen == 2 * slots + slot]
            if selected.numel():
                population.source_b[selected, slot] = torch.randint(
                    4 + slot, (selected.numel(),), generator=generator
                )
        selected = rows[chosen == genes - 1]
        if selected.numel():
            population.output[selected] = torch.randint(
                4 + slots, (selected.numel(),), generator=generator
            )


def extend_rows(
    population: Population, rows: torch.Tensor, generator: torch.Generator
) -> None:
    """Preserve the current expression and wrap it in one fresh operation."""
    for row in rows.tolist():
        current = int(population.output[row])
        earliest_slot = max(0, current - 3)
        if earliest_slot >= population.slots:
            continue
        slot = int(
            torch.randint(
                earliest_slot, population.slots, (), generator=generator
            )
        )
        population.opcode[row, slot] = torch.randint(
            len(OP_NAMES), (), generator=generator
        )
        if bool(torch.rand((), generator=generator) < 0.5):
            population.source_a[row, slot] = current
            population.source_b[row, slot] = torch.randint(
                4 + slot, (), generator=generator
            )
        else:
            population.source_b[row, slot] = current
            population.source_a[row, slot] = torch.randint(
                4 + slot, (), generator=generator
            )
        population.output[row] = 4 + slot


def describe(population: Population, row: int) -> list[str]:
    names = list(REGISTER_NAMES)
    program = []
    for slot in range(population.slots):
        destination = f"r{4 + slot}"
        opcode = OP_NAMES[int(population.opcode[row, slot])]
        left = names[int(population.source_a[row, slot])]
        right = names[int(population.source_b[row, slot])]
        program.append(f"{destination}={opcode}({left},{right})")
        names.append(destination)
    program.append(f"return {names[int(population.output[row])]}")
    return program


def synthesize(
    target_name: str,
    *,
    seed: int = 0,
    size: int = 1024,
    slots: int = 4,
    generations: int = 600,
) -> dict[str, object]:
    function, distance_kind = TARGETS[target_name]
    generator = torch.Generator().manual_seed(20_000 + seed)
    population = random_population(size, slots, generator)
    train_pairs = _audit_pairs(TRAIN_MODULI)
    ood_pairs = _audit_pairs(OOD_MODULI)
    train_target = torch.tensor([function(*row) for row in train_pairs])
    ood_target = torch.tensor([function(*row) for row in ood_pairs])
    elite_count = max(16, size // 8)
    restart_count = max(8, size // 8)
    initial_perfect = False
    created_generation = None
    unique_programs: set[tuple[int, ...]] = set()

    for generation in range(generations + 1):
        registers = execute(population, train_pairs, return_registers=True)
        output = select_output_registers(
            population, registers, train_target, train_pairs, distance_kind
        )
        score, metrics = endpoint_score(
            output, train_target, train_pairs, distance_kind
        )
        exact = metrics["exact"]
        perfect = exact == 1.0
        if generation == 0:
            initial_perfect = bool(perfect.any())
        if bool(perfect.any()):
            created_generation = generation
            best = int(perfect.nonzero()[0])
            break
        best = int(score.argmax())
        if generation == generations:
            break

        fitness = posterior_gradient_fitness(score)
        order = fitness.argsort(descending=True)
        elites = order[:elite_count]
        children = order[elite_count:]
        parents = elites[
            torch.randint(elite_count, (children.numel(),), generator=generator)
        ]
        _copy_rows(population, children, parents)
        restart_rows = children[:restart_count]
        _randomize_rows(population, restart_rows, generator)
        mutate = children[restart_count:]
        mutate_rows(population, mutate, generator)
        twice = mutate[
            torch.rand(mutate.numel(), generator=generator) < 0.20
        ]
        if twice.numel():
            mutate_rows(population, twice, generator)
        extensions = mutate[
            torch.rand(mutate.numel(), generator=generator) < 0.35
        ]
        if extensions.numel():
            extend_rows(population, extensions, generator)

        # Track actual generated syntax, not just generations elapsed.
        flattened = torch.cat(
            (
                population.opcode,
                population.source_a,
                population.source_b,
                population.output[:, None],
            ),
            dim=1,
        )
        unique_programs.update(tuple(row.tolist()) for row in flattened)

    train_output = execute(population, train_pairs)[best]
    ood_output = execute(population, ood_pairs)[best]
    train_exact = float((train_output == train_target).float().mean())
    ood_exact = float((ood_output == ood_target).float().mean())
    return {
        "target": target_name,
        "seed": seed,
        "initial_perfect": initial_perfect,
        "created_generation": created_generation,
        "train_exact": train_exact,
        "ood_exact": ood_exact,
        "unique_programs": len(unique_programs),
        "program": describe(population, best),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=(*TARGETS, "all"), default="all")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--population", type=int, default=1024)
    parser.add_argument("--slots", type=int, default=6)
    parser.add_argument("--generations", type=int, default=600)
    args = parser.parse_args()
    names = list(TARGETS) if args.target == "all" else [args.target]
    results = [
        synthesize(
            name,
            seed=args.seed,
            size=args.population,
            slots=args.slots,
            generations=args.generations,
        )
        for name in names
    ]
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
