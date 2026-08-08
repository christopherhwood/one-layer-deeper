"""Measure recurrence and reduction expressivity of the generated-program DSL."""

from __future__ import annotations

import itertools
import json
from collections.abc import Callable, Iterable

import torch

from benchmark import ModelSpec, OptimizerSpec, TokenLossBatch
from submissions.generated_program_easy import submission as MODEL


CONFIGURATIONS = list(
    itertools.product(*(range(size) for size in (5, 7, 2, 2, 2, 2, 2, 2)))
)
CUBE = (1, 1, 1, 0, 0, 1, 0, 1)


def _set_configuration(model, row: int, configuration: tuple[int, ...]) -> None:
    with torch.no_grad():
        for name, value in zip(MODEL.FIELD_NAMES, configuration, strict=True):
            parameter = getattr(model, name)
            parameter[row].fill_(-MODEL.MUTATION_LOGIT)
            parameter[row, value] = MODEL.MUTATION_LOGIT


@torch.no_grad()
def _language_outputs(
    model, pairs: list[tuple[int, int]]
) -> tuple[list[tuple[int, int]], torch.Tensor]:
    modulus = torch.tensor([row[0] for row in pairs])
    value = torch.tensor([row[1] for row in pairs])
    width = max(2, int(modulus.max()).bit_length() + 1)
    outputs = []
    fields = [getattr(model, name) for name in MODEL.FIELD_NAMES]
    for start in range(0, len(CONFIGURATIONS), MODEL.POPULATION):
        chunk = CONFIGURATIONS[start : start + MODEL.POPULATION]
        for row, configuration in enumerate(chunk):
            _set_configuration(model, row, configuration)
        for row in range(len(chunk), MODEL.POPULATION):
            for parameter in fields:
                parameter[row].copy_(parameter[0])
        value_bits = model._bits(value)[:, :width][None].expand(
            MODEL.POPULATION, -1, -1, -1
        )
        modulus_bits = model._bits(modulus)[:, :width][None].expand_as(value_bits)
        indices = torch.arange(MODEL.POPULATION)
        table = model._transition_tables(indices)
        orientation = model._programs(indices)[:, 2]
        branch = torch.stack((1.0 - orientation, orientation), dim=-1)
        recurrence = model._recurrence_programs(indices)
        bits = model._recur(
            value_bits, modulus_bits, table, branch, recurrence
        ).argmax(dim=-1)
        powers = 1 << torch.arange(width)
        outputs.append((bits * powers).sum(dim=-1)[: len(chunk)])
    return pairs, torch.cat(outputs)


def _audit_pairs(moduli: Iterable[int]) -> list[tuple[int, int]]:
    """Stratified nontrivial values without exhaustive-set collision inflation."""
    pairs = []
    for modulus in moduli:
        candidates = {
            2,
            3,
            modulus // 7,
            modulus // 5,
            modulus // 3,
            modulus // 2,
            2 * modulus // 3,
            4 * modulus // 5,
            modulus - 3,
            modulus - 2,
        }
        for index in range(10):
            candidates.add(2 + ((index * 37 + modulus * 11) % (modulus - 3)))
        pairs.extend(
            (modulus, value)
            for value in sorted(candidates)
            if 2 <= value <= modulus - 2
        )
    return pairs


def representability_matrix() -> list[dict[str, object]]:
    torch.manual_seed(74)
    torch.set_num_threads(1)
    model = MODEL.build_model(ModelSpec(17, 13, 500_000_000))
    model.eval()
    # The primary audit excludes x=0, x=1, tiny N, and exhaustive-set collision
    # inflation. It mixes prime/composite two-digit training moduli and freezes
    # the selected program before wider, entirely unseen OOD moduli.
    train_pairs = _audit_pairs(
        (17, 19, 23, 29, 31, 35, 41, 47, 53, 59, 71, 77, 83, 91, 97)
    )
    ood_pairs = _audit_pairs(
        (101, 103, 107, 109, 113, 121, 127, 131, 137, 143, 149, 163, 181, 187, 211, 221, 247, 251)
    )
    train_pairs, train_outputs = _language_outputs(model, train_pairs)
    ood_pairs, ood_outputs = _language_outputs(model, ood_pairs)
    functions: dict[str, tuple[Callable[[int, int], int], str]] = {
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
    results = []
    for name, (function, distance_kind) in functions.items():
        train_target = torch.tensor([function(*row) for row in train_pairs])
        ood_target = torch.tensor([function(*row) for row in ood_pairs])
        train_exact = (train_outputs == train_target).float().mean(dim=1)
        ood_exact = (ood_outputs == ood_target).float().mean(dim=1)
        train_prefix = _prefix_fraction(train_outputs, train_target).mean(dim=1)
        ood_prefix = _prefix_fraction(ood_outputs, ood_target).mean(dim=1)
        train_closeness = _numeric_closeness(
            train_outputs, train_target, train_pairs, distance_kind
        ).mean(dim=1)
        ood_closeness = _numeric_closeness(
            ood_outputs, ood_target, ood_pairs, distance_kind
        ).mean(dim=1)
        best_train = train_exact.max()
        train_optimal = train_exact == best_train
        # Training exactness is decisive, then the proposed leading-prefix and
        # numeric-closeness signals break ties. OOD never selects the program.
        selection = 100.0 * train_exact + train_prefix + 0.01 * train_closeness
        best_index = int(selection.argmax())
        results.append(
            {
                "function": name,
                "best_train_exact": float(best_train),
                "best_ood_exact_among_train_optima": float(
                    ood_exact[train_optimal].max()
                ),
                "selected_ood_exact": float(ood_exact[best_index]),
                "number_of_train_optimal_programs": int(train_optimal.sum()),
                "best_configuration": CONFIGURATIONS[best_index],
                "perfectly_expressible": bool(best_train == 1.0),
                "selected_train_prefix": float(train_prefix[best_index]),
                "selected_ood_prefix": float(ood_prefix[best_index]),
                "selected_train_numeric_closeness": float(
                    train_closeness[best_index]
                ),
                "selected_ood_numeric_closeness": float(
                    ood_closeness[best_index]
                ),
            }
        )
    return results


def _decimal_length(value: torch.Tensor) -> torch.Tensor:
    length = torch.ones_like(value, dtype=torch.long)
    threshold = 10
    # Stop before Python attempts to convert 10**19 into signed int64.
    while threshold <= (1 << 62) and bool((value >= threshold).any()):
        length += value >= threshold
        threshold *= 10
    return length


def _prefix_fraction(outputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target = target[None].expand_as(outputs)
    output_length = _decimal_length(outputs)
    target_length = _decimal_length(target)
    running = output_length == target_length
    matched = torch.zeros_like(outputs, dtype=torch.float)
    maximum_digits = int(target_length.max())
    for position in range(maximum_digits - 1, -1, -1):
        active = target_length > position
        same_digit = (
            outputs.div(10**position, rounding_mode="floor").remainder(10)
            == target.div(10**position, rounding_mode="floor").remainder(10)
        )
        running = running & (~active | same_digit)
        matched += (running & active).float()
    return matched / target_length.clamp_min(1)


def _numeric_closeness(
    outputs: torch.Tensor,
    target: torch.Tensor,
    pairs: list[tuple[int, int]],
    kind: str,
) -> torch.Tensor:
    modulus = torch.tensor([row[0] for row in pairs])[None]
    target = target[None]
    if kind in {"n", "n_plus_1"}:
        period = modulus + int(kind == "n_plus_1")
        prediction = outputs.remainder(period)
        distance = (prediction - target).abs()
        distance = torch.minimum(distance, period - distance)
        scale = (period // 2).clamp_min(1)
    else:
        distance = (outputs - target).abs()
        scale = (modulus - 1).clamp_min(1)
    return 1.0 - (distance.float() / scale).clamp(max=1.0)


def _cube_rows(model) -> torch.Tensor:
    indices = torch.stack(
        [getattr(model, name).argmax(dim=-1) for name in MODEL.FIELD_NAMES], dim=1
    )
    return (indices == indices.new_tensor(CUBE)[None]).all(dim=1).nonzero().flatten()


def _endpoint_batch(power: int = 3):
    pairs = [(modulus, value) for modulus in range(2, 13) for value in range(modulus)]

    def prompt(modulus: int, value: int) -> list[int]:
        tokens = [
            MODEL.N_MARK,
            *[MODEL.DIGIT_OFFSET + int(digit) for digit in str(modulus)],
            MODEL.X_MARK,
            *[MODEL.DIGIT_OFFSET + int(digit) for digit in str(value)],
            MODEL.T_MARK,
            MODEL.DIGIT_OFFSET + 1,
        ]
        return tokens + [MODEL.PAD] * (13 - len(tokens))

    inputs = torch.tensor([prompt(*row) for row in pairs])
    labels = torch.full((len(pairs), 2), -100, dtype=torch.long)
    valid = torch.zeros_like(labels, dtype=torch.bool)
    for row, (modulus, value) in enumerate(pairs):
        answer = str(value**power % modulus)
        labels[row, -len(answer) :] = torch.tensor(
            [MODEL.DIGIT_OFFSET + int(digit) for digit in answer]
        )
        valid[row, -len(answer) :] = True
    return inputs, labels, valid


def target_free_cube_generation(seed: int = 0) -> dict[str, object]:
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    model = MODEL.build_model(ModelSpec(17, 13, 500_000_000))
    model.train()
    initial_count = int(_cube_rows(model).numel())
    if initial_count:
        raise AssertionError(f"seed {seed} unexpectedly begins with a cube program")
    inputs, labels, valid = _endpoint_batch(3)
    optimizer = MODEL.build_optimizer(
        model, OptimizerSpec(1000.0, "cpu")
    ).optimizer
    created_generation = None
    for generation in range(1, 31):
        optimizer.zero_grad()
        logits, auxiliary = model(inputs, inputs != MODEL.PAD)
        loss = MODEL.token_training_loss(
            TokenLossBatch(logits[:, :2], labels, valid, None, auxiliary)
        )
        loss.backward()
        optimizer.step()
        if _cube_rows(model).numel():
            created_generation = generation
            break
    model.evolution_enabled = False
    for _ in range(25):
        optimizer.zero_grad()
        logits, auxiliary = model(inputs, inputs != MODEL.PAD)
        loss = MODEL.token_training_loss(
            TokenLossBatch(logits[:, :2], labels, valid, None, auxiliary)
        )
        loss.backward()
        optimizer.step()
    selected = int(model.program_logits.argmax())
    selected_is_cube = selected in _cube_rows(model).tolist()
    model.eval()
    moduli = torch.arange(13, 24)
    values = moduli - 1
    with torch.no_grad():
        actual = model._hard_execute(
            values, moduli, torch.ones_like(values), selected
        )
    expected = values.pow(3).remainder(moduli)
    return {
        "seed": seed,
        "initial_cube_programs": initial_count,
        "created_generation": created_generation,
        "selected_is_cube": selected_is_cube,
        "unseen_modulus_exact": float((actual == expected).float().mean()),
    }


if __name__ == "__main__":
    print(
        json.dumps(
            {
                "language_size": len(CONFIGURATIONS),
                "representability": representability_matrix(),
                "target_free_cube": target_free_cube_generation(),
            },
            indent=2,
        )
    )
