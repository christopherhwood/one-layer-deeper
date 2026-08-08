"""Bottom-up endpoint-only synthesis in the general register expression DSL."""

from __future__ import annotations

import json
from dataclasses import dataclass

import torch

from submissions.generated_program_easy.register_machine_synthesis import (
    CAP,
    OOD_MODULI,
    TARGETS,
    TRAIN_MODULI,
)


COMMUTATIVE = {"ADD", "MUL", "AND", "MIN"}
OPERATIONS = ("ADD", "SUB", "MUL", "DIV", "MOD", "AND", "MIN")


@dataclass(frozen=True)
class Expression:
    operation: str
    left: "Expression | None" = None
    right: "Expression | None" = None

    def __str__(self) -> str:
        if self.left is None:
            return self.operation
        return f"{self.operation}({self.left},{self.right})"


def exhaustive_pairs(moduli: tuple[int, ...]) -> list[tuple[int, int]]:
    return [
        (modulus, value)
        for modulus in moduli
        for value in range(2, modulus - 1)
    ]


def apply(operation: str, left: int, right: int) -> int:
    if operation == "ADD":
        return min(left + right, CAP)
    if operation == "SUB":
        return max(left - right, 0)
    if operation == "MUL":
        return min(left * right, CAP)
    if operation == "DIV":
        return left // max(right, 1)
    if operation == "MOD":
        return left % max(right, 1)
    if operation == "AND":
        return left & right
    if operation == "MIN":
        return min(left, right)
    raise ValueError(operation)


def evaluate(expression: Expression, pairs: list[tuple[int, int]]) -> tuple[int, ...]:
    if expression.left is None:
        if expression.operation == "x":
            return tuple(value for _, value in pairs)
        if expression.operation == "N":
            return tuple(modulus for modulus, _ in pairs)
        return (int(expression.operation),) * len(pairs)
    left = evaluate(expression.left, pairs)
    assert expression.right is not None
    right = evaluate(expression.right, pairs)
    return tuple(apply(expression.operation, a, b) for a, b in zip(left, right))


def build_archive(max_operations: int = 4):
    train_pairs = exhaustive_pairs(TRAIN_MODULI)
    terminals = tuple(Expression(name) for name in ("x", "N", "0", "1"))
    levels: list[dict[tuple[int, ...], Expression]] = [
        {evaluate(expression, train_pairs): expression for expression in terminals}
    ]
    archive = dict(levels[0])
    for size in range(1, max_operations + 1):
        level: dict[tuple[int, ...], Expression] = {}
        for left_size in range(size):
            right_size = size - 1 - left_size
            for left_behavior, left_expression in levels[left_size].items():
                for right_behavior, right_expression in levels[right_size].items():
                    for operation in OPERATIONS:
                        if operation in COMMUTATIVE and str(left_expression) > str(
                            right_expression
                        ):
                            continue
                        behavior = tuple(
                            apply(operation, left, right)
                            for left, right in zip(left_behavior, right_behavior)
                        )
                        if behavior in archive or behavior in level:
                            continue
                        level[behavior] = Expression(
                            operation, left_expression, right_expression
                        )
        levels.append(level)
        archive.update(level)
    return train_pairs, terminals, levels, archive


def synthesize(max_operations: int = 4) -> dict[str, object]:
    train_pairs, terminals, levels, archive = build_archive(max_operations)
    ood_pairs = exhaustive_pairs(OOD_MODULI)
    targets = {
        name: tuple(function(*row) for row in train_pairs)
        for name, (function, _) in TARGETS.items()
    }
    discovered: dict[str, dict[str, object]] = {}
    for size, level in enumerate(levels):
        for name, behavior in targets.items():
            if name not in discovered and behavior in level:
                discovered[name] = _result(
                    level[behavior], size, train_pairs, ood_pairs, name
                )
    level_sizes = [len(level) for level in levels]

    behaviors = list(archive)
    expressions = list(archive.values())
    behavior_tensor = torch.tensor(behaviors)
    # Run the actual posterior backward pass. Expected endpoint error is linear
    # in softmax weight, so a unique zero-error behavior should receive the
    # most-negative logit gradient without differentiating through integer ops.
    for name, result in discovered.items():
        target = torch.tensor([TARGETS[name][0](*row) for row in train_pairs])
        losses = (behavior_tensor != target[None]).float().mean(dim=1)
        logits = torch.zeros(len(losses), requires_grad=True)
        posterior_loss = (logits.softmax(dim=0) * losses).sum()
        posterior_loss.backward()
        assert logits.grad is not None
        order = logits.grad.argsort()
        selected = int(order[0])
        selected_expression = expressions[selected]
        selected_ood = evaluate(selected_expression, ood_pairs)
        ood_target = tuple(TARGETS[name][0](*row) for row in ood_pairs)
        result["endpoint_loss"] = float(losses[selected])
        result["selected_by_gradient"] = str(selected_expression)
        result["gradient_rank"] = 1
        result["gradient_margin_over_runner_up"] = float(
            logits.grad[order[1]] - logits.grad[order[0]]
        )
        result["gradient_selected_ood_exact"] = sum(
            a == b for a, b in zip(selected_ood, ood_target)
        ) / len(ood_target)
        result["strict_gradient_winner"] = (
            str(selected_expression) == result["expression"]
        )

    return {
        "initial_identity_available": True,
        "initial_nontrivial_complete_programs": 0,
        "terminals": [str(expression) for expression in terminals],
        "operations": list(OPERATIONS),
        "level_sizes": level_sizes,
        "semantic_archive_size": len(archive),
        "all_targets_discovered": len(discovered) == len(TARGETS),
        "targets": discovered,
    }


def _result(
    expression: Expression,
    size: int,
    train_pairs: list[tuple[int, int]],
    ood_pairs: list[tuple[int, int]],
    target_name: str,
) -> dict[str, object]:
    function = TARGETS[target_name][0]
    train_prediction = evaluate(expression, train_pairs)
    ood_prediction = evaluate(expression, ood_pairs)
    train_target = tuple(function(*row) for row in train_pairs)
    ood_target = tuple(function(*row) for row in ood_pairs)
    return {
        "operations_required": size,
        "created_after_initialization": size > 0,
        "expression": str(expression),
        "train_exact": sum(a == b for a, b in zip(train_prediction, train_target))
        / len(train_target),
        "ood_exact": sum(a == b for a, b in zip(ood_prediction, ood_target))
        / len(ood_target),
    }


if __name__ == "__main__":
    print(json.dumps(synthesize(), indent=2, sort_keys=True))
