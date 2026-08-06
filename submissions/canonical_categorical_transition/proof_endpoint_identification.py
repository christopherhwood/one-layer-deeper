"""Constructive endpoint-only identification of a canonical transition.

This diagnostic deliberately uses a bounded residue domain so every one-step
input can be covered.  A randomly initialized categorical transition table is
trained only on endpoints ``(x, N) -> x^2 mod N`` at T=1.  The learned table is
then tied across entirely unseen rollout depths.  No intermediate square is a
training target.

The exhaustive run demonstrates the positive theorem.  The partial-coverage
control uses the same model and optimizer but withholds some one-step states;
it demonstrates why the benchmark's much larger state space still needs a
compact, identifiable shared rule.
"""

from __future__ import annotations

import argparse
import random

import torch
import torch.nn.functional as F
from torch import Tensor, nn


DEFAULT_MAX_MODULUS = 31
DEFAULT_STEPS = 600
DEPTHS = (1, 2, 4, 8, 16, 32, 64)


def examples(max_modulus: int) -> tuple[Tensor, Tensor, Tensor]:
    moduli: list[int] = []
    values: list[int] = []
    targets: list[int] = []
    for modulus in range(2, max_modulus + 1):
        for value in range(modulus):
            moduli.append(modulus)
            values.append(value)
            targets.append(value * value % modulus)
    return (
        torch.tensor(moduli, dtype=torch.long),
        torch.tensor(values, dtype=torch.long),
        torch.tensor(targets, dtype=torch.long),
    )


class CanonicalTransitionTable(nn.Module):
    """A randomly initialized next-residue distribution for every input state."""

    def __init__(self, max_modulus: int) -> None:
        super().__init__()
        self.max_modulus = max_modulus
        self.logits = nn.Parameter(
            torch.empty(max_modulus + 1, max_modulus, max_modulus)
        )
        nn.init.normal_(self.logits, std=0.02)

    def forward(self, modulus: Tensor, value: Tensor) -> Tensor:
        logits = self.logits[modulus, value]
        output = torch.arange(
            self.max_modulus, device=logits.device
        )[None]
        return logits.masked_fill(output >= modulus[:, None], -1e4)

    def step(self, modulus: Tensor, value: Tensor) -> Tensor:
        return self(modulus, value).argmax(dim=-1)

    def rollout(self, modulus: Tensor, value: Tensor, depth: int) -> Tensor:
        for _ in range(depth):
            value = self.step(modulus, value)
        return value


def exact_rollout(
    model: CanonicalTransitionTable,
    modulus: Tensor,
    value: Tensor,
    depth: int,
) -> float:
    expected = value.clone()
    for _ in range(depth):
        expected = expected.square().remainder(modulus)
    predicted = model.rollout(modulus, value, depth)
    return float((predicted == expected).float().mean())


def train(
    max_modulus: int,
    steps: int,
    coverage: float,
    seed: int,
) -> tuple[CanonicalTransitionTable, Tensor, Tensor, Tensor, Tensor]:
    random.seed(seed)
    torch.manual_seed(seed)
    modulus, value, target = examples(max_modulus)
    permutation = torch.randperm(value.shape[0])
    train_count = max(1, int(round(coverage * value.shape[0])))
    train_index = permutation[:train_count]
    heldout_index = permutation[train_count:]
    model = CanonicalTransitionTable(max_modulus)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.08, betas=(0.9, 0.95), weight_decay=0.0
    )
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = model(modulus[train_index], value[train_index])
        loss = F.cross_entropy(logits, target[train_index])
        loss.backward()
        optimizer.step()
    return model, modulus, value, target, heldout_index


def report(max_modulus: int, steps: int, coverage: float, seed: int) -> None:
    model, modulus, value, target, heldout = train(
        max_modulus, steps, coverage, seed
    )
    with torch.no_grad():
        one_step = model.step(modulus, value)
        all_t1 = float((one_step == target).float().mean())
        heldout_t1 = (
            float((one_step[heldout] == target[heldout]).float().mean())
            if heldout.numel()
            else None
        )
        ladder = {
            depth: exact_rollout(model, modulus, value, depth)
            for depth in DEPTHS
        }
    print(
        {
            "coverage": coverage,
            "covered_states": value.numel() - heldout.numel(),
            "all_states": value.numel(),
            "all_t1_exact": all_t1,
            "withheld_t1_exact": heldout_t1,
            "unseen_depth_exact": ladder,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-modulus", type=int, default=DEFAULT_MAX_MODULUS)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--seed", type=int, default=74)
    args = parser.parse_args()
    report(args.max_modulus, args.steps, 1.0, args.seed)
    report(args.max_modulus, args.steps, 0.6, args.seed)


if __name__ == "__main__":
    main()
