"""Grounded latent-quotient/borrow FST diagnostic.

Exact x^2 and candidate qN digits are diagnostic scaffolding. Quotients and
borrow states are never training labels. A two-state, position-tied local
transducer scores the observed remainder for every quotient, and training
marginalizes both latent paths. After a uniform-prior decoder warm-up, a small
proposer learns from the detached target-conditioned quotient posterior.

This is deliberately not benchmark submission code: exact qN construction is
used to isolate latent credit assignment from multiplication.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader


BASE = 10
WIDTH = 4
PRODUCT_WIDTH = 2 * WIDTH
MAX_QUOTIENT = 2048
DATA_ROOT = Path(
    "data/generated/squaring_mod_new11_easy_bidirectional_variable_b1011_t123"
)


def digits_lsd(value: int, width: int) -> list[int]:
    result = []
    for _ in range(width):
        result.append(value % BASE)
        value //= BASE
    return result


def load_rows(split: str) -> list[dict]:
    with (DATA_ROOT / f"{split}.jsonl").open() as handle:
        rows = [json.loads(line) for line in handle]
    if split in ("train", "test"):
        rows = [row for row in rows if row["time_steps"] == 1]
    return rows


def collate(rows: list[dict]) -> tuple[Tensor, Tensor, Tensor]:
    return (
        torch.tensor(
            [digits_lsd(row["x"] * row["x"], PRODUCT_WIDTH) for row in rows]
        ),
        torch.tensor([digits_lsd(row["modulus"], WIDTH) for row in rows]),
        torch.tensor(
            [digits_lsd(row["result"], PRODUCT_WIDTH) for row in rows]
        ),
    )


def values_from_digits(digits: Tensor) -> Tensor:
    place = digits.new_tensor(
        [BASE**position for position in range(digits.shape[1])]
    )
    return (digits * place).sum(dim=1)


def implied_quotient(products: Tensor, moduli: Tensor, targets: Tensor) -> Tensor:
    return (values_from_digits(products) - values_from_digits(targets)) // values_from_digits(moduli)


class GroundedQuotientFST(nn.Module):
    def __init__(self, hidden: int = 24, states: int = 2) -> None:
        super().__init__()
        self.hidden = hidden
        self.states = states
        self.proposer = nn.GRU(2 * BASE, hidden, batch_first=True)
        self.quotient_positions = nn.Parameter(torch.randn(WIDTH, hidden) * 0.02)
        self.quotient_head = nn.Linear(hidden, BASE)

        self.product_embedding = nn.Embedding(BASE, hidden)
        self.multiple_embedding = nn.Embedding(BASE, hidden)
        self.transition = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, states * BASE * states),
        )
        candidates = torch.tensor(
            [digits_lsd(value, WIDTH) for value in range(MAX_QUOTIENT)]
        )
        self.register_buffer("candidates", candidates, persistent=False)

    def quotient_log_prior(self, products: Tensor, moduli: Tensor) -> Tensor:
        product = F.one_hot(products, BASE).float()
        modulus = F.one_hot(moduli, BASE).float()
        modulus = F.pad(modulus, (0, 0, 0, PRODUCT_WIDTH - WIDTH))
        _, final = self.proposer(torch.cat((product, modulus), dim=-1))
        features = torch.tanh(
            final[-1, :, None, :] + self.quotient_positions[None, :, :]
        )
        digit_log_probs = F.log_softmax(self.quotient_head(features), dim=-1)
        candidate_digits = self.candidates[None, :, :, None].expand(
            products.shape[0], -1, -1, 1
        )
        selected = digit_log_probs[:, None].expand(
            -1, MAX_QUOTIENT, -1, -1
        ).gather(3, candidate_digits).squeeze(3)
        scores = selected.sum(dim=-1)
        return scores - torch.logsumexp(scores, dim=1, keepdim=True)

    def exact_multiples(self, moduli: Tensor) -> Tensor:
        place = moduli.new_tensor([BASE**position for position in range(WIDTH)])
        modulus_values = (moduli * place).sum(dim=1)
        quotient_values = torch.arange(MAX_QUOTIENT, device=moduli.device)
        values = modulus_values[:, None] * quotient_values[None, :]
        return torch.stack(
            [
                (values // BASE**position) % BASE
                for position in range(PRODUCT_WIDTH)
            ],
            dim=2,
        )

    def potentials(self, products: Tensor, moduli: Tensor) -> Tensor:
        multiples = self.exact_multiples(moduli)
        product_digits = products[:, None, :].expand(-1, MAX_QUOTIENT, -1)
        features = (
            self.product_embedding(product_digits)
            + self.multiple_embedding(multiples)
        )
        logits = self.transition(features).view(
            products.shape[0],
            MAX_QUOTIENT,
            PRODUCT_WIDTH,
            self.states,
            BASE,
            self.states,
        )
        # Conditional distribution over (output digit, next state) for each
        # local input pair and current state.
        return F.log_softmax(logits.flatten(start_dim=4), dim=-1).view_as(logits)

    def target_log_likelihood(self, potentials: Tensor, targets: Tensor) -> Tensor:
        batch = targets.shape[0]
        alpha = potentials.new_full((batch, MAX_QUOTIENT, self.states), -1e9)
        alpha[:, :, 0] = 0.0
        for position in range(PRODUCT_WIDTH):
            digit = targets[:, position]
            selected = potentials[:, :, position].gather(
                3,
                digit[:, None, None, None, None].expand(
                    -1, MAX_QUOTIENT, self.states, 1, self.states
                ),
            ).squeeze(3)
            alpha = torch.logsumexp(alpha[:, :, :, None] + selected, dim=2)
        # Correct subtraction has no final borrow. Requiring the start/end
        # state grounds the otherwise permutation-symmetric state pair.
        return alpha[:, :, 0]

    def digit_marginals(self, potentials: Tensor) -> Tensor:
        """Output digit marginals for one chosen quotient per row."""
        batch = potentials.shape[0]
        forward = [potentials.new_full((batch, self.states), -1e9)]
        forward[0][:, 0] = 0.0
        for position in range(PRODUCT_WIDTH):
            scores = forward[-1][:, :, None, None] + potentials[:, position]
            forward.append(torch.logsumexp(scores, dim=(1, 2)))

        backward: list[Tensor] = [torch.empty(0)] * (PRODUCT_WIDTH + 1)
        backward[-1] = potentials.new_full((batch, self.states), -1e9)
        backward[-1][:, 0] = 0.0
        for position in range(PRODUCT_WIDTH - 1, -1, -1):
            scores = potentials[:, position] + backward[position + 1][
                :, None, None, :
            ]
            backward[position] = torch.logsumexp(scores, dim=(2, 3))

        result = []
        for position in range(PRODUCT_WIDTH):
            scores = (
                forward[position][:, :, None, None]
                + potentials[:, position]
                + backward[position + 1][:, None, None, :]
            )
            digits = torch.logsumexp(scores, dim=(1, 3))
            result.append(digits - torch.logsumexp(digits, dim=1, keepdim=True))
        return torch.stack(result, dim=1)

    def predict(self, products: Tensor, moduli: Tensor) -> tuple[Tensor, Tensor]:
        prior = self.quotient_log_prior(products, moduli)
        quotient = prior.argmax(dim=1)
        all_potentials = self.potentials(products, moduli)
        chosen = all_potentials[
            torch.arange(products.shape[0], device=products.device), quotient
        ]
        return self.digit_marginals(chosen).argmax(dim=-1), quotient


def evaluate(
    model: GroundedQuotientFST, rows: list[dict]
) -> tuple[float, float, float, float]:
    loader = DataLoader(rows, batch_size=32, collate_fn=collate)
    remainder_correct = 0
    prior_correct = 0
    posterior_correct = 0
    oracle_remainder_correct = 0
    total = 0
    offset = 0
    model.eval()
    with torch.no_grad():
        for products, moduli, targets in loader:
            prediction, quotient = model.predict(products, moduli)
            potentials = model.potentials(products, moduli)
            likelihood = model.target_log_likelihood(potentials, targets)
            posterior_quotient = likelihood.argmax(dim=1)
            rows_here = rows[offset : offset + products.shape[0]]
            truth = torch.tensor(
                [row["x"] * row["x"] // row["modulus"] for row in rows_here]
            )
            oracle_potentials = potentials[
                torch.arange(products.shape[0]), truth
            ]
            oracle_prediction = model.digit_marginals(
                oracle_potentials
            ).argmax(dim=-1)
            remainder_correct += int((prediction == targets).all(dim=1).sum())
            oracle_remainder_correct += int(
                (oracle_prediction == targets).all(dim=1).sum()
            )
            prior_correct += int((quotient == truth).sum())
            posterior_correct += int((posterior_quotient == truth).sum())
            total += products.shape[0]
            offset += products.shape[0]
    return (
        remainder_correct / total,
        prior_correct / total,
        posterior_correct / total,
        oracle_remainder_correct / total,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--warmup-steps", type=int, default=750)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=24)
    parser.add_argument("--states", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--oracle-quotient", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(74)
    torch.set_num_threads(4)

    train_rows = load_rows("train")
    test_rows = load_rows("test")
    depth_rows = load_rows("depth_t_1")
    loader = DataLoader(
        train_rows,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate,
        generator=torch.Generator().manual_seed(45),
    )
    model = GroundedQuotientFST(args.hidden, args.states)
    decoder_parameters = [
        *model.product_embedding.parameters(),
        *model.multiple_embedding.parameters(),
        *model.transition.parameters(),
    ]
    proposer_parameters = [
        *model.proposer.parameters(),
        model.quotient_positions,
        *model.quotient_head.parameters(),
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": decoder_parameters},
            {"params": proposer_parameters},
        ],
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=0.02,
    )
    iterator = iter(loader)
    for step in range(1, args.steps + 1):
        try:
            products, moduli, targets = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            products, moduli, targets = next(iterator)
        optimizer.zero_grad(set_to_none=True)
        potentials = model.potentials(products, moduli)
        likelihood = model.target_log_likelihood(potentials, targets)
        prior = model.quotient_log_prior(products, moduli)
        if args.oracle_quotient:
            truth = implied_quotient(products, moduli, targets)
            decoder_loss = -likelihood.gather(1, truth[:, None]).mean()
            proposer_loss = -prior.gather(1, truth[:, None]).mean()
        else:
            # Decoder always sees a uniform quotient prior. This prevents it
            # from co-adapting to an arbitrary early proposal code.
            decoder_loss = -torch.logsumexp(likelihood, dim=1).mean()
            decoder_loss = decoder_loss + math.log(MAX_QUOTIENT)
        if not args.oracle_quotient and step > args.warmup_steps:
            teacher = F.softmax(likelihood.detach(), dim=1)
            proposer_loss = -(teacher * prior).sum(dim=1).mean()
        elif not args.oracle_quotient:
            proposer_loss = prior.sum() * 0.0
        loss = decoder_loss + proposer_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 100 == 0:
            teacher_entropy = -(
                F.softmax(likelihood, 1) * F.log_softmax(likelihood, 1)
            ).sum(1).mean()
            print(
                f"step={step} decoder={decoder_loss.item():.4f} "
                f"proposer={proposer_loss.item():.4f} "
                f"posterior_entropy={teacher_entropy.item():.3f}",
                flush=True,
            )

    names = ("remainder", "prior_q", "posterior_q", "oracle_remainder")
    print(
        {
            "hidden": args.hidden,
            "states": args.states,
            "oracle_quotient": args.oracle_quotient,
            "train_t1": dict(zip(names, evaluate(model, train_rows))),
            "test_t1": dict(zip(names, evaluate(model, test_rows))),
            "depth_t1": dict(zip(names, evaluate(model, depth_rows))),
        }
    )


if __name__ == "__main__":
    main()
