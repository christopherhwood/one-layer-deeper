"""Endpoint-only latent-path marginalization probe for modular reduction.

This local diagnostic supplies exact normalized x^2 digits, then trains a
conditional finite-state decoder to predict x^2 mod N.  Carry/quotient-like
states receive no labels: forward/backward dynamic programming sums every
latent path compatible with the observed final remainder.  The probe tests
whether marginalization gives a better learning signal than committing to one
soft recurrent reduction trajectory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader


BASE = 10
WIDTH = 4
PRODUCT_WIDTH = 2 * WIDTH
ENCODER_WIDTH = 64
DEFAULT_STATES = 8
DATA_ROOT = Path(
    "data/generated/squaring_mod_new11_easy_bidirectional_variable_b1011_t123"
)


def digits_lsd(value: int, width: int) -> list[int]:
    digits = []
    for _ in range(width):
        digits.append(value % BASE)
        value //= BASE
    return digits


def load_rows(split: str) -> list[dict]:
    with (DATA_ROOT / f"{split}.jsonl").open() as handle:
        rows = [json.loads(line) for line in handle]
    if split in ("train", "test"):
        rows = [row for row in rows if row["time_steps"] == 1]
    return rows


def collate(rows: list[dict]) -> tuple[Tensor, Tensor, Tensor]:
    products = torch.tensor(
        [digits_lsd(row["x"] * row["x"], PRODUCT_WIDTH) for row in rows]
    )
    moduli = torch.tensor([digits_lsd(row["modulus"], WIDTH) for row in rows])
    targets = torch.tensor([digits_lsd(row["result"], WIDTH) for row in rows])
    return products, moduli, targets


class LatentPathReducer(nn.Module):
    """Conditional finite-state transducer over remainder digits."""

    def __init__(self, states: int) -> None:
        super().__init__()
        self.states = states
        self.encoder = nn.GRU(
            2 * BASE,
            ENCODER_WIDTH,
            batch_first=True,
            bidirectional=True,
        )
        self.feature_norm = nn.LayerNorm(2 * ENCODER_WIDTH)
        self.transition = nn.Sequential(
            nn.Linear(2 * ENCODER_WIDTH, 2 * ENCODER_WIDTH),
            nn.SiLU(),
            nn.Linear(2 * ENCODER_WIDTH, states * BASE * states),
        )
        self.initial = nn.Parameter(torch.zeros(states))

    def potentials(self, product_digits: Tensor, modulus_digits: Tensor) -> Tensor:
        product = F.one_hot(product_digits, BASE).float()
        modulus = F.one_hot(modulus_digits, BASE).float()
        modulus = F.pad(modulus, (0, 0, 0, PRODUCT_WIDTH - WIDTH))
        encoded, _ = self.encoder(torch.cat((product, modulus), dim=-1))
        # Each output significance sees its aligned encoder feature plus a
        # global summary. The transition network itself is tied across digits.
        features = encoded[:, :WIDTH] + encoded.mean(dim=1, keepdim=True)
        features = self.feature_norm(features)
        logits = self.transition(features).view(
            product.shape[0], WIDTH, self.states, BASE, self.states
        )
        # A locally normalized transition makes the sum over every complete
        # digit/state path exactly one, avoiding a separate partition estimate.
        return F.log_softmax(
            logits.flatten(start_dim=3), dim=-1
        ).view_as(logits)

    def target_nll(self, log_potential: Tensor, targets: Tensor) -> Tensor:
        batch = targets.shape[0]
        alpha = F.log_softmax(self.initial, dim=0).expand(batch, -1)
        for position in range(WIDTH):
            digit = targets[:, position]
            selected = log_potential[:, position].gather(
                2,
                digit[:, None, None, None].expand(
                    -1, self.states, 1, self.states
                ),
            ).squeeze(2)
            alpha = torch.logsumexp(alpha[:, :, None] + selected, dim=1)
        return -torch.logsumexp(alpha, dim=1)

    def digit_log_marginals(self, log_potential: Tensor) -> Tensor:
        batch = log_potential.shape[0]
        forward = [F.log_softmax(self.initial, dim=0).expand(batch, -1)]
        for position in range(WIDTH):
            scores = forward[-1][:, :, None, None] + log_potential[:, position]
            forward.append(torch.logsumexp(scores, dim=(1, 2)))

        backward: list[Tensor] = [torch.empty(0)] * (WIDTH + 1)
        backward[WIDTH] = torch.zeros(
            batch, self.states, device=log_potential.device
        )
        for position in range(WIDTH - 1, -1, -1):
            scores = (
                log_potential[:, position]
                + backward[position + 1][:, None, None, :]
            )
            backward[position] = torch.logsumexp(scores, dim=(2, 3))

        marginals = []
        for position in range(WIDTH):
            scores = (
                forward[position][:, :, None, None]
                + log_potential[:, position]
                + backward[position + 1][:, None, None, :]
            )
            digit_scores = torch.logsumexp(scores, dim=(1, 3))
            marginals.append(digit_scores - torch.logsumexp(digit_scores, dim=1, keepdim=True))
        return torch.stack(marginals, dim=1)

    def forward(self, product_digits: Tensor, modulus_digits: Tensor) -> Tensor:
        return self.digit_log_marginals(
            self.potentials(product_digits, modulus_digits)
        )


def exact(model: LatentPathReducer, rows: list[dict]) -> float:
    loader = DataLoader(rows, batch_size=256, collate_fn=collate)
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for products, moduli, targets in loader:
            prediction = model(products, moduli).argmax(dim=-1)
            correct += int((prediction == targets).all(dim=1).sum())
            total += targets.shape[0]
    return correct / total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--states", type=int, default=DEFAULT_STATES)
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
    model = LatentPathReducer(args.states)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1.5e-3, betas=(0.9, 0.95), weight_decay=0.1
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
        path_nll = model.target_nll(potentials, targets)
        marginals = model.digit_log_marginals(potentials)
        marginal_nll = F.nll_loss(
            marginals.transpose(1, 2), targets, reduction="none"
        ).mean(dim=1)
        loss = (0.75 * path_nll + 0.25 * marginal_nll).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 500 == 0:
            batch_exact = (marginals.argmax(-1) == targets).all(dim=1).float().mean()
            print(
                f"step={step} loss={loss.item():.4f} "
                f"batch_exact={batch_exact.item():.4f}",
                flush=True,
            )

    print(
        {
            "states": args.states,
            "train_t1": exact(model, train_rows),
            "test_t1": exact(model, test_rows),
            "depth_t1": exact(model, depth_rows),
        }
    )


if __name__ == "__main__":
    main()
