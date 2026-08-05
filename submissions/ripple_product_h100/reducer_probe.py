"""Oracle-product probe for a structured, digit-serial learned reducer.

This is a local architecture diagnostic, not a submission.  It supplies the
normalized decimal digits of x^2 so the experiment measures only whether the
reducer can learn length-independent modular reduction from final remainders.
No quotient digits or intermediate remainders are supervised.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader


BASE = 10
WIDTH = 4
PRODUCT_WIDTH = 2 * WIDTH
DEFAULT_HIDDEN = 64
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
    products = torch.tensor(
        [digits_lsd(row["x"] * row["x"], PRODUCT_WIDTH) for row in rows]
    )
    moduli = torch.tensor([digits_lsd(row["modulus"], WIDTH) for row in rows])
    targets = torch.tensor([digits_lsd(row["result"], WIDTH) for row in rows])
    return products, moduli, targets


class CorrectionStep(nn.Module):
    """Learn one long-division quotient digit and corrected remainder."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.inspect = nn.GRU(
            2 * BASE, hidden, batch_first=True, bidirectional=True
        )
        self.quotient = nn.Linear(2 * hidden, BASE)
        self.quotient_embedding = nn.Linear(BASE, hidden, bias=False)
        self.correct = nn.GRU(
            2 * BASE + hidden,
            hidden,
            batch_first=True,
            bidirectional=True,
        )
        self.output = nn.Linear(2 * hidden, BASE)
        self.inspect_gate = nn.Parameter(torch.tensor(-1.5))

    def forward(
        self, shifted: Tensor, modulus: Tensor, temperature: float
    ) -> tuple[Tensor, Tensor]:
        inspected, _ = self.inspect(torch.cat((shifted, modulus), dim=-1))
        # Both ends are visible: the high end determines the quotient while the
        # low end contains the digit injected by the long-division shift.
        summary = inspected[:, 0] + inspected[:, -1]
        quotient_logits = self.quotient(summary)
        quotient = F.softmax(quotient_logits / temperature, dim=-1)
        quotient_features = self.quotient_embedding(quotient)[:, None, :]
        quotient_features = quotient_features.expand(-1, WIDTH, -1)
        corrected, _ = self.correct(
            torch.cat((shifted, modulus, quotient_features), dim=-1)
        )
        # Preserve a short route from the inspected digits while the learned
        # correction sharpens.
        corrected = corrected + torch.sigmoid(self.inspect_gate) * inspected
        logits = self.output(corrected)
        return F.softmax(logits / temperature, dim=-1), quotient_logits


class DigitSerialReducer(nn.Module):
    def __init__(self, hidden: int = DEFAULT_HIDDEN) -> None:
        super().__init__()
        self.step = CorrectionStep(hidden)

    def forward(
        self, product_digits: Tensor, modulus_digits: Tensor, temperature: float
    ) -> tuple[Tensor, tuple[Tensor, ...]]:
        product = F.one_hot(product_digits, BASE).float()
        modulus = F.one_hot(modulus_digits, BASE).float()
        remainder = torch.zeros(
            product.shape[0], WIDTH, BASE, device=product.device
        )
        remainder[:, :, 0] = 1.0
        quotient_logits: list[Tensor] = []
        # Long division consumes the most-significant product digit first. The
        # structural shift is generic positional state movement; the quotient
        # and correction themselves remain learned from endpoint labels.
        for position in range(PRODUCT_WIDTH - 1, -1, -1):
            shifted = torch.cat(
                (product[:, position : position + 1], remainder[:, :-1]), dim=1
            )
            remainder, quotient = self.step(shifted, modulus, temperature)
            quotient_logits.append(quotient)
        return remainder, tuple(quotient_logits)


def exact(model: DigitSerialReducer, rows: list[dict], temperature: float) -> float:
    loader = DataLoader(rows, batch_size=256, collate_fn=collate)
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for products, moduli, targets in loader:
            output, _ = model(products, moduli, temperature)
            correct += int((output.argmax(-1) == targets).all(dim=1).sum())
            total += targets.shape[0]
    return correct / total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=DEFAULT_HIDDEN)
    args = parser.parse_args()
    random.seed(74)
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
    model = DigitSerialReducer(args.hidden)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=2e-3, betas=(0.9, 0.95), weight_decay=0.02
    )
    iterator = iter(loader)
    for step in range(1, args.steps + 1):
        try:
            products, moduli, targets = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            products, moduli, targets = next(iterator)
        fraction = step / args.steps
        temperature = max(0.25, 1.25 * (0.25 / 1.25) ** fraction)
        optimizer.zero_grad(set_to_none=True)
        output, _ = model(products, moduli, temperature)
        losses = F.nll_loss(
            output.clamp_min(1e-8).log().transpose(1, 2),
            targets,
            reduction="none",
        )
        # Exactness-oriented objective: emphasize the worst digit and hardest
        # rows without inventing any intermediate arithmetic targets.
        row_loss = 0.5 * losses.mean(dim=1) + 0.5 * losses.max(dim=1).values
        keep = max(1, row_loss.shape[0] // 2)
        loss = 0.6 * row_loss.mean() + 0.4 * row_loss.topk(keep).values.mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 300 == 0:
            batch_exact = (output.argmax(-1) == targets).all(dim=1).float().mean()
            print(
                f"step={step} loss={loss.item():.4f} "
                f"batch_exact={batch_exact.item():.4f} tau={temperature:.3f}",
                flush=True,
            )

    print(
        {
            "hidden": args.hidden,
            "train_t1": exact(model, train_rows, 0.12),
            "test_t1": exact(model, test_rows, 0.12),
            "depth_t1": exact(model, depth_rows, 0.12),
        }
    )


if __name__ == "__main__":
    main()
