"""Endpoint-only latent-quotient probe for modular reduction.

This is a local diagnostic, not a submission.  It supplies exact normalized
digits of ``x**2`` and enumerates every possible quotient ``q``.  The quotient
is never used as a training label.  Instead, a small proposer assigns a prior
to quotient digit strings and a tied local decoder explains the observed final
remainder from ``x**2`` and ``q * N``.  Training marginalizes all quotients.

Unlike an unconstrained latent-state CRF, every latent choice has a fixed
arithmetic interpretation.  The experiment asks whether endpoint supervision
can discover that interpretation when the hypothesis class cannot turn latent
states into arbitrary per-example memory.
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
MAX_QUOTIENT = 2048
DEFAULT_HIDDEN = 16
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
    targets = torch.tensor(
        [digits_lsd(row["result"], PRODUCT_WIDTH) for row in rows]
    )
    return products, moduli, targets


def quotient_digits() -> Tensor:
    return torch.tensor(
        [digits_lsd(value, WIDTH) for value in range(MAX_QUOTIENT)]
    )


class LatentQuotientReducer(nn.Module):
    """Marginalize grounded quotient strings, then decode P - qN locally."""

    def __init__(
        self, hidden: int = DEFAULT_HIDDEN, exact_multiple: bool = False
    ) -> None:
        super().__init__()
        self.hidden = hidden
        self.exact_multiple = exact_multiple
        self.proposer = nn.GRU(
            2 * BASE,
            hidden,
            batch_first=True,
        )
        self.quotient_positions = nn.Parameter(torch.randn(WIDTH, hidden) * 0.02)
        self.quotient_head = nn.Linear(hidden, BASE)

        # A shared 10x10 lookup is the only way N enters the decoder.  Pair
        # features are summed into schoolbook significance columns i+j, so a
        # quotient candidate cannot be an unstructured latent identifier.
        if not exact_multiple:
            self.pair_embedding = nn.Parameter(
                torch.randn(BASE, BASE, hidden) / hidden**0.5
            )
        self.decoder = nn.GRU(
            2 * BASE if exact_multiple else BASE + hidden,
            hidden,
            batch_first=True,
        )
        self.remainder_head = nn.Linear(hidden, BASE)
        self.register_buffer("candidates", quotient_digits(), persistent=False)

    def quotient_log_prior(self, products: Tensor, moduli: Tensor) -> Tensor:
        product = F.one_hot(products, BASE).float()
        modulus = F.one_hot(moduli, BASE).float()
        modulus = F.pad(modulus, (0, 0, 0, PRODUCT_WIDTH - WIDTH))
        _, final = self.proposer(torch.cat((product, modulus), dim=-1))
        position_features = torch.tanh(
            final[-1, :, None, :] + self.quotient_positions[None, :, :]
        )
        digit_log_probs = F.log_softmax(
            self.quotient_head(position_features), dim=-1
        )
        batch = products.shape[0]
        candidate_digits = self.candidates[None, :, :, None].expand(
            batch, -1, -1, 1
        )
        selected = digit_log_probs[:, None, :, :].expand(
            -1, MAX_QUOTIENT, -1, -1
        ).gather(3, candidate_digits).squeeze(3)
        scores = selected.sum(dim=-1)
        return scores - torch.logsumexp(scores, dim=1, keepdim=True)

    def decode_candidates(
        self,
        products: Tensor,
        moduli: Tensor,
        candidate_digits: Tensor,
    ) -> Tensor:
        """Return digit log probabilities for each row/candidate pairing."""

        batch, candidates, _ = candidate_digits.shape
        flat_n_digits = moduli[:, None, :].expand(
            -1, candidates, -1
        ).reshape(-1, WIDTH)
        flat_product_digits = products[:, None, :].expand(
            -1, candidates, -1
        ).reshape(-1, PRODUCT_WIDTH)
        flat_product = F.one_hot(flat_product_digits, BASE).float()
        if self.exact_multiple:
            # Diagnostic boundary only: q remains latent, but its multiplication
            # by N is supplied exactly so the learned component is solely a
            # tied digit-serial subtraction/borrow cell.
            place_values = flat_n_digits.new_tensor(
                [BASE**position for position in range(WIDTH)]
            )
            flat_n_values = (flat_n_digits * place_values).sum(dim=1)
            flat_q_values = (
                candidate_digits[..., 0]
                + BASE * candidate_digits[..., 1]
                + BASE**2 * candidate_digits[..., 2]
                + BASE**3 * candidate_digits[..., 3]
            ).reshape(-1)
            multiples = flat_q_values * flat_n_values
            multiple_digits = torch.stack(
                [
                    (multiples // BASE**position) % BASE
                    for position in range(PRODUCT_WIDTH)
                ],
                dim=1,
            )
            columns = F.one_hot(multiple_digits, BASE).float()
        else:
            flat_q = F.one_hot(candidate_digits.reshape(-1, WIDTH), BASE).float()
            flat_n = F.one_hot(flat_n_digits, BASE).float()
            # Learned pair interactions, placed at exact schoolbook
            # significance. Carry propagation remains the decoder's job.
            pairs = torch.einsum(
                "bid,bje,deh->bijh", flat_q, flat_n, self.pair_embedding
            )
            columns = pairs.new_zeros(
                batch * candidates, PRODUCT_WIDTH, self.hidden
            )
            for q_position in range(WIDTH):
                for n_position in range(WIDTH):
                    columns[:, q_position + n_position] += pairs[
                        :, q_position, n_position
                    ]
        decoded, _ = self.decoder(torch.cat((flat_product, columns), dim=-1))
        logits = self.remainder_head(decoded)
        return F.log_softmax(logits, dim=-1).view(
            batch, candidates, PRODUCT_WIDTH, BASE
        )

    def marginal_nll(
        self, products: Tensor, moduli: Tensor, targets: Tensor
    ) -> tuple[Tensor, Tensor]:
        prior = self.quotient_log_prior(products, moduli)
        candidates = self.candidates[None, :, :].expand(products.shape[0], -1, -1)
        log_digits = self.decode_candidates(products, moduli, candidates)
        selected = log_digits.gather(
            3,
            targets[:, None, :, None].expand(-1, MAX_QUOTIENT, -1, 1),
        ).squeeze(3)
        decoder_log_likelihood = selected.sum(dim=-1)
        nll = -torch.logsumexp(prior + decoder_log_likelihood, dim=1)
        return nll, prior

    def predict(self, products: Tensor, moduli: Tensor) -> tuple[Tensor, Tensor]:
        prior = self.quotient_log_prior(products, moduli)
        quotient = prior.argmax(dim=1)
        chosen = self.candidates[quotient][:, None, :]
        prediction = self.decode_candidates(products, moduli, chosen)
        return prediction[:, 0].argmax(dim=-1), quotient


def evaluate(
    model: LatentQuotientReducer, rows: list[dict]
) -> tuple[float, float]:
    loader = DataLoader(rows, batch_size=256, collate_fn=collate)
    correct = 0
    quotient_correct = 0
    total = 0
    offset = 0
    model.eval()
    with torch.no_grad():
        for products, moduli, targets in loader:
            prediction, quotient = model.predict(products, moduli)
            correct += int((prediction == targets).all(dim=1).sum())
            batch_rows = rows[offset : offset + products.shape[0]]
            true_quotient = torch.tensor(
                [row["x"] * row["x"] // row["modulus"] for row in batch_rows]
            )
            quotient_correct += int((quotient == true_quotient).sum())
            total += targets.shape[0]
            offset += products.shape[0]
    return correct / total, quotient_correct / total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=DEFAULT_HIDDEN)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--exact-multiple", action="store_true")
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
    model = LatentQuotientReducer(args.hidden, args.exact_multiple)
    optimizer = torch.optim.AdamW(
        model.parameters(),
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
        row_nll, prior = model.marginal_nll(products, moduli, targets)
        loss = row_nll.mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 100 == 0:
            print(
                f"step={step} loss={loss.item():.4f} "
                f"prior_entropy={(-(prior.exp() * prior).sum(1).mean()).item():.3f}",
                flush=True,
            )

    train_exact, train_quotient = evaluate(model, train_rows)
    test_exact, test_quotient = evaluate(model, test_rows)
    depth_exact, depth_quotient = evaluate(model, depth_rows)
    print(
        {
            "hidden": args.hidden,
            "exact_multiple": args.exact_multiple,
            "train_t1": train_exact,
            "test_t1": test_exact,
            "depth_t1": depth_exact,
            "train_quotient": train_quotient,
            "test_quotient": test_quotient,
            "depth_quotient": depth_quotient,
        }
    )


if __name__ == "__main__":
    main()
