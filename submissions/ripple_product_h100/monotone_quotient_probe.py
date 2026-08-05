"""Proposer-free monotone quotient search with endpoint-only supervision.

This diagnostic supplies exact product and candidate qN digits, but never uses
the true quotient or borrow path in its loss. A tied two-state local FST models
subtraction. The quotient is the learned boundary between candidates classified
as nonnegative (P-qN) and negative (P-(q+1)N), removing the failed global
quotient proposer from ``latent_quotient_fst_probe.py``.

The structural learning signals are quotient-order monotonicity, two endpoint
anchors, an annealed boundary distribution, and a generic copy/residual bias in
the local transition. Exact qN enumeration makes this a local diagnostic rather
than a rules-valid benchmark submission.
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader

from latent_quotient_fst_probe import (
    BASE,
    MAX_QUOTIENT,
    PRODUCT_WIDTH,
    WIDTH,
    collate,
    digits_lsd,
    implied_quotient,
    load_rows,
)


class MonotoneQuotientReducer(nn.Module):
    def __init__(
        self,
        hidden: int = 24,
        states: int = 2,
        separate_comparator: bool = False,
    ) -> None:
        super().__init__()
        self.hidden = hidden
        self.states = states
        self.separate_comparator = separate_comparator
        self.product_embedding = nn.Embedding(BASE, hidden)
        self.multiple_embedding = nn.Embedding(BASE, hidden)
        self.transition = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, states * BASE * states),
        )
        # Starts each local transition near a copy/state-preserving map. The
        # learned branch can override it, as in a residual network.
        self.copy_strength = nn.Parameter(torch.tensor(1.5))
        if separate_comparator:
            self.comparator = nn.GRU(2 * BASE, hidden, batch_first=True)
            self.comparator_head = nn.Linear(hidden, 1)
        candidates = torch.tensor(
            [digits_lsd(value, WIDTH) for value in range(MAX_QUOTIENT)]
        )
        self.register_buffer("candidates", candidates, persistent=False)

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
        product_one_hot = F.one_hot(product_digits, BASE).to(logits.dtype)
        same_state = torch.eye(
            self.states, device=logits.device, dtype=logits.dtype
        )
        copy_route = (
            product_one_hot[:, :, :, None, :, None]
            * same_state[None, None, None, :, None, :]
        )
        logits = logits + F.softplus(self.copy_strength) * copy_route
        return F.log_softmax(logits.flatten(start_dim=4), dim=-1).view_as(logits)

    def final_state_log_probs(self, potentials: Tensor) -> Tensor:
        batch, candidates = potentials.shape[:2]
        alpha = potentials.new_full((batch, candidates, self.states), -1e9)
        alpha[:, :, 0] = 0.0
        for position in range(PRODUCT_WIDTH):
            scores = alpha[:, :, :, None, None] + potentials[:, :, position]
            alpha = torch.logsumexp(scores, dim=(2, 3))
        return alpha - torch.logsumexp(alpha, dim=2, keepdim=True)

    def target_log_likelihood(self, potentials: Tensor, targets: Tensor) -> Tensor:
        batch, candidates = potentials.shape[:2]
        alpha = potentials.new_full((batch, candidates, self.states), -1e9)
        alpha[:, :, 0] = 0.0
        for position in range(PRODUCT_WIDTH):
            digit = targets[:, position]
            selected = potentials[:, :, position].gather(
                3,
                digit[:, None, None, None, None].expand(
                    -1, candidates, self.states, 1, self.states
                ),
            ).squeeze(3)
            alpha = torch.logsumexp(alpha[:, :, :, None] + selected, dim=2)
        return alpha[:, :, 0]

    def boundary_log_probs(
        self, final_states: Tensor, temperature: float
    ) -> Tensor:
        # Candidate q is valid when qN is nonnegative but (q+1)N crosses into
        # the negative/borrow state. The last enumerated value acts as a guard.
        scores = final_states[:, :-1, 0] + final_states[:, 1:, 1]
        return F.log_softmax(scores / temperature, dim=1)

    def comparison_log_probs(self, products: Tensor, moduli: Tensor) -> Tensor:
        multiples = self.exact_multiples(moduli)
        product_digits = products[:, None, :].expand(-1, MAX_QUOTIENT, -1)
        inputs = torch.cat(
            (
                F.one_hot(product_digits, BASE),
                F.one_hot(multiples, BASE),
            ),
            dim=-1,
        ).float()
        inputs = inputs.flip(2).reshape(-1, PRODUCT_WIDTH, 2 * BASE)
        _, final = self.comparator(inputs)
        logits = self.comparator_head(final[-1]).view(products.shape[0], -1)
        return torch.stack((F.logsigmoid(logits), F.logsigmoid(-logits)), dim=2)

    def digit_log_marginals(self, potentials: Tensor) -> Tensor:
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

    def losses(
        self,
        products: Tensor,
        moduli: Tensor,
        targets: Tensor,
        temperature: float,
        consistency_weight: float,
        oracle_quotient: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        potentials = self.potentials(products, moduli)
        final_states = (
            self.comparison_log_probs(products, moduli)
            if self.separate_comparator
            else self.final_state_log_probs(potentials)
        )
        boundary = self.boundary_log_probs(final_states, temperature)
        target_likelihood = self.target_log_likelihood(potentials, targets)[:, :-1]
        if oracle_quotient is None:
            endpoint = -torch.logsumexp(
                boundary + target_likelihood, dim=1
            ).mean()
            boundary_supervision = endpoint.new_zeros(())
        else:
            endpoint = -target_likelihood.gather(
                1, oracle_quotient[:, None]
            ).mean()
            boundary_supervision = -boundary.gather(
                1, oracle_quotient[:, None]
            ).mean()

        nonnegative = final_states[:, :, 0].exp()
        anchors = -(final_states[:, 0, 0] + final_states[:, -1, 1]).mean()
        violations = F.relu(nonnegative[:, 1:] - nonnegative[:, :-1]).mean()
        # Encourage an actual boundary rather than a nearly-flat monotone curve.
        uncertainty = (nonnegative * (1.0 - nonnegative)).mean()
        consistency = anchors + 10.0 * violations + 0.1 * uncertainty
        total = endpoint + boundary_supervision + consistency_weight * consistency
        metrics = {
            "endpoint": endpoint.detach(),
            "boundary_supervision": boundary_supervision.detach(),
            "anchors": anchors.detach(),
            "violations": violations.detach(),
            "boundary_entropy": -(boundary.exp() * boundary).sum(1).mean().detach(),
        }
        return total, metrics

    def predict(self, products: Tensor, moduli: Tensor) -> tuple[Tensor, Tensor]:
        potentials = self.potentials(products, moduli)
        final_states = (
            self.comparison_log_probs(products, moduli)
            if self.separate_comparator
            else self.final_state_log_probs(potentials)
        )
        quotient = self.boundary_log_probs(final_states, 0.08).argmax(dim=1)
        chosen = potentials[
            torch.arange(products.shape[0], device=products.device), quotient
        ]
        prediction = self.digit_log_marginals(chosen).argmax(dim=-1)
        return prediction, quotient


def evaluate(
    model: MonotoneQuotientReducer, rows: list[dict]
) -> tuple[float, float, float]:
    loader = DataLoader(rows, batch_size=32, collate_fn=collate)
    remainder_correct = 0
    quotient_correct = 0
    oracle_remainder_correct = 0
    total = 0
    offset = 0
    model.eval()
    with torch.no_grad():
        for products, moduli, targets in loader:
            prediction, quotient = model.predict(products, moduli)
            rows_here = rows[offset : offset + products.shape[0]]
            truth = torch.tensor(
                [row["x"] * row["x"] // row["modulus"] for row in rows_here]
            )
            potentials = model.potentials(products, moduli)
            oracle = potentials[torch.arange(products.shape[0]), truth]
            oracle_prediction = model.digit_log_marginals(oracle).argmax(dim=-1)
            remainder_correct += int((prediction == targets).all(dim=1).sum())
            quotient_correct += int((quotient == truth).sum())
            oracle_remainder_correct += int(
                (oracle_prediction == targets).all(dim=1).sum()
            )
            total += products.shape[0]
            offset += products.shape[0]
    return (
        remainder_correct / total,
        quotient_correct / total,
        oracle_remainder_correct / total,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=24)
    parser.add_argument("--states", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--consistency-weight", type=float, default=0.25)
    parser.add_argument("--oracle-quotient", action="store_true")
    parser.add_argument("--separate-comparator", action="store_true")
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
    model = MonotoneQuotientReducer(
        args.hidden, args.states, args.separate_comparator
    )
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
        fraction = step / args.steps
        temperature = max(0.12, 1.5 * (0.12 / 1.5) ** fraction)
        consistency_weight = args.consistency_weight * min(1.0, step / 250)
        optimizer.zero_grad(set_to_none=True)
        quotient = (
            implied_quotient(products, moduli, targets)
            if args.oracle_quotient
            else None
        )
        loss, metrics = model.losses(
            products,
            moduli,
            targets,
            temperature,
            consistency_weight,
            quotient,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 100 == 0:
            print(
                f"step={step} loss={loss.item():.4f} "
                f"endpoint={metrics['endpoint'].item():.4f} "
                f"boundary={metrics['boundary_supervision'].item():.4f} "
                f"anchors={metrics['anchors'].item():.3f} "
                f"violations={metrics['violations'].item():.5f} "
                f"entropy={metrics['boundary_entropy'].item():.3f} "
                f"tau={temperature:.3f}",
                flush=True,
            )

    names = ("remainder", "quotient", "oracle_remainder")
    print(
        {
            "hidden": args.hidden,
            "states": args.states,
            "consistency_weight": args.consistency_weight,
            "oracle_quotient": args.oracle_quotient,
            "separate_comparator": args.separate_comparator,
            "train_t1": dict(zip(names, evaluate(model, train_rows))),
            "test_t1": dict(zip(names, evaluate(model, test_rows))),
            "depth_t1": dict(zip(names, evaluate(model, depth_rows))),
        }
    )


if __name__ == "__main__":
    main()
