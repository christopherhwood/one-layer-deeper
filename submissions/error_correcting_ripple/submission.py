"""Endpoint-trained anchored ripple with learned latent error correction.

One weight-tied digit transition first proposes a modular square, snaps the
proposal to an observable digit state, and then re-runs on that state while
retaining the original operand and modulus.  During training a small random
subset of T=1 digit states is replaced between proposal and correction.  Thus
the correction pass cannot merely preserve its input: it must learn to recover
using the same generic recurrent computation that produced the proposal.
"""

from __future__ import annotations

import math
import time

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from benchmark import (
    ModelSpec,
    OptimizerBundle,
    OptimizerSpec,
    Submission,
    TokenLossBatch,
    assert_model_state,
)


PAD = 0
N_MARK = 2
X_MARK = 3
T_MARK = 4
DIGIT_OFFSET = 7
NUM_DIGITS = 10
HIDDEN = 48
MAX_OUTER_STEPS = 64
TRAIN_OUTER_STEPS = 3
CORRUPTION_RATE = 0.15
BASE_LR = 1.5e-3


class Config:
    def __init__(self, vocab_size: int, max_seq_len: int) -> None:
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len


class RMSNorm(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, value: Tensor) -> Tensor:
        return F.rms_norm(value, (value.shape[-1],), self.weight)


class ErrorCorrectingSquare(nn.Module):
    """A shared proposal/correction transition over observable digit states."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = width
        self.columns = 2 * width - 1
        self.pair_embedding = nn.Parameter(
            torch.empty(NUM_DIGITS, NUM_DIGITS, HIDDEN)
        )
        nn.init.normal_(self.pair_embedding, std=HIDDEN**-0.5)
        self.input_projection = nn.Linear(
            2 * NUM_DIGITS + HIDDEN + 3, HIDDEN, bias=False
        )
        self.scan = nn.GRU(
            HIDDEN, HIDDEN, batch_first=True, bidirectional=True
        )
        self.scan_norm = RMSNorm(2 * HIDDEN)
        self.scan_output = nn.Linear(2 * HIDDEN, HIDDEN, bias=False)
        self.delta = nn.Linear(HIDDEN, NUM_DIGITS)
        self.gate = nn.Linear(HIDDEN, NUM_DIGITS)
        self.gate_bias = nn.Parameter(torch.full((NUM_DIGITS,), -1.5))

    def _product_columns(self, operand: Tensor) -> Tensor:
        batch = operand.shape[0]
        pairs = torch.einsum(
            "bid,bje,deh->bijh", operand, operand, self.pair_embedding
        )
        significance = (
            torch.arange(self.width, device=operand.device)[:, None]
            + torch.arange(self.width, device=operand.device)[None, :]
        ).reshape(1, self.width * self.width, 1)
        significance = significance.expand(batch, -1, HIDDEN)
        columns = torch.zeros(
            batch,
            self.columns,
            HIDDEN,
            device=operand.device,
            dtype=operand.dtype,
        )
        return columns.scatter_add(
            1, significance, pairs.reshape(batch, -1, HIDDEN)
        )

    @staticmethod
    def _anchor(probabilities: Tensor) -> Tensor:
        hard = F.one_hot(
            probabilities.argmax(dim=-1), NUM_DIGITS
        ).to(probabilities.dtype)
        return probabilities + (hard - probabilities).detach()

    def _pass(
        self,
        state: Tensor,
        modulus: Tensor,
        product: Tensor,
        correction: bool,
        temperature: float,
    ) -> Tensor:
        boundary = torch.zeros(
            self.width, 2, device=state.device, dtype=state.dtype
        )
        boundary[0, 0] = 1.0
        boundary[-1, 1] = 1.0
        boundary = boundary[None].expand(state.shape[0], -1, -1)
        mode = torch.full(
            (state.shape[0], self.width, 1),
            float(correction),
            device=state.device,
            dtype=state.dtype,
        )
        logits = state.clamp_min(1e-6).log()
        for column_index in range(self.columns - 1, -1, -1):
            beliefs = F.softmax(logits / temperature, dim=-1)
            column = product[:, column_index, None].expand(-1, self.width, -1)
            features = self.input_projection(
                torch.cat((beliefs, modulus, column, boundary, mode), dim=-1)
            )
            scanned, _ = self.scan(features)
            hidden = self.scan_output(self.scan_norm(scanned))
            rate = torch.sigmoid(self.gate(hidden) + self.gate_bias)
            logits = F.log_softmax(logits + rate * self.delta(hidden), dim=-1)
        return F.softmax(logits / temperature, dim=-1)

    def forward(
        self,
        operand: Tensor,
        modulus: Tensor,
        temperature: float,
        corrupt_rows: Tensor,
        evaluation_corrections: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        product = self._product_columns(operand)
        proposal = self._pass(
            operand, modulus, product, correction=False, temperature=temperature
        )
        anchored = self._anchor(proposal)
        if self.training:
            corrupt = (
                torch.rand(
                    anchored.shape[:2], device=anchored.device
                ) < CORRUPTION_RATE
            ) & corrupt_rows[:, None]
            replacement_digit = torch.randint(
                NUM_DIGITS, anchored.shape[:2], device=anchored.device
            )
            replacement = F.one_hot(
                replacement_digit, NUM_DIGITS
            ).to(anchored.dtype)
            anchored = torch.where(corrupt[:, :, None], replacement, anchored)

        corrected = self._pass(
            anchored,
            modulus,
            product,
            correction=True,
            temperature=temperature,
        )
        candidates = [proposal, corrected]
        current = corrected
        for _ in range(evaluation_corrections):
            current = self._pass(
                self._anchor(current),
                modulus,
                product,
                correction=True,
                temperature=temperature,
            )
            candidates.append(current)

        if self.training:
            final = corrected
        else:
            stacked = torch.stack(candidates, dim=1)
            confidence = stacked.amax(dim=-1).mean(dim=-1)
            chosen = confidence.argmax(dim=1)
            rows = torch.arange(stacked.shape[0], device=stacked.device)
            final = stacked[rows, chosen]
        return final, proposal, corrected


class Model(nn.Module):
    num_loops = 1

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.width = max(2, (spec.max_seq_len - 4) // 2)
        self.square = ErrorCorrectingSquare(self.width)
        self.training_temperature = 1.2
        self.eval_temperature = 0.12

    @staticmethod
    def _time_steps(input_ids: Tensor, valid: Tensor) -> Tensor:
        value = torch.zeros(
            input_ids.shape[0], device=input_ids.device, dtype=torch.long
        )
        reading = torch.zeros_like(value, dtype=torch.bool)
        stopped = torch.zeros_like(value, dtype=torch.bool)
        for position in range(input_ids.shape[1]):
            token = input_ids[:, position]
            marker = (token == T_MARK) & valid[:, position]
            reading = reading | marker
            digit = (
                reading
                & ~stopped
                & valid[:, position]
                & (token >= DIGIT_OFFSET)
                & (token < DIGIT_OFFSET + NUM_DIGITS)
            )
            value = torch.where(
                digit, value * 10 + token - DIGIT_OFFSET, value
            )
            stopped = stopped | (reading & ~marker & ~digit)
        return value.clamp(min=1, max=MAX_OUTER_STEPS)

    def _register(self, input_ids: Tensor, mask: Tensor) -> Tensor:
        batch = input_ids.shape[0]
        values = (input_ids - DIGIT_OFFSET).clamp(0, NUM_DIGITS - 1)
        source = F.one_hot(values, NUM_DIGITS).float() * mask.unsqueeze(-1)
        count = mask.sum(dim=1)
        rank = mask.long().cumsum(dim=1) - 1
        destination = (
            self.width - count[:, None] + rank
        ).clamp(0, self.width - 1)
        register = torch.zeros(
            batch,
            self.width,
            NUM_DIGITS,
            device=input_ids.device,
            dtype=torch.float32,
        )
        occupied = torch.zeros(
            batch, self.width, 1, device=input_ids.device
        )
        register.scatter_add_(
            1, destination.unsqueeze(-1).expand(-1, -1, NUM_DIGITS), source
        )
        occupied.scatter_add_(
            1, destination.unsqueeze(-1), mask.unsqueeze(-1).float()
        )
        register[:, :, 0] += 1.0 - occupied.squeeze(-1)
        return register.flip(1)

    def _parse(
        self, input_ids: Tensor, valid: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        digits = (
            valid
            & (input_ids >= DIGIT_OFFSET)
            & (input_ids < DIGIT_OFFSET + NUM_DIGITS)
        )
        after_n = (input_ids == N_MARK).cumsum(dim=1) > 0
        after_x = (input_ids == X_MARK).cumsum(dim=1) > 0
        after_t = (input_ids == T_MARK).cumsum(dim=1) > 0
        return (
            self._register(input_ids, digits & after_n & ~after_x),
            self._register(input_ids, digits & after_x & ~after_t),
            self._time_steps(input_ids, valid),
        )

    def _place(
        self, digit_probabilities: Tensor, lengths: Tensor, prompt: int
    ) -> Tensor:
        digit_logits = digit_probabilities.clamp_min(1e-8).log().flip(1)
        destination = lengths[:, None] - self.width + torch.arange(
            self.width, device=digit_logits.device
        )[None]
        destination = destination.clamp(0, prompt - 1)
        canvas = torch.zeros(
            digit_logits.shape[0],
            prompt,
            NUM_DIGITS,
            device=digit_logits.device,
            dtype=digit_logits.dtype,
        )
        canvas = canvas.scatter(
            1,
            destination.unsqueeze(-1).expand(-1, -1, NUM_DIGITS),
            digit_logits,
        )
        return F.pad(
            canvas,
            (DIGIT_OFFSET, self.config.vocab_size - DIGIT_OFFSET - NUM_DIGITS),
            value=-16.0,
        )

    def forward(
        self, input_ids: Tensor, attention_mask: Tensor | None = None
    ) -> tuple[Tensor, dict[str, Tensor]]:
        _, prompt = input_ids.shape
        if attention_mask is None:
            attention_mask = input_ids != PAD
        valid = attention_mask.bool()
        modulus, register, t_values = self._parse(input_ids, valid)
        loops = TRAIN_OUTER_STEPS if self.training else int(t_values.max().item())
        temperature = (
            self.training_temperature if self.training else self.eval_temperature
        )
        first_proposal = register
        first_corrected = register
        for outer_step in range(loops):
            transition_input = (
                register.detach()
                if self.training and outer_step > 0
                else register
            )
            candidate, proposal, corrected = self.square(
                transition_input,
                modulus,
                temperature,
                corrupt_rows=(t_values == 1),
                evaluation_corrections=1 if not self.training else 0,
            )
            if outer_step == 0:
                first_proposal = proposal
                first_corrected = corrected
            register = torch.where(
                (t_values > outer_step)[:, None, None], candidate, register
            )
        lengths = valid.sum(dim=1)
        return self._place(register, lengths, prompt), {
            "t_values": t_values,
            "proposal_logits": self._place(first_proposal, lengths, prompt),
            "corrected_logits": self._place(first_corrected, lengths, prompt),
        }


def _target_aligned(full_logits: Tensor, batch: TokenLossBatch) -> Tensor:
    if batch.target_positions is None:
        return full_logits[:, : batch.logits.shape[1]]
    rows = torch.arange(full_logits.shape[0], device=full_logits.device)[:, None]
    return full_logits[rows, batch.target_positions.clamp_min(0)]


def _masked_ce(logits: Tensor, labels: Tensor, mask: Tensor) -> Tensor:
    losses = F.cross_entropy(
        logits.transpose(1, 2), labels, ignore_index=-100, reduction="none"
    )
    weights = mask.float()
    return (losses * weights).sum() / weights.sum().clamp_min(1.0)


def token_training_loss(batch: TokenLossBatch) -> Tensor:
    valid = batch.valid_mask
    t_values = batch.auxiliary["t_values"]
    row_weight = torch.where(
        t_values == 1,
        torch.full_like(t_values, 4.0, dtype=torch.float32),
        torch.ones_like(t_values, dtype=torch.float32),
    )
    loss = _masked_ce(batch.logits, batch.labels, valid.float() * row_weight[:, None])
    t1 = valid & (t_values == 1)[:, None]
    proposal = _target_aligned(batch.auxiliary["proposal_logits"], batch)
    corrected = _target_aligned(batch.auxiliary["corrected_logits"], batch)
    return loss + 0.20 * _masked_ce(proposal, batch.labels, t1) + 0.35 * _masked_ce(
        corrected, batch.labels, t1
    )


class WallClockSchedule:
    def __init__(
        self, optimizer: torch.optim.Optimizer, model: Model, seconds: float
    ) -> None:
        self.optimizer = optimizer
        self.model = model
        self.started = time.monotonic()
        self.budget = max(float(seconds) * 0.98, 1.0)

    def step(self) -> None:
        fraction = min((time.monotonic() - self.started) / self.budget, 1.0)
        if fraction < 0.05:
            multiplier = 0.1 + 18.0 * fraction
        elif fraction < 0.85:
            multiplier = 1.0
        else:
            progress = (fraction - 0.85) / 0.15
            multiplier = 0.08 + 0.92 * 0.5 * (1.0 + math.cos(math.pi * progress))
        for group in self.optimizer.param_groups:
            group["lr"] = group["base_lr"] * multiplier
        self.model.training_temperature = max(
            0.20, 1.20 * (0.20 / 1.20) ** fraction
        )


def build_model(spec: ModelSpec) -> Model:
    model = Model(spec)
    assert_model_state(model, spec)
    return model


def build_optimizer(model: Model, spec: OptimizerSpec) -> OptimizerBundle:
    decay: list[Tensor] = []
    no_decay: list[Tensor] = []
    for parameter in model.parameters():
        (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": 0.1, "base_lr": BASE_LR},
            {"params": no_decay, "weight_decay": 0.0, "base_lr": BASE_LR},
        ],
        lr=BASE_LR * 0.1,
        betas=(0.9, 0.95),
        eps=1e-8,
        fused=spec.device_type == "cuda",
    )
    return OptimizerBundle(
        optimizer=optimizer,
        scheduler=WallClockSchedule(optimizer, model, spec.training_time_seconds),
    )


SUBMISSION = Submission(
    build_model=build_model,
    build_optimizer=build_optimizer,
    token_training_loss=token_training_loss,
    batch_size=32,
    eval_batch_size=256,
)
