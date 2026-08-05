"""Tiny shared digit transducer for repeated modular squaring.

The model has no prompt transformer, learned position table, or free global
latent.  N and x are parsed into right-aligned decimal registers on device.  A
single learned bidirectional scan transition is shared across digit positions,
refinement rounds, and outer T steps.  Scan order is the only positional signal.
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
WIDTH = 64
SCAN_WIDTH = WIDTH // 2
REFINEMENT_STEPS = 6
MAX_OUTER_STEPS = 64
BASE_LR = 2e-3


class Config:
    def __init__(self, vocab_size: int, max_seq_len: int) -> None:
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len


class SharedDigitTransition(nn.Module):
    """One length-independent scan and gated digit-register replacement."""

    def __init__(self) -> None:
        super().__init__()
        self.digit_embedding = nn.Parameter(torch.empty(NUM_DIGITS, WIDTH))
        self.modulus_embedding = nn.Parameter(torch.empty(NUM_DIGITS, WIDTH))
        nn.init.normal_(self.digit_embedding, std=WIDTH**-0.5)
        nn.init.normal_(self.modulus_embedding, std=WIDTH**-0.5)
        self.initialize = nn.Linear(2 * WIDTH, WIDTH, bias=False)
        self.scan = nn.GRU(
            input_size=3 * WIDTH,
            hidden_size=SCAN_WIDTH,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.proposal = nn.Sequential(
            nn.Linear(WIDTH, 2 * WIDTH),
            nn.SiLU(),
            nn.Linear(2 * WIDTH, WIDTH),
        )
        self.hidden_gate = nn.Linear(WIDTH, WIDTH)
        self.digit_head = nn.Linear(WIDTH, NUM_DIGITS)
        self.replace_gate = nn.Linear(WIDTH, 1)
        nn.init.constant_(self.hidden_gate.bias, -1.0)
        nn.init.constant_(self.replace_gate.bias, -0.75)

    def _embed(
        self, register: Tensor, modulus: Tensor
    ) -> tuple[Tensor, Tensor]:
        dtype = self.digit_embedding.dtype
        return (
            register.to(dtype) @ self.digit_embedding,
            modulus.to(dtype) @ self.modulus_embedding,
        )

    def forward(
        self,
        register: Tensor,
        modulus: Tensor,
        temperature: float,
    ) -> Tensor:
        residue_embedding, modulus_embedding = self._embed(register, modulus)
        hidden = self.initialize(
            torch.cat((residue_embedding, modulus_embedding), dim=-1)
        )
        for _ in range(REFINEMENT_STEPS):
            residue_embedding, modulus_embedding = self._embed(register, modulus)
            scan_input = torch.cat(
                (hidden, residue_embedding, modulus_embedding), dim=-1
            )
            scanned, _ = self.scan(scan_input)
            update = self.proposal(scanned)
            hidden = hidden + torch.sigmoid(self.hidden_gate(hidden)) * update
            digit_logits = self.digit_head(hidden).float()
            candidate = F.softmax(digit_logits / temperature, dim=-1)
            gate = torch.sigmoid(self.replace_gate(hidden).float())
            register = (1.0 - gate) * register + gate * candidate
            register = register / register.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return register


class Model(nn.Module):
    num_loops = REFINEMENT_STEPS

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.cells = spec.max_seq_len
        self.transition = SharedDigitTransition()
        self.training_outer_steps = 4
        self.training_temperature = 1.25
        self.eval_temperature = 0.16

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
            value = torch.where(digit, value * 10 + token - DIGIT_OFFSET, value)
            stopped = stopped | (reading & ~marker & ~digit)
        return value.clamp(min=1, max=MAX_OUTER_STEPS)

    def _right_aligned_register(self, input_ids: Tensor, field_mask: Tensor) -> Tensor:
        batch, _ = input_ids.shape
        digits = (input_ids - DIGIT_OFFSET).clamp(min=0, max=NUM_DIGITS - 1)
        source = F.one_hot(digits, NUM_DIGITS).float() * field_mask.unsqueeze(-1)
        count = field_mask.sum(dim=1)
        rank = field_mask.long().cumsum(dim=1) - 1
        destination = (self.cells - count[:, None] + rank).clamp(
            min=0, max=self.cells - 1
        )
        register = torch.zeros(
            batch,
            self.cells,
            NUM_DIGITS,
            device=input_ids.device,
            dtype=torch.float32,
        )
        occupied = torch.zeros(
            batch, self.cells, 1, device=input_ids.device, dtype=torch.float32
        )
        register.scatter_add_(
            1, destination.unsqueeze(-1).expand(-1, -1, NUM_DIGITS), source
        )
        occupied.scatter_add_(
            1, destination.unsqueeze(-1), field_mask.unsqueeze(-1).float()
        )
        register[:, :, 0] = register[:, :, 0] + (1.0 - occupied.squeeze(-1))
        return register

    def _parse(self, input_ids: Tensor, valid: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        digits = (
            valid
            & (input_ids >= DIGIT_OFFSET)
            & (input_ids < DIGIT_OFFSET + NUM_DIGITS)
        )
        after_n = (input_ids == N_MARK).cumsum(dim=1) > 0
        after_x = (input_ids == X_MARK).cumsum(dim=1) > 0
        after_t = (input_ids == T_MARK).cumsum(dim=1) > 0
        n_mask = digits & after_n & ~after_x
        x_mask = digits & after_x & ~after_t
        return (
            self._right_aligned_register(input_ids, n_mask),
            self._right_aligned_register(input_ids, x_mask),
            self._time_steps(input_ids, valid),
        )

    def _vocab_logits(self, register: Tensor, prompt_length: int) -> Tensor:
        digit_logits = register.clamp_min(1e-8).log()
        logits = F.pad(
            digit_logits,
            (DIGIT_OFFSET, self.config.vocab_size - DIGIT_OFFSET - NUM_DIGITS),
            value=-16.0,
        )
        return logits[:, self.cells - prompt_length :]

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        _, prompt_length = input_ids.shape
        if prompt_length > self.cells:
            raise ValueError("input is longer than model.config.max_seq_len")
        valid = (
            attention_mask.bool()
            if attention_mask is not None
            else input_ids != PAD
        )
        modulus, register, t_values = self._parse(input_ids, valid)
        outer_steps = self.training_outer_steps if self.training else MAX_OUTER_STEPS
        temperature = self.training_temperature if self.training else self.eval_temperature
        for outer_step in range(outer_steps):
            candidate = self.transition(register, modulus, temperature)
            active = (t_values > outer_step)[:, None, None]
            register = torch.where(active, candidate, register)
        return self._vocab_logits(register, prompt_length), {"t_values": t_values}


def token_training_loss(batch: TokenLossBatch) -> Tensor:
    losses = F.cross_entropy(
        batch.logits.transpose(1, 2),
        batch.labels,
        ignore_index=-100,
        reduction="none",
    )
    valid = batch.valid_mask
    t_values = batch.auxiliary["t_values"]
    row_weight = torch.where(
        t_values == 1,
        torch.full_like(t_values, 3.0, dtype=torch.float32),
        torch.ones_like(t_values, dtype=torch.float32),
    )
    token_weight = valid.float() * row_weight[:, None]
    token_term = (losses * token_weight).sum() / token_weight.sum().clamp_min(1.0)
    counts = valid.sum(dim=1).clamp_min(1)
    sequence_losses = (losses * valid).sum(dim=1) / counts.float().mean()
    sequence_term = (
        sequence_losses * row_weight
    ).sum() / row_weight.sum().clamp_min(1.0)
    return 0.65 * token_term + 0.35 * sequence_term


class WallClockSchedule:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        model: Model,
        budget_seconds: float,
    ) -> None:
        self.optimizer = optimizer
        self.model = model
        self.started_at = time.monotonic()
        self.budget_seconds = max(float(budget_seconds) * 0.98, 1.0)

    def step(self) -> None:
        fraction = min(
            (time.monotonic() - self.started_at) / self.budget_seconds, 1.0
        )
        if fraction < 0.05:
            multiplier = 0.1 + 0.9 * fraction / 0.05
        elif fraction < 0.82:
            multiplier = 1.0
        else:
            progress = (fraction - 0.82) / 0.18
            multiplier = 0.08 + 0.92 * 0.5 * (
                1.0 + math.cos(math.pi * progress)
            )
        for group in self.optimizer.param_groups:
            group["lr"] = group["base_lr"] * multiplier
        self.model.training_temperature = max(
            0.18, 1.25 * (0.18 / 1.25) ** fraction
        )


def build_model(spec: ModelSpec) -> Model:
    model = Model(spec)
    assert_model_state(model, spec)
    return model


def build_optimizer(model: Model, spec: OptimizerSpec) -> OptimizerBundle:
    model.training_outer_steps = 4 if spec.training_time_seconds <= 90 else 16
    decay: list[Tensor] = []
    no_decay: list[Tensor] = []
    for parameter in model.parameters():
        (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {
                "params": decay,
                "weight_decay": 0.02,
                "base_lr": BASE_LR,
                "lr": BASE_LR * 0.1,
            },
            {
                "params": no_decay,
                "weight_decay": 0.0,
                "base_lr": BASE_LR,
                "lr": BASE_LR * 0.1,
            },
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
    batch_size=64,
    eval_batch_size=512,
)
