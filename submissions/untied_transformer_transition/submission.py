"""Untied Transformer transition with full compositional credit assignment.

The learned transition sees a current digit register and immutable modulus
context, but no arithmetic-specific features or process labels.  Four distinct
Transformer blocks refine answer and scratch slots, allowing depth to learn a
sequence of operations instead of forcing one block to be a fixed point.  The
complete transition is reused exactly T times, with full gradient flow through
the three supervised training applications.  T remains only a compute control
signal and is never exposed to the learned transition.
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
WIDTH = 128
HEADS = 4
SEGMENTS = 4
MICRO_STEPS = 1
MAX_OUTER_STEPS = 64
BASE_LR = 1.0e-3


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


class ReasoningBlock(nn.Module):
    """One general pre-normalized token-mixing block."""

    def __init__(self) -> None:
        super().__init__()
        self.head_width = WIDTH // HEADS
        self.attention_norm = RMSNorm(WIDTH)
        self.qkv = nn.Linear(WIDTH, 3 * WIDTH, bias=False)
        self.attention_out = nn.Linear(WIDTH, WIDTH, bias=False)
        self.mlp_norm = RMSNorm(WIDTH)
        self.mlp_up = nn.Linear(WIDTH, 4 * WIDTH, bias=False)
        self.mlp_down = nn.Linear(2 * WIDTH, WIDTH, bias=False)
        self.attention_gate = nn.Parameter(torch.full((WIDTH,), -2.0))
        self.mlp_gate = nn.Parameter(torch.full((WIDTH,), -2.0))

    def forward(self, state: Tensor) -> Tensor:
        batch, length, _ = state.shape
        normalized = self.attention_norm(state)
        query, key, value = self.qkv(normalized).chunk(3, dim=-1)
        query = query.view(batch, length, HEADS, self.head_width).transpose(1, 2)
        key = key.view(batch, length, HEADS, self.head_width).transpose(1, 2)
        value = value.view(batch, length, HEADS, self.head_width).transpose(1, 2)
        attended = F.scaled_dot_product_attention(query, key, value)
        attended = attended.transpose(1, 2).contiguous().view(batch, length, WIDTH)
        state = state + torch.sigmoid(self.attention_gate) * self.attention_out(attended)
        left, right = self.mlp_up(self.mlp_norm(state)).chunk(2, dim=-1)
        mixed = F.silu(left) * right
        return state + torch.sigmoid(self.mlp_gate) * self.mlp_down(mixed)


class DeepRecurrentTransition(nn.Module):
    """Refine a discrete output interface using generic answer/scratch slots."""

    def __init__(self, digits: int) -> None:
        super().__init__()
        self.digits = digits
        self.digit_projection = nn.Linear(NUM_DIGITS, WIDTH, bias=False)
        self.position_projection = nn.Linear(2, WIDTH, bias=False)
        self.role_embedding = nn.Parameter(torch.empty(4, WIDTH))
        self.answer_seed = nn.Parameter(torch.empty(digits, WIDTH))
        self.scratch_seed = nn.Parameter(torch.empty(digits, WIDTH))
        self.blocks = nn.ModuleList(ReasoningBlock() for _ in range(SEGMENTS))
        self.output_norm = RMSNorm(WIDTH)
        self.output = nn.Linear(WIDTH, NUM_DIGITS)
        self.snap_gate = nn.Parameter(torch.full((WIDTH,), -1.0))
        nn.init.normal_(self.role_embedding, std=WIDTH**-0.5)
        nn.init.normal_(self.answer_seed, std=WIDTH**-0.5)
        nn.init.normal_(self.scratch_seed, std=WIDTH**-0.5)

    def _positions(self, device: torch.device, dtype: torch.dtype) -> Tensor:
        position = torch.linspace(0.0, 1.0, self.digits, device=device, dtype=dtype)
        return self.position_projection(torch.stack((position, position.square()), -1))

    def forward(
        self,
        source: Tensor,
        modulus: Tensor,
        temperature: float,
        detach_segments: bool,
    ) -> tuple[Tensor, tuple[Tensor, ...]]:
        batch = source.shape[0]
        position = self._positions(source.device, source.dtype)
        modulus_context = (
            self.digit_projection(modulus)
            + position
            + self.role_embedding[0]
        )
        source_context = (
            self.digit_projection(source)
            + position
            + self.role_embedding[1]
        )
        context = torch.cat((modulus_context, source_context), dim=1)
        answer = (
            self.answer_seed
            + self.digit_projection(source)
            + position
            + self.role_embedding[2]
        )
        scratch = (
            self.scratch_seed + position + self.role_embedding[3]
        ).unsqueeze(0).expand(batch, -1, -1)

        phase_logits: list[Tensor] = []
        for segment, block in enumerate(self.blocks):
            for _ in range(MICRO_STEPS):
                state = block(torch.cat((context, answer, scratch), dim=1))
                answer = state[:, 2 * self.digits : 3 * self.digits]
                scratch = state[:, 3 * self.digits :]
            logits = self.output(self.output_norm(answer))
            phase_logits.append(logits)
            probabilities = F.softmax(logits / temperature, dim=-1)
            snapped = (
                self.digit_projection(probabilities)
                + position
                + self.role_embedding[2]
            )
            gate = torch.sigmoid(self.snap_gate)
            answer = answer + gate * (snapped - answer)
            if detach_segments and segment + 1 < SEGMENTS:
                answer = answer.detach()
                scratch = scratch.detach()
        return probabilities, tuple(phase_logits)


class Model(nn.Module):
    num_loops = SEGMENTS

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.max_length = spec.max_seq_len
        self.digits = max(2, (spec.max_seq_len - 5) // 2)
        self.transition = DeepRecurrentTransition(self.digits)
        self.training_outer_steps = 3
        self.training_temperature = 1.0
        self.eval_temperature = 0.10
        self.detach_segments = False

    @staticmethod
    def _time_steps(input_ids: Tensor, valid: Tensor) -> Tensor:
        value = torch.zeros(input_ids.shape[0], device=input_ids.device, dtype=torch.long)
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

    def _digit_register(self, input_ids: Tensor, mask: Tensor) -> Tensor:
        batch = input_ids.shape[0]
        values = (input_ids - DIGIT_OFFSET).clamp(0, NUM_DIGITS - 1)
        source = F.one_hot(values, NUM_DIGITS).float() * mask.unsqueeze(-1)
        count = mask.sum(dim=1)
        rank = mask.long().cumsum(dim=1) - 1
        destination = (self.digits - count[:, None] + rank).clamp(0, self.digits - 1)
        register = torch.zeros(
            batch, self.digits, NUM_DIGITS, device=input_ids.device
        )
        occupied = torch.zeros(batch, self.digits, 1, device=input_ids.device)
        register.scatter_add_(
            1, destination.unsqueeze(-1).expand(-1, -1, NUM_DIGITS), source
        )
        occupied.scatter_add_(1, destination.unsqueeze(-1), mask.unsqueeze(-1).float())
        register[:, :, 0] += 1.0 - occupied.squeeze(-1)
        return register.flip(1)

    def _parse(self, input_ids: Tensor, valid: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        digits = (
            valid
            & (input_ids >= DIGIT_OFFSET)
            & (input_ids < DIGIT_OFFSET + NUM_DIGITS)
        )
        after_n = (input_ids == N_MARK).cumsum(dim=1) > 0
        after_x = (input_ids == X_MARK).cumsum(dim=1) > 0
        after_t = (input_ids == T_MARK).cumsum(dim=1) > 0
        return (
            self._digit_register(input_ids, digits & after_n & ~after_x),
            self._digit_register(input_ids, digits & after_x & ~after_t),
            self._time_steps(input_ids, valid),
        )

    def _place_logits(
        self, digit_logits_lsd: Tensor, input_lengths: Tensor, prompt: int
    ) -> Tensor:
        digit_logits = digit_logits_lsd.flip(1)
        destination = input_lengths[:, None] - self.digits + torch.arange(
            self.digits, device=digit_logits.device
        )[None, :]
        destination = destination.clamp(0, prompt - 1)
        canvas = torch.zeros(
            digit_logits.shape[0], prompt, NUM_DIGITS,
            device=digit_logits.device, dtype=digit_logits.dtype,
        )
        canvas = canvas.scatter(
            1, destination.unsqueeze(-1).expand(-1, -1, NUM_DIGITS), digit_logits
        )
        return F.pad(
            canvas,
            (DIGIT_OFFSET, self.config.vocab_size - DIGIT_OFFSET - NUM_DIGITS),
            value=-16.0,
        )

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, object]]:
        batch, prompt = input_ids.shape
        if attention_mask is None:
            attention_mask = input_ids != PAD
        valid = attention_mask.bool()
        modulus, register, t_values = self._parse(input_ids, valid)
        outer_steps = (
            self.training_outer_steps
            if self.training
            else int(t_values.max().item())
        )
        temperature = self.training_temperature if self.training else self.eval_temperature
        terminal_phases: list[Tensor] = [
            torch.zeros(
                batch, self.digits, NUM_DIGITS,
                device=input_ids.device, dtype=register.dtype,
            )
            for _ in range(SEGMENTS)
        ]
        for outer_step in range(outer_steps):
            source = register
            candidate, phases = self.transition(
                source, modulus, temperature, self.training and self.detach_segments
            )
            terminal = (t_values == outer_step + 1)[:, None, None]
            terminal_phases = [
                torch.where(terminal, phase, previous)
                for phase, previous in zip(phases, terminal_phases)
            ]
            active = (t_values > outer_step)[:, None, None]
            register = torch.where(active, candidate, register)

        lengths = valid.sum(dim=1)
        logits = self._place_logits(register.clamp_min(1e-8).log(), lengths, prompt)
        phase_logits = tuple(
            self._place_logits(phase, lengths, prompt) for phase in terminal_phases
        )
        return logits, {"t_values": t_values, "phase_logits": phase_logits}


def _target_aligned(full_logits: Tensor, batch: TokenLossBatch) -> Tensor:
    if batch.target_positions is None:
        return full_logits[:, : batch.logits.shape[1]]
    rows = torch.arange(full_logits.shape[0], device=full_logits.device)[:, None]
    return full_logits[rows, batch.target_positions.clamp_min(0)]


def _masked_cross_entropy(logits: Tensor, labels: Tensor, valid: Tensor) -> Tensor:
    losses = F.cross_entropy(
        logits.transpose(1, 2), labels, ignore_index=-100, reduction="none"
    )
    weights = valid.to(losses.dtype)
    return (losses * weights).sum() / weights.sum().clamp_min(1.0)


def token_training_loss(batch: TokenLossBatch) -> Tensor:
    endpoint = _masked_cross_entropy(batch.logits, batch.labels, batch.valid_mask)
    phases = batch.auxiliary["phase_logits"]
    deep_terms = [
        _masked_cross_entropy(
            _target_aligned(phase, batch), batch.labels, batch.valid_mask
        )
        for phase in phases
    ]
    return endpoint + 0.5 * torch.stack(deep_terms).mean()


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
        fraction = min((time.monotonic() - self.started_at) / self.budget_seconds, 1.0)
        if fraction < 0.05:
            multiplier = 0.1 + 0.9 * fraction / 0.05
        elif fraction < 0.88:
            multiplier = 1.0
        else:
            progress = (fraction - 0.88) / 0.12
            multiplier = 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))
        for group in self.optimizer.param_groups:
            group["lr"] = group["base_lr"] * multiplier
        self.model.training_temperature = max(
            0.20, 1.0 * (0.20 ** fraction)
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
    batch_size=64,
    eval_batch_size=512,
)
