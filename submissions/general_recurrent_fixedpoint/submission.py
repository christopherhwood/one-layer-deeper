"""Endpoint-anchored fixed-point learning for a generic recurrent reasoner.

The transition is a small weight-tied attention block over immutable input
context plus answer and scratch slots.  It receives no arithmetic features or
process labels.  Every refinement phase predicts the observed endpoint, while
late phases are also trained to agree in hidden representation and output
distribution.  Those generic fixed-point losses use states from this single
forward pass and stop-gradient targets in both directions.
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
WIDTH = 96
HEADS = 4
SEGMENTS = 6
MICRO_STEPS = 1
CONSISTENCY_START = 2
MAX_OUTER_STEPS = 64
BASE_LR = 1.0e-3
DEEP_WEIGHT = 0.45
REPRESENTATION_WEIGHT = 0.10
READOUT_WEIGHT = 0.10


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
    """One general token-mixing block, tied across all reasoning segments."""

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


class FixedPointTransition(nn.Module):
    """Refine generic answer/scratch slots and expose each phase once."""

    def __init__(self, digits: int) -> None:
        super().__init__()
        self.digits = digits
        self.digit_projection = nn.Linear(NUM_DIGITS, WIDTH, bias=False)
        self.position_projection = nn.Linear(2, WIDTH, bias=False)
        self.role_embedding = nn.Parameter(torch.empty(4, WIDTH))
        self.answer_seed = nn.Parameter(torch.empty(digits, WIDTH))
        self.scratch_seed = nn.Parameter(torch.empty(digits, WIDTH))
        self.block = ReasoningBlock()
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
    ) -> tuple[Tensor, tuple[Tensor, ...], tuple[Tensor, ...]]:
        batch = source.shape[0]
        position = self._positions(source.device, source.dtype)
        modulus_context = (
            self.digit_projection(modulus) + position + self.role_embedding[0]
        )
        source_context = (
            self.digit_projection(source) + position + self.role_embedding[1]
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
        phase_states: list[Tensor] = []
        for segment in range(SEGMENTS):
            for _ in range(MICRO_STEPS):
                state = self.block(torch.cat((context, answer, scratch), dim=1))
                answer = state[:, 2 * self.digits : 3 * self.digits]
                scratch = state[:, 3 * self.digits :]
            logits = self.output(self.output_norm(answer))
            phase_logits.append(logits)
            phase_states.append(torch.cat((answer, scratch), dim=1))
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
        return probabilities, tuple(phase_logits), tuple(phase_states)


class Model(nn.Module):
    num_loops = SEGMENTS

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.max_length = spec.max_seq_len
        self.digits = max(2, (spec.max_seq_len - 5) // 2)
        self.transition = FixedPointTransition(self.digits)
        self.training_outer_steps = 3
        self.training_temperature = 1.0
        self.eval_temperature = 0.10
        self.detach_segments = True

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
        register = torch.zeros(batch, self.digits, NUM_DIGITS, device=input_ids.device)
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
        outer_steps = self.training_outer_steps if self.training else MAX_OUTER_STEPS
        temperature = self.training_temperature if self.training else self.eval_temperature
        terminal_phases = [
            torch.zeros(
                batch, self.digits, NUM_DIGITS,
                device=input_ids.device, dtype=register.dtype,
            )
            for _ in range(SEGMENTS)
        ]
        terminal_states = [
            torch.zeros(
                batch, 2 * self.digits, WIDTH,
                device=input_ids.device, dtype=register.dtype,
            )
            for _ in range(SEGMENTS)
        ]
        for outer_step in range(outer_steps):
            source = register.detach() if self.training and outer_step > 0 else register
            candidate, phases, states = self.transition(
                source, modulus, temperature, self.training and self.detach_segments
            )
            terminal = (t_values == outer_step + 1)[:, None, None]
            terminal_phases = [
                torch.where(terminal, phase, previous)
                for phase, previous in zip(phases, terminal_phases)
            ]
            terminal_states = [
                torch.where(terminal, state, previous)
                for state, previous in zip(states, terminal_states)
            ]
            active = (t_values > outer_step)[:, None, None]
            register = torch.where(active, candidate, register)

        lengths = valid.sum(dim=1)
        logits = self._place_logits(register.clamp_min(1e-8).log(), lengths, prompt)
        phase_logits = tuple(
            self._place_logits(phase, lengths, prompt) for phase in terminal_phases
        )
        return logits, {
            "t_values": t_values,
            "phase_logits": phase_logits,
            "phase_states": tuple(terminal_states),
            "supervised_rows": t_values <= outer_steps,
        }


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


def _representation_agreement(left: Tensor, right: Tensor, rows: Tensor) -> Tensor:
    left = F.normalize(left.float(), dim=-1)
    right = F.normalize(right.float(), dim=-1)
    forward = 1.0 - (left * right.detach()).sum(dim=-1)
    backward = 1.0 - (left.detach() * right).sum(dim=-1)
    weights = rows[:, None].to(forward.dtype)
    return (0.5 * (forward + backward) * weights).sum() / (
        weights.sum().clamp_min(1.0) * forward.shape[1]
    )


def _readout_agreement(
    left: Tensor, right: Tensor, valid: Tensor
) -> Tensor:
    left_log = F.log_softmax(left.float(), dim=-1)
    right_log = F.log_softmax(right.float(), dim=-1)
    left_probability = left_log.exp()
    right_probability = right_log.exp()
    forward = F.kl_div(
        left_log, right_probability.detach(), reduction="none"
    ).sum(dim=-1)
    backward = F.kl_div(
        right_log, left_probability.detach(), reduction="none"
    ).sum(dim=-1)
    weights = valid.to(forward.dtype)
    return (0.5 * (forward + backward) * weights).sum() / weights.sum().clamp_min(1.0)


def token_training_loss(batch: TokenLossBatch) -> Tensor:
    endpoint = _masked_cross_entropy(batch.logits, batch.labels, batch.valid_mask)
    auxiliary = batch.auxiliary
    supervised_rows = auxiliary["supervised_rows"]
    supervised_valid = batch.valid_mask & supervised_rows[:, None]
    phase_logits = auxiliary["phase_logits"]
    aligned_phases = tuple(_target_aligned(phase, batch) for phase in phase_logits)
    deep_terms = [
        _masked_cross_entropy(phase, batch.labels, supervised_valid)
        for phase in aligned_phases
    ]
    representation_terms = []
    readout_terms = []
    phase_states = auxiliary["phase_states"]
    for phase in range(CONSISTENCY_START, SEGMENTS - 1):
        representation_terms.append(
            _representation_agreement(
                phase_states[phase], phase_states[phase + 1], supervised_rows
            )
        )
        readout_terms.append(
            _readout_agreement(
                aligned_phases[phase], aligned_phases[phase + 1], supervised_valid
            )
        )
    fixed_point = torch.stack(representation_terms).mean()
    readout = torch.stack(readout_terms).mean()
    return (
        endpoint
        + DEEP_WEIGHT * torch.stack(deep_terms).mean()
        + REPRESENTATION_WEIGHT * fixed_point
        + READOUT_WEIGHT * readout
    )


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
        self.model.training_temperature = max(0.20, 1.0 * (0.20 ** fraction))


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
