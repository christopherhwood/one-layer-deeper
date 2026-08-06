"""Learned Fourier numeric operator for repeated composition.

Soft digit registers are mapped to generic normalized numeric coordinates and
processed by sinusoidal residual layers.  Shared position queries decode the
operator state back into digit distributions, which are composed across outer
steps.  Learned Fourier features are a general basis for discontinuous and
periodic functions; no squaring, quotient, comparison, or modular-reduction
operation is fixed in the model.
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
HIDDEN = 192
REFINEMENT_STEPS = 4
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


class FourierBlock(nn.Module):
    """One learned sinusoidal residual transformation."""

    def __init__(self) -> None:
        super().__init__()
        self.norm = RMSNorm(HIDDEN)
        self.state = nn.Linear(HIDDEN, HIDDEN)
        self.condition = nn.Linear(4, HIDDEN, bias=False)
        self.gate = nn.Parameter(torch.full((HIDDEN,), -1.5))

    def forward(self, state: Tensor, coordinates: Tensor) -> Tensor:
        phase = self.state(self.norm(state)) + self.condition(coordinates)
        return state + torch.sigmoid(self.gate) * torch.sin(phase)


class NumericOperator(nn.Module):
    """Map two soft decimal values to one soft decimal value."""

    def __init__(self, digits: int) -> None:
        super().__init__()
        self.digits = digits
        self.input = nn.Linear(4, HIDDEN)
        self.blocks = nn.ModuleList(
            FourierBlock() for _ in range(REFINEMENT_STEPS)
        )
        self.position = nn.Linear(2, HIDDEN, bias=False)
        self.readout_norm = RMSNorm(HIDDEN)
        self.readout = nn.Sequential(
            nn.Linear(HIDDEN, 2 * HIDDEN),
            nn.SiLU(),
            nn.Linear(2 * HIDDEN, NUM_DIGITS),
        )

    def _value(self, register: Tensor) -> Tensor:
        digit = torch.arange(
            NUM_DIGITS, device=register.device, dtype=register.dtype
        )
        expected = (register * digit).sum(dim=-1)
        places = torch.pow(
            torch.tensor(10.0, device=register.device, dtype=register.dtype),
            torch.arange(self.digits, device=register.device, dtype=register.dtype),
        ) / (10.0**self.digits)
        return (expected * places).sum(dim=-1)

    def _decode(self, state: Tensor) -> Tensor:
        position = torch.linspace(
            0.0, 1.0, self.digits, device=state.device, dtype=state.dtype
        )
        position_features = self.position(
            torch.stack((position, position.square()), dim=-1)
        )
        hidden = self.readout_norm(state[:, None] + position_features)
        return self.readout(hidden)

    def forward(
        self,
        source: Tensor,
        modulus: Tensor,
        temperature: float,
    ) -> tuple[Tensor, tuple[Tensor, ...]]:
        source_value = self._value(source)
        modulus_value = self._value(modulus).clamp_min(1e-4)
        coordinates = torch.stack(
            (
                source_value,
                modulus_value,
                source_value / modulus_value,
                modulus_value.log(),
            ),
            dim=-1,
        )
        state = torch.sin(2.0 * math.pi * self.input(coordinates))
        phase_logits: list[Tensor] = []
        for block in self.blocks:
            state = block(state, coordinates)
            phase_logits.append(self._decode(state))
        probabilities = F.softmax(phase_logits[-1] / temperature, dim=-1)
        return probabilities, tuple(phase_logits)


class Model(nn.Module):
    num_loops = REFINEMENT_STEPS

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.max_length = spec.max_seq_len
        self.digits = max(2, (spec.max_seq_len - 5) // 2)
        self.transition = NumericOperator(self.digits)
        self.training_temperature = 1.0
        self.eval_temperature = 0.10

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

    def _digit_register(self, input_ids: Tensor, mask: Tensor) -> Tensor:
        batch = input_ids.shape[0]
        values = (input_ids - DIGIT_OFFSET).clamp(0, NUM_DIGITS - 1)
        source = F.one_hot(values, NUM_DIGITS).float() * mask.unsqueeze(-1)
        count = mask.sum(dim=1)
        rank = mask.long().cumsum(dim=1) - 1
        destination = (self.digits - count[:, None] + rank).clamp(
            0, self.digits - 1
        )
        register = torch.zeros(
            batch, self.digits, NUM_DIGITS, device=input_ids.device
        )
        occupied = torch.zeros(batch, self.digits, 1, device=input_ids.device)
        register.scatter_add_(
            1, destination.unsqueeze(-1).expand(-1, -1, NUM_DIGITS), source
        )
        occupied.scatter_add_(
            1, destination.unsqueeze(-1), mask.unsqueeze(-1).float()
        )
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
            3 if self.training else int(t_values.max().item())
        )
        temperature = (
            self.training_temperature if self.training else self.eval_temperature
        )
        terminal_phases = [
            torch.zeros(
                batch,
                self.digits,
                NUM_DIGITS,
                device=input_ids.device,
                dtype=register.dtype,
            )
            for _ in range(REFINEMENT_STEPS)
        ]
        for outer_step in range(outer_steps):
            candidate, phases = self.transition(register, modulus, temperature)
            terminal = (t_values == outer_step + 1)[:, None, None]
            terminal_phases = [
                torch.where(terminal, phase, previous)
                for phase, previous in zip(phases, terminal_phases)
            ]
            active = (t_values > outer_step)[:, None, None]
            register = torch.where(active, candidate, register)

        lengths = valid.sum(dim=1)
        logits = self._place_logits(
            register.clamp_min(1e-8).log(), lengths, prompt
        )
        phase_logits = tuple(
            self._place_logits(phase, lengths, prompt)
            for phase in terminal_phases
        )
        return logits, {"phase_logits": phase_logits}


def _target_aligned(full_logits: Tensor, batch: TokenLossBatch) -> Tensor:
    if batch.target_positions is None:
        return full_logits[:, : batch.logits.shape[1]]
    rows = torch.arange(full_logits.shape[0], device=full_logits.device)[:, None]
    return full_logits[rows, batch.target_positions.clamp_min(0)]


def _masked_cross_entropy(logits: Tensor, batch: TokenLossBatch) -> Tensor:
    losses = F.cross_entropy(
        logits.transpose(1, 2),
        batch.labels,
        ignore_index=-100,
        reduction="none",
    )
    weights = batch.valid_mask.to(losses.dtype)
    return (losses * weights).sum() / weights.sum().clamp_min(1.0)


def token_training_loss(batch: TokenLossBatch) -> Tensor:
    endpoint = _masked_cross_entropy(batch.logits, batch)
    phases = [
        _masked_cross_entropy(_target_aligned(phase, batch), batch)
        for phase in batch.auxiliary["phase_logits"]
    ]
    return endpoint + 0.5 * torch.stack(phases).mean()


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
        elif fraction < 0.88:
            multiplier = 1.0
        else:
            progress = (fraction - 0.88) / 0.12
            multiplier = 0.1 + 0.9 * 0.5 * (
                1.0 + math.cos(math.pi * progress)
            )
        for group in self.optimizer.param_groups:
            group["lr"] = group["base_lr"] * multiplier
        self.model.training_temperature = max(0.20, 1.0 * (0.20**fraction))


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
            {"params": decay, "weight_decay": 0.05, "base_lr": BASE_LR},
            {"params": no_decay, "weight_decay": 0.0, "base_lr": BASE_LR},
        ],
        lr=BASE_LR * 0.1,
        betas=(0.9, 0.95),
        eps=1e-8,
        fused=spec.device_type == "cuda",
    )
    return OptimizerBundle(
        optimizer=optimizer,
        scheduler=WallClockSchedule(
            optimizer, model, spec.training_time_seconds
        ),
    )


SUBMISSION = Submission(
    build_model=build_model,
    build_optimizer=build_optimizer,
    token_training_loss=token_training_loss,
    batch_size=64,
    eval_batch_size=512,
)
