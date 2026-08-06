"""Persistent latent digit-register dynamics for repeated composition.

The input initializes one continuous latent cell per digit exactly once.  A
shared local/global gated update then advances that latent register for each
requested outer application while immutable context remains available.  Digit
logits are readouts only: unlike prior recurrent candidates they are not fed
back as the next state, so scratch information can persist across composition.
No arithmetic process, transition table, or intermediate target is encoded.
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
HIDDEN = 128
MAX_OUTER_STEPS = 64
BASE_LR = 2.0e-3


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


class CellularUpdate(nn.Module):
    """One tied, translation-equivariant gated state update."""

    def __init__(self) -> None:
        super().__init__()
        self.norm = RMSNorm(HIDDEN)
        self.local = nn.Linear(3 * HIDDEN, 3 * HIDDEN, bias=False)
        self.context = nn.Linear(HIDDEN, 3 * HIDDEN, bias=False)
        self.global_mix = nn.Linear(HIDDEN, 3 * HIDDEN, bias=False)
        self.update_bias = nn.Parameter(torch.full((HIDDEN,), -1.5))

    def forward(self, state: Tensor, context: Tensor) -> Tensor:
        normalized = self.norm(state)
        left = F.pad(normalized[:, :-1], (0, 0, 1, 0))
        right = F.pad(normalized[:, 1:], (0, 0, 0, 1))
        local = self.local(torch.cat((left, normalized, right), dim=-1))
        global_state = normalized.mean(dim=1, keepdim=True)
        gates = local + self.context(context) + self.global_mix(global_state)
        reset, update, proposal = gates.chunk(3, dim=-1)
        proposal = torch.tanh(proposal + torch.sigmoid(reset) * normalized)
        rate = torch.sigmoid(update + self.update_bias)
        return state + rate * (proposal - state)


class CellularTransition(nn.Module):
    """Initialize and advance one persistent latent digit register."""

    def __init__(self, digits: int) -> None:
        super().__init__()
        self.digits = digits
        self.source_projection = nn.Linear(NUM_DIGITS, HIDDEN, bias=False)
        self.modulus_projection = nn.Linear(NUM_DIGITS, HIDDEN, bias=False)
        self.boundary_projection = nn.Linear(2, HIDDEN, bias=False)
        self.state_norm = RMSNorm(HIDDEN)
        self.state_mix = nn.Linear(3 * HIDDEN, HIDDEN, bias=False)
        self.context_norm = RMSNorm(HIDDEN)
        self.context_mix = nn.Linear(2 * HIDDEN, HIDDEN, bias=False)
        self.seed = nn.Parameter(torch.empty(1, 1, HIDDEN))
        self.cell = CellularUpdate()
        self.output_norm = RMSNorm(HIDDEN)
        self.output = nn.Linear(HIDDEN, NUM_DIGITS)
        nn.init.normal_(self.seed, std=HIDDEN**-0.5)

    def _boundaries(self, reference: Tensor) -> Tensor:
        boundary = torch.zeros(
            self.digits, 2, device=reference.device, dtype=reference.dtype
        )
        boundary[0, 0] = 1.0
        boundary[-1, 1] = 1.0
        return self.boundary_projection(boundary).unsqueeze(0)

    def initialize(
        self,
        source: Tensor,
        modulus: Tensor,
    ) -> tuple[Tensor, Tensor]:
        source_features = self.source_projection(source)
        modulus_features = self.modulus_projection(modulus)
        boundary_features = self._boundaries(source).expand(
            source.shape[0], -1, -1
        )
        context = self.context_norm(
            self.context_mix(
                torch.cat((modulus_features, boundary_features), dim=-1)
            )
        )
        state = self.state_norm(
            self.state_mix(
                torch.cat(
                    (source_features, modulus_features, boundary_features),
                    dim=-1,
                )
            )
        ) + self.seed
        return state, context

    def step(self, state: Tensor, context: Tensor) -> tuple[Tensor, Tensor]:
        for _ in range(self.digits):
            state = self.cell(state, context)
        return state, self.output(self.output_norm(state))


class Model(nn.Module):
    num_loops = 1

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.max_length = spec.max_seq_len
        self.digits = max(2, (spec.max_seq_len - 5) // 2)
        self.transition = CellularTransition(self.digits)
        self.training_temperature = 1.0

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
        modulus, source, t_values = self._parse(input_ids, valid)
        state, context = self.transition.initialize(source, modulus)
        outer_steps = 3 if self.training else int(t_values.max().item())
        terminal_logits = torch.zeros(
            batch,
            self.digits,
            NUM_DIGITS,
            device=input_ids.device,
            dtype=state.dtype,
        )
        for outer_step in range(outer_steps):
            candidate, digit_logits = self.transition.step(state, context)
            terminal = (t_values == outer_step + 1)[:, None, None]
            terminal_logits = torch.where(
                terminal, digit_logits, terminal_logits
            )
            active = (t_values > outer_step)[:, None, None]
            state = torch.where(active, candidate, state)

        lengths = valid.sum(dim=1)
        logits = self._place_logits(terminal_logits, lengths, prompt)
        return logits, {}


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
    return _masked_cross_entropy(batch.logits, batch)


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
