"""Length-scaled gated convolutional reasoner for hidden recurrence.

The learned transition is a small Neural-GPU-style cellular state machine. A
digit register and all immutable prompt context are projected into one cell per digit;
the same local gated update is then applied a number of times proportional to
the register length.  Nothing in the update encodes multiplication, carries,
comparison, or modular reduction.  The transition is reused unchanged for
each requested outer application and receives endpoint supervision after each
recurrent sweep.
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
SWEEPS = 2
TRAIN_OUTER_STEPS = 8
CRF_STEPS = 2
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
        self.local = nn.Conv1d(
            HIDDEN, 3 * HIDDEN, kernel_size=3, padding=1, bias=False
        )
        self.context = nn.Linear(HIDDEN, 3 * HIDDEN, bias=False)
        self.global_mix = nn.Linear(HIDDEN, 3 * HIDDEN, bias=False)
        self.update_bias = nn.Parameter(torch.full((HIDDEN,), -1.5))

    def forward(self, state: Tensor, context: Tensor) -> Tensor:
        normalized = self.norm(state)
        local = self.local(normalized.transpose(1, 2)).transpose(1, 2)
        global_state = normalized.mean(dim=1, keepdim=True)
        gates = local + self.context(context) + self.global_mix(global_state)
        reset, update, proposal = gates.chunk(3, dim=-1)
        proposal = torch.tanh(proposal + torch.sigmoid(reset) * normalized)
        rate = torch.sigmoid(update + self.update_bias)
        return state + rate * (proposal - state)


class CellularTransition(nn.Module):
    """Apply a general recurrent cellular program to one digit register."""

    def __init__(self, digits: int) -> None:
        super().__init__()
        self.digits = digits
        self.source_projection = nn.Linear(NUM_DIGITS, HIDDEN, bias=False)
        self.modulus_projection = nn.Linear(NUM_DIGITS, HIDDEN, bias=False)
        self.boundary_projection = nn.Linear(2, HIDDEN, bias=False)
        self.input_norm = RMSNorm(HIDDEN)
        self.original_projection = nn.Linear(NUM_DIGITS, HIDDEN, bias=False)
        self.input_mix = nn.Linear(4 * HIDDEN, HIDDEN, bias=False)
        self.seed = nn.Parameter(torch.empty(1, 1, HIDDEN))
        self.cell = CellularUpdate()
        self.output_norm = RMSNorm(HIDDEN)
        self.output = nn.Linear(HIDDEN, NUM_DIGITS)
        self.feedback = nn.Linear(NUM_DIGITS, HIDDEN, bias=False)
        self.feedback_gate = nn.Parameter(torch.full((HIDDEN,), -1.5))
        self.crf_transition = nn.Parameter(torch.empty(NUM_DIGITS, NUM_DIGITS))
        self.crf_scale = nn.Parameter(torch.tensor(-1.0))
        nn.init.normal_(self.seed, std=HIDDEN**-0.5)
        nn.init.normal_(self.crf_transition, std=0.02)

    def _refine_digits(self, unary: Tensor) -> Tensor:
        logits = unary
        scale = torch.sigmoid(self.crf_scale)
        for _ in range(CRF_STEPS):
            probability = F.softmax(logits, dim=-1)
            left_probability = F.pad(probability[:, :-1], (0, 0, 1, 0))
            right_probability = F.pad(probability[:, 1:], (0, 0, 0, 1))
            left_message = left_probability @ self.crf_transition
            right_message = right_probability @ self.crf_transition.transpose(0, 1)
            logits = unary + scale * (left_message + right_message)
        return logits

    def _boundaries(self, reference: Tensor) -> Tensor:
        boundary = torch.zeros(
            self.digits, 2, device=reference.device, dtype=reference.dtype
        )
        boundary[0, 0] = 1.0
        boundary[-1, 1] = 1.0
        return self.boundary_projection(boundary).unsqueeze(0)

    def forward(
        self,
        source: Tensor,
        modulus: Tensor,
        original: Tensor,
        temperature: float,
    ) -> tuple[Tensor, tuple[Tensor, ...]]:
        source_features = self.source_projection(source)
        modulus_features = self.modulus_projection(modulus)
        original_features = self.original_projection(original)
        boundary_features = self._boundaries(source)
        context = self.input_norm(
            self.input_mix(
                torch.cat(
                    (
                        source_features,
                        modulus_features,
                        original_features,
                        boundary_features.expand(source.shape[0], -1, -1),
                    ),
                    dim=-1,
                )
            )
        )
        state = context + self.seed
        phase_logits: list[Tensor] = []
        for _ in range(SWEEPS):
            for _ in range(self.digits):
                state = self.cell(state, context)
            logits = self._refine_digits(self.output(self.output_norm(state)))
            phase_logits.append(logits)
            probabilities = F.softmax(logits / temperature, dim=-1)
            state = state + torch.sigmoid(self.feedback_gate) * self.feedback(
                probabilities
            )
        return probabilities, tuple(phase_logits)


class Model(nn.Module):
    num_loops = SWEEPS

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.max_length = spec.max_seq_len
        self.digits = max(2, (spec.max_seq_len - 5) // 2)
        self.transition = CellularTransition(self.digits)
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
        original = register
        outer_steps = (
            min(TRAIN_OUTER_STEPS, int(t_values.max().item()))
            if self.training
            else int(t_values.max().item())
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
            for _ in range(SWEEPS)
        ]
        for outer_step in range(outer_steps):
            source = (
                register.detach()
                if self.training and outer_step > 0
                else register
            )
            candidate, phases = self.transition(
                source, modulus, original, temperature
            )
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
