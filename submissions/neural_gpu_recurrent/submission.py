"""Length-scaled gated convolutional reasoner for repeated composition.

The learned transition is a small Neural-GPU-style cellular state machine.  A
digit register and immutable context are projected into one cell per digit;
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
MAX_OUTER_STEPS = 64
BASE_LR = 1.0e-3
PHASE_WEIGHT = 0.5
BIT_WEIGHT = 1.0
ORDINAL_BINS = 32
ORDINAL_WEIGHT = 0.5


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
    """Apply a general recurrent cellular program to one digit register."""

    def __init__(self, digits: int) -> None:
        super().__init__()
        self.digits = digits
        self.bit_width = (3322 * digits + 999) // 1000 + 1
        self.source_projection = nn.Linear(NUM_DIGITS, HIDDEN, bias=False)
        self.modulus_projection = nn.Linear(NUM_DIGITS, HIDDEN, bias=False)
        self.boundary_projection = nn.Linear(2, HIDDEN, bias=False)
        self.input_norm = RMSNorm(HIDDEN)
        self.input_mix = nn.Linear(3 * HIDDEN, HIDDEN, bias=False)
        self.seed = nn.Parameter(torch.empty(1, 1, HIDDEN))
        self.cell = CellularUpdate()
        self.output_norm = RMSNorm(HIDDEN)
        self.output = nn.Linear(HIDDEN, NUM_DIGITS)
        self.bit_output = nn.Linear(HIDDEN, 2 * self.bit_width)
        self.ordinal_output = nn.Linear(HIDDEN, ORDINAL_BINS)
        self.feedback = nn.Linear(NUM_DIGITS, HIDDEN, bias=False)
        self.feedback_gate = nn.Parameter(torch.full((HIDDEN,), -1.5))
        nn.init.normal_(self.seed, std=HIDDEN**-0.5)

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
        temperature: float,
    ) -> tuple[
        Tensor, tuple[Tensor, ...], tuple[Tensor, ...], tuple[Tensor, ...]
    ]:
        source_features = self.source_projection(source)
        modulus_features = self.modulus_projection(modulus)
        boundary_features = self._boundaries(source)
        context = self.input_norm(
            self.input_mix(
                torch.cat(
                    (
                        source_features,
                        modulus_features,
                        boundary_features.expand(source.shape[0], -1, -1),
                    ),
                    dim=-1,
                )
            )
        )
        state = context + self.seed
        phase_logits: list[Tensor] = []
        phase_bit_logits: list[Tensor] = []
        phase_ordinal_logits: list[Tensor] = []
        for _ in range(SWEEPS):
            for _ in range(self.digits):
                state = self.cell(state, context)
            logits = self.output(self.output_norm(state))
            phase_logits.append(logits)
            pooled = self.output_norm(state).mean(dim=1)
            phase_bit_logits.append(
                self.bit_output(pooled).reshape(
                    source.shape[0], self.bit_width, 2
                )
            )
            phase_ordinal_logits.append(self.ordinal_output(pooled))
            probabilities = F.softmax(logits / temperature, dim=-1)
            state = state + torch.sigmoid(self.feedback_gate) * self.feedback(
                probabilities
            )
        return (
            probabilities,
            tuple(phase_logits),
            tuple(phase_bit_logits),
            tuple(phase_ordinal_logits),
        )


class Model(nn.Module):
    num_loops = SWEEPS

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.max_length = spec.max_seq_len
        # Three markers and at least one T digit leave the remaining prompt
        # capacity split between N and x.  Using ``-5`` drops a leading digit
        # for even prompt lengths (for example max_seq_len=10).
        self.digits = max(2, (spec.max_seq_len - 4) // 2)
        self.bit_width = (3322 * self.digits + 999) // 1000 + 1
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

    @torch.autocast(device_type="cuda", enabled=False)
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
        # Reach every evaluator-provided endpoint during training.  A fixed
        # three-step cap makes larger-T rows supervise the wrong state and
        # disconnects their true endpoints from the loss.
        outer_steps = int(t_values.max().item())
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
        terminal_bit_phases = [
            torch.zeros(
                batch,
                self.bit_width,
                2,
                device=input_ids.device,
                dtype=register.dtype,
            )
            for _ in range(SWEEPS)
        ]
        terminal_ordinal_phases = [
            torch.zeros(
                batch,
                ORDINAL_BINS,
                device=input_ids.device,
                dtype=register.dtype,
            )
            for _ in range(SWEEPS)
        ]
        for outer_step in range(outer_steps):
            transition_output = self.transition(register, modulus, temperature)
            if len(transition_output) == 4:
                candidate, phases, bit_phases, ordinal_phases = transition_output
            elif len(transition_output) == 3:
                candidate, phases, bit_phases = transition_output
                ordinal_phases = tuple(
                    register.new_zeros(batch, ORDINAL_BINS)
                    for _ in range(SWEEPS)
                )
            else:
                candidate, phases = transition_output
                bit_phases = tuple(
                    register.new_zeros(batch, self.bit_width, 2)
                    for _ in range(SWEEPS)
                )
                ordinal_phases = tuple(
                    register.new_zeros(batch, ORDINAL_BINS)
                    for _ in range(SWEEPS)
                )
            terminal = (t_values == outer_step + 1)[:, None, None]
            terminal_phases = [
                torch.where(terminal, phase, previous)
                for phase, previous in zip(phases, terminal_phases)
            ]
            terminal_bit_phases = [
                torch.where(terminal, phase, previous)
                for phase, previous in zip(bit_phases, terminal_bit_phases)
            ]
            ordinal_terminal = (t_values == outer_step + 1)[:, None]
            terminal_ordinal_phases = [
                torch.where(ordinal_terminal, phase, previous)
                for phase, previous in zip(
                    ordinal_phases, terminal_ordinal_phases
                )
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
        decimal_powers = 10 ** torch.arange(
            self.digits, device=modulus.device
        )
        modulus_integer = (
            modulus.argmax(dim=-1).long() * decimal_powers[None]
        ).sum(dim=1)
        return logits, {
            "phase_logits": phase_logits,
            "bit_logits": tuple(terminal_bit_phases),
            "bit_width": self.bit_width,
            "ordinal_logits": tuple(terminal_ordinal_phases),
            "modulus_integer": modulus_integer,
        }


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
    if PHASE_WEIGHT == 0.0:
        return endpoint
    phases = [
        _masked_cross_entropy(_target_aligned(phase, batch), batch)
        for phase in batch.auxiliary["phase_logits"]
    ]
    labels = batch.labels
    target = torch.zeros(
        labels.shape[0], device=labels.device, dtype=torch.long
    )
    for position in range(labels.shape[1]):
        valid = batch.valid_mask[:, position]
        digit = (labels[:, position] - DIGIT_OFFSET).clamp(0, 9)
        target = torch.where(valid, target * 10 + digit, target)
    width = int(batch.auxiliary["bit_width"])
    positions = torch.arange(width, device=target.device)
    target_bits = ((target[:, None] >> positions[None]) & 1).long()
    bit_losses = [
        F.cross_entropy(logits.transpose(1, 2), target_bits)
        for logits in batch.auxiliary["bit_logits"]
    ]
    modulus = batch.auxiliary["modulus_integer"].clamp_min(1)
    thresholds = torch.arange(
        1, ORDINAL_BINS + 1, device=target.device
    )[None]
    ordinal_target = (
        target[:, None] * (ORDINAL_BINS + 1)
        >= modulus[:, None] * thresholds
    ).to(torch.float32)
    ordinal_losses = [
        F.binary_cross_entropy_with_logits(logits.float(), ordinal_target)
        for logits in batch.auxiliary["ordinal_logits"]
    ]
    return (
        endpoint
        + PHASE_WEIGHT * torch.stack(phases).mean()
        + BIT_WEIGHT * torch.stack(bit_losses).mean()
        + ORDINAL_WEIGHT * torch.stack(ordinal_losses).mean()
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
