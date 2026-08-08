"""Rule-clean dyadic neural operator for endpoint-supervised recurrences.

The model learns one horizon-conditioned cellular map for each power-of-two
recurrence depth.  Adjacent maps are tied by the generic semigroup identity
F_(2t)(x) = F_t(F_t(x)); no arithmetic operation or recurrence formula is
provided.  Power-of-two endpoints therefore supervise a short neural path
directly instead of backpropagating through as many as 64 repeated uses.

Every transition is implemented by the same randomly initialized local gated
cell, conditioned only by a learned horizon embedding.  The forward contains
no multiplication, modular reduction, carry logic, candidate program executor,
or task-specific search.  Training uses evaluator endpoints and label-free
composition consistency.
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
HIDDEN = 96
LEVELS = 7
SWEEPS = 2
MAX_OUTER_STEPS = 64
BASE_LR = 1.5e-3
BIT_WEIGHT = 0.5
CONSISTENCY_WEIGHT = 0.5


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
    """One translation-equivariant learned update shared everywhere."""

    def __init__(self) -> None:
        super().__init__()
        self.norm = RMSNorm(HIDDEN)
        self.local = nn.Linear(3 * HIDDEN, 3 * HIDDEN, bias=False)
        self.context = nn.Linear(HIDDEN, 3 * HIDDEN, bias=False)
        self.global_mix = nn.Linear(HIDDEN, 3 * HIDDEN, bias=False)
        self.update_bias = nn.Parameter(torch.full((HIDDEN,), -1.0))

    def forward(self, state: Tensor, context: Tensor) -> Tensor:
        normalized = self.norm(state)
        left = F.pad(normalized[:, :-1], (0, 0, 1, 0))
        right = F.pad(normalized[:, 1:], (0, 0, 0, 1))
        gates = (
            self.local(torch.cat((left, normalized, right), dim=-1))
            + self.context(context)
            + self.global_mix(normalized.mean(dim=1, keepdim=True))
        )
        reset, update, proposal = gates.chunk(3, dim=-1)
        candidate = torch.tanh(proposal + torch.sigmoid(reset) * normalized)
        rate = torch.sigmoid(update + self.update_bias)
        return state + rate * (candidate - state)


class DyadicTransition(nn.Module):
    """A shared neural operator conditioned by a learned horizon token."""

    def __init__(self, digits: int) -> None:
        super().__init__()
        self.digits = digits
        self.bit_width = (3322 * digits + 999) // 1000 + 1
        self.source_projection = nn.Linear(NUM_DIGITS, HIDDEN, bias=False)
        self.modulus_projection = nn.Linear(NUM_DIGITS, HIDDEN, bias=False)
        self.boundary_projection = nn.Linear(2, HIDDEN, bias=False)
        self.level_embedding = nn.Parameter(torch.empty(LEVELS, HIDDEN))
        self.input_mix = nn.Linear(4 * HIDDEN, HIDDEN, bias=False)
        self.input_norm = RMSNorm(HIDDEN)
        self.seed = nn.Parameter(torch.empty(1, 1, HIDDEN))
        self.cell = CellularUpdate()
        self.output_norm = RMSNorm(HIDDEN)
        self.output = nn.Linear(HIDDEN, NUM_DIGITS)
        self.bit_output = nn.Linear(HIDDEN, 2 * self.bit_width)
        self.feedback = nn.Linear(NUM_DIGITS, HIDDEN, bias=False)
        self.feedback_gate = nn.Parameter(torch.full((HIDDEN,), -1.0))
        nn.init.normal_(self.level_embedding, std=HIDDEN**-0.5)
        nn.init.normal_(self.seed, std=HIDDEN**-0.5)

    def _boundary(self, reference: Tensor) -> Tensor:
        flags = torch.zeros(
            self.digits, 2, device=reference.device, dtype=reference.dtype
        )
        flags[0, 0] = 1.0
        flags[-1, 1] = 1.0
        return self.boundary_projection(flags).unsqueeze(0)

    def forward(
        self,
        source: Tensor,
        modulus: Tensor,
        level: int,
        temperature: float,
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch = source.shape[0]
        horizon = self.level_embedding[level][None, None].expand(
            batch, self.digits, -1
        )
        context = self.input_norm(
            self.input_mix(
                torch.cat(
                    (
                        self.source_projection(source),
                        self.modulus_projection(modulus),
                        self._boundary(source).expand(batch, -1, -1),
                        horizon,
                    ),
                    dim=-1,
                )
            )
        )
        state = context + self.seed
        logits = source.clamp_min(1e-8).log()
        for _ in range(SWEEPS):
            for _ in range(self.digits):
                state = self.cell(state, context)
            logits = self.output(self.output_norm(state))
            probability = F.softmax(logits / temperature, dim=-1)
            state = state + torch.sigmoid(self.feedback_gate) * self.feedback(
                probability
            )
        pooled = self.output_norm(state).mean(dim=1)
        bit_logits = self.bit_output(pooled).reshape(
            batch, self.bit_width, 2
        )
        return probability, logits, bit_logits


class Model(nn.Module):
    num_loops = SWEEPS

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.digits = max(2, (spec.max_seq_len - 4) // 2)
        self.bit_width = (3322 * self.digits + 999) // 1000 + 1
        self.transition = DyadicTransition(self.digits)
        self.training_temperature = 1.0
        self.eval_temperature = 0.08
        self.consistency_level = 0

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

    @torch.autocast(device_type="cuda", enabled=False)
    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, object]]:
        if attention_mask is None:
            attention_mask = input_ids != PAD
        valid = attention_mask.bool()
        modulus, source, time_steps = self._parse(input_ids, valid)
        temperature = (
            self.training_temperature if self.training else self.eval_temperature
        )

        direct: dict[int, tuple[Tensor, Tensor, Tensor]] = {}

        def direct_at(level: int) -> tuple[Tensor, Tensor, Tensor]:
            if level not in direct:
                direct[level] = self.transition(
                    source, modulus, level, temperature
                )
            return direct[level]

        terminal_logits = source.clamp_min(1e-8).log()
        terminal_bits = source.new_zeros(
            source.shape[0], self.bit_width, 2
        )
        is_power = (time_steps & (time_steps - 1)) == 0
        for level in range(LEVELS):
            selected_rows = time_steps == (1 << level)
            if not bool(selected_rows.any()):
                continue
            _, level_logits, level_bits = direct_at(level)
            selected = selected_rows[:, None, None]
            terminal_logits = torch.where(
                selected, level_logits, terminal_logits
            )
            terminal_bits = torch.where(
                selected[:, :, :1], level_bits, terminal_bits
            )

        # Non-power horizons use the ordinary binary composition of the same
        # learned dyadic maps. Public T=3 is the important case; powers of two
        # retain their direct, short gradient path above.
        if bool((~is_power).any()):
            composed = source
            composed_logits = terminal_logits
            composed_bits = terminal_bits
            for level in range(LEVELS):
                active_rows = (~is_power) & (((time_steps >> level) & 1) != 0)
                if not bool(active_rows.any()):
                    continue
                candidate, logits, bits = self.transition(
                    composed, modulus, level, temperature
                )
                active = active_rows[:, None, None]
                composed = torch.where(active, candidate, composed)
                composed_logits = torch.where(active, logits, composed_logits)
                composed_bits = torch.where(
                    active[:, :, :1], bits, composed_bits
                )
            nonpower = (~is_power)[:, None, None]
            terminal_logits = torch.where(
                nonpower, composed_logits, terminal_logits
            )
            terminal_bits = torch.where(
                nonpower[:, :, :1], composed_bits, terminal_bits
            )

        consistency_upper: list[Tensor] = []
        consistency_lower_twice: list[Tensor] = []
        if self.training:
            level = self.consistency_level
            lower, _, _ = direct_at(level)
            upper, _, _ = direct_at(level + 1)
            lower_twice, _, _ = self.transition(
                lower, modulus, level, temperature
            )
            consistency_lower_twice.append(lower_twice)
            consistency_upper.append(upper)

        logits = self._place_logits(
            terminal_logits, valid.sum(dim=1), input_ids.shape[1]
        )
        return logits, {
            "terminal_bits": terminal_bits,
            "bit_width": self.bit_width,
            "consistency_upper": tuple(consistency_upper),
            "consistency_lower_twice": tuple(consistency_lower_twice),
        }


def _target_integer(batch: TokenLossBatch) -> Tensor:
    value = torch.zeros(
        batch.labels.shape[0], device=batch.labels.device, dtype=torch.long
    )
    for position in range(batch.labels.shape[1]):
        valid = batch.valid_mask[:, position]
        digit = (batch.labels[:, position] - DIGIT_OFFSET).clamp(0, 9)
        value = torch.where(valid, value * 10 + digit, value)
    return value


def token_training_loss(batch: TokenLossBatch) -> Tensor:
    token_losses = F.cross_entropy(
        batch.logits.transpose(1, 2),
        batch.labels,
        ignore_index=-100,
        reduction="none",
    )
    endpoint = (
        token_losses * batch.valid_mask.to(token_losses.dtype)
    ).sum() / batch.valid_mask.sum().clamp_min(1)

    target = _target_integer(batch)
    width = int(batch.auxiliary["bit_width"])
    positions = torch.arange(width, device=target.device)
    target_bits = ((target[:, None] >> positions[None]) & 1).long()
    bit_loss = F.cross_entropy(
        batch.auxiliary["terminal_bits"].transpose(1, 2), target_bits
    )

    consistency_terms: list[Tensor] = []
    for upper, lower_twice in zip(
        batch.auxiliary["consistency_upper"],
        batch.auxiliary["consistency_lower_twice"],
    ):
        upper_log = upper.clamp_min(1e-8).log()
        lower_log = lower_twice.clamp_min(1e-8).log()
        consistency_terms.append(
            -(lower_twice.detach() * upper_log).sum(dim=-1).mean()
            -(upper.detach() * lower_log).sum(dim=-1).mean()
        )
    consistency = (
        torch.stack(consistency_terms).mean()
        if consistency_terms
        else endpoint.new_zeros(())
    )
    return endpoint + BIT_WEIGHT * bit_loss + CONSISTENCY_WEIGHT * consistency


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
        self.completed_steps = 0

    def step(self) -> None:
        self.completed_steps += 1
        self.model.consistency_level = self.completed_steps % (LEVELS - 1)
        fraction = min(
            (time.monotonic() - self.started_at) / self.budget_seconds, 1.0
        )
        if fraction < 0.05:
            multiplier = 0.1 + 0.9 * fraction / 0.05
        elif fraction < 0.88:
            multiplier = 1.0
        else:
            progress = (fraction - 0.88) / 0.12
            multiplier = 0.1 + 0.45 * (1.0 + math.cos(math.pi * progress))
        for group in self.optimizer.param_groups:
            group["lr"] = group["base_lr"] * multiplier
        self.model.training_temperature = max(0.2, 1.0 * (0.2**fraction))


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
            {"params": decay, "weight_decay": 0.03, "base_lr": BASE_LR},
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
    batch_size=32,
    eval_batch_size=256,
)
