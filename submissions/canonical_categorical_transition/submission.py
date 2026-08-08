"""Canonical categorical cellular program for a hidden recurrence.

The model turns the positive endpoint-identification argument into an explicit
architecture.  The state that survives every local update contains only a
distribution over the ten observable decimal digits and a small categorical
controller. One translation-equivariant cell updates that state everywhere and
at every recurrence depth. The digit state
is straight-through canonicalized and is itself the next recurrence input, so no
free decoder can hide a non-reusable endpoint code.

The cell receives only neighboring categorical states, immutable prompt digit
registers, boundary flags, and a generic learned clock. No digit product,
arithmetic operation, carry, quotient, comparison, or reduction transition is
fixed. Training uses evaluator endpoint labels plus label-free categorical
sharpness/balance terms.  All trainable state is randomly initialized.
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
CONTROL_STATES = 16
HIDDEN = 64
COMPUTE_SWEEPS = 2
MAX_OUTER_STEPS = 64
BASE_LR = 2.0e-3
PHASE_WEIGHT = 0.25
CONTROLLER_MI_WEIGHT = 0.01
DIGIT_ENTROPY_WEIGHT = 0.01


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


def _categorical(
    logits: Tensor,
    temperature: float,
    hardness: float,
) -> tuple[Tensor, Tensor]:
    """Return an annealed straight-through categorical state and its soft view."""

    soft = F.softmax(logits.float() / temperature, dim=-1).to(logits.dtype)
    hard = F.one_hot(soft.argmax(dim=-1), soft.shape[-1]).to(soft.dtype)
    state = soft + float(hardness) * (hard - soft).detach()
    return state, soft


class CategoricalCell(nn.Module):
    """One tied local update over observable digits and a finite controller."""

    def __init__(self) -> None:
        super().__init__()
        local_width = 3 * (NUM_DIGITS + CONTROL_STATES)
        context_width = 2 * NUM_DIGITS + HIDDEN + 2 + 4
        self.input = nn.Linear(local_width + context_width, 2 * HIDDEN)
        self.norm = RMSNorm(2 * HIDDEN)
        self.body = nn.Sequential(
            nn.Linear(2 * HIDDEN, 2 * HIDDEN, bias=False),
            nn.SiLU(),
            nn.Linear(2 * HIDDEN, HIDDEN, bias=False),
            nn.SiLU(),
        )
        self.digit = nn.Linear(HIDDEN, NUM_DIGITS)
        self.controller = nn.Linear(HIDDEN, CONTROL_STATES)
        self.identity_scale = nn.Parameter(torch.tensor(0.5))

    @staticmethod
    def _neighbors(value: Tensor) -> tuple[Tensor, Tensor]:
        zero = torch.zeros_like(value[:, :1])
        left = torch.cat((zero, value[:, :-1]), dim=1)
        right = torch.cat((value[:, 1:], zero), dim=1)
        return left, right

    def forward(
        self,
        digits: Tensor,
        controller: Tensor,
        source: Tensor,
        modulus: Tensor,
        clock: Tensor,
        boundary: Tensor,
        phase: Tensor,
        temperature: float,
        hardness: float,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        left_digit, right_digit = self._neighbors(digits)
        left_control, right_control = self._neighbors(controller)
        batch, width, _ = digits.shape
        context = torch.cat(
            (
                left_digit,
                digits,
                right_digit,
                left_control,
                controller,
                right_control,
                source,
                modulus,
                clock[:, None].expand(-1, width, -1),
                boundary.expand(batch, -1, -1),
                phase[:, None].expand(-1, width, -1),
            ),
            dim=-1,
        )
        hidden = self.body(self.norm(self.input(context)))
        residual = F.softplus(self.identity_scale)
        digit_logits = self.digit(hidden) + residual * digits
        controller_logits = self.controller(hidden) + residual * controller
        next_digits, digit_soft = _categorical(
            digit_logits, temperature, hardness
        )
        next_controller, controller_soft = _categorical(
            controller_logits, temperature, hardness
        )
        return (
            next_digits,
            next_controller,
            digit_logits,
            controller_logits,
            digit_soft,
            controller_soft,
        )


class CanonicalProgram(nn.Module):
    """One recurrence step executed by one generic finite local program."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = width
        self.compute_steps = COMPUTE_SWEEPS * width
        self.clock_embedding = nn.Parameter(
            torch.empty(COMPUTE_SWEEPS, HIDDEN)
        )
        nn.init.normal_(self.clock_embedding, std=HIDDEN**-0.5)
        self.cell = CategoricalCell()

    def forward(
        self,
        register: Tensor,
        modulus: Tensor,
        temperature: float,
        hardness: float,
    ) -> tuple[
        Tensor,
        Tensor,
        tuple[Tensor, ...],
        Tensor,
        Tensor,
    ]:
        batch = register.shape[0]
        digits = register / register.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        controller = torch.zeros(
            batch,
            self.width,
            CONTROL_STATES,
            device=register.device,
            dtype=register.dtype,
        )
        controller[:, :, 0] = 1.0
        boundary = torch.zeros(
            1, self.width, 2, device=register.device, dtype=register.dtype
        )
        boundary[:, 0, 0] = 1.0
        boundary[:, -1, 1] = 1.0

        phases: list[Tensor] = []
        controller_soft_states: list[Tensor] = []
        digit_soft_states: list[Tensor] = []
        digit_logits = digits.clamp_min(1e-6).log()
        for step in range(self.compute_steps):
            sweep = step // self.width
            within_sweep = step % self.width
            phase = torch.tensor(
                (
                    float(step == 0),
                    float(step + 1 == self.compute_steps),
                    float(within_sweep == 0),
                    float(within_sweep + 1 == self.width),
                ),
                device=register.device,
                dtype=register.dtype,
            )[None].expand(batch, -1)
            (
                digits,
                controller,
                digit_logits,
                _,
                digit_soft,
                controller_soft,
            ) = self.cell(
                digits,
                controller,
                register,
                modulus,
                self.clock_embedding[sweep][None].expand(batch, -1),
                boundary,
                phase,
                temperature,
                hardness,
            )
            controller_soft_states.append(controller_soft)
            digit_soft_states.append(digit_soft)
            if within_sweep + 1 == self.width:
                phases.append(digit_logits)

        return (
            digits,
            digit_logits,
            tuple(phases),
            torch.stack(controller_soft_states, dim=1),
            torch.stack(digit_soft_states, dim=1),
        )


class Model(nn.Module):
    num_loops = 1

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.max_length = spec.max_seq_len
        # A prompt contains N and x plus three markers and one or two T
        # digits.  Subtracting five under-allocates even-length prompts (for
        # example max_seq_len=10 with three-digit N and x), irreversibly
        # truncating the most-significant decimal digit.
        self.width = max(2, (spec.max_seq_len - 4) // 2)
        self.program = CanonicalProgram(self.width)
        self.training_temperature = 1.25
        self.eval_temperature = 0.20
        self.canonical_hardness = 0.15

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
            batch, self.width, 1, device=input_ids.device, dtype=torch.float32
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
        n_mask = digits & after_n & ~after_x
        x_mask = digits & after_x & ~after_t
        return (
            self._register(input_ids, n_mask),
            self._register(input_ids, x_mask),
            self._time_steps(input_ids, valid),
        )

    def _place_logits(
        self, digit_logits_lsd: Tensor, input_lengths: Tensor, prompt: int
    ) -> Tensor:
        digit_logits = digit_logits_lsd.flip(1)
        destination = input_lengths[:, None] - self.width + torch.arange(
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
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, object]]:
        _, prompt_length = input_ids.shape
        if attention_mask is None:
            attention_mask = input_ids != PAD
        valid = attention_mask.bool()
        modulus, register, t_values = self._parse(input_ids, valid)
        # Every labelled endpoint must actually be reached.  Capping this
        # rollout silently disconnects rows with larger T from the supervised
        # loss: their terminal logits remain the constant zero tensor.
        loops = int(t_values.max().item())
        temperature = (
            self.training_temperature if self.training else self.eval_temperature
        )
        hardness = self.canonical_hardness if self.training else 1.0

        terminal_logits = torch.zeros_like(register)
        terminal_phases = [torch.zeros_like(register) for _ in range(COMPUTE_SWEEPS)]
        first_controller: Tensor | None = None
        first_digit_soft: Tensor | None = None
        for outer_step in range(loops):
            (
                candidate,
                candidate_logits,
                phases,
                controller_soft,
                digit_soft,
            ) = self.program(
                register,
                modulus,
                temperature,
                hardness,
            )
            if outer_step == 0:
                first_controller = controller_soft
                first_digit_soft = digit_soft
            terminal = (t_values == outer_step + 1)[:, None, None]
            terminal_logits = torch.where(
                terminal, candidate_logits, terminal_logits
            )
            terminal_phases = [
                torch.where(terminal, phase, previous)
                for phase, previous in zip(
                    phases, terminal_phases, strict=True
                )
            ]
            active = (t_values > outer_step)[:, None, None]
            register = torch.where(active, candidate, register)

        if first_controller is None or first_digit_soft is None:
            raise RuntimeError("at least one recurrence step is required")
        lengths = valid.sum(dim=1)
        logits = self._place_logits(
            terminal_logits, lengths, prompt_length
        )
        phase_logits = tuple(
            self._place_logits(phase, lengths, prompt_length)
            for phase in terminal_phases
        )
        return logits, {
            "t_values": t_values,
            "phase_logits": phase_logits,
            "controller_soft": first_controller,
            "digit_soft": first_digit_soft,
        }


def _target_aligned(full_logits: Tensor, batch: TokenLossBatch) -> Tensor:
    if batch.target_positions is None:
        return full_logits[:, : batch.logits.shape[1]]
    rows = torch.arange(full_logits.shape[0], device=full_logits.device)[:, None]
    return full_logits[rows, batch.target_positions.clamp_min(0)]


def _masked_loss(logits: Tensor, labels: Tensor, mask: Tensor) -> Tensor:
    losses = F.cross_entropy(
        logits.transpose(1, 2), labels, ignore_index=-100, reduction="none"
    )
    weights = mask.to(losses.dtype)
    return (losses * weights).sum() / weights.sum().clamp_min(1.0)


def _entropy(probability: Tensor) -> Tensor:
    return -(probability.clamp_min(1e-8).log() * probability).sum(dim=-1)


def token_training_loss(batch: TokenLossBatch) -> Tensor:
    valid = batch.valid_mask
    t_values = batch.auxiliary["t_values"]
    row_weight = torch.where(
        t_values == 1,
        torch.full_like(t_values, 4.0, dtype=torch.float32),
        torch.ones_like(t_values, dtype=torch.float32),
    )
    endpoint_losses = F.cross_entropy(
        batch.logits.transpose(1, 2),
        batch.labels,
        ignore_index=-100,
        reduction="none",
    )
    endpoint_mask = valid.float() * row_weight[:, None]
    loss = (
        (endpoint_losses * endpoint_mask).sum()
        / endpoint_mask.sum().clamp_min(1.0)
    )

    phase_terms = [
        _masked_loss(
            _target_aligned(full_logits, batch), batch.labels, valid
        )
        for full_logits in batch.auxiliary["phase_logits"]
    ]
    if phase_terms:
        loss = loss + PHASE_WEIGHT * torch.stack(phase_terms).mean()

    controller = batch.auxiliary["controller_soft"].float()
    local_entropy = _entropy(controller).mean()
    aggregate = controller.mean(dim=(0, 1, 2))
    aggregate_entropy = _entropy(aggregate)
    controller_mi = local_entropy - aggregate_entropy

    digit_soft = batch.auxiliary["digit_soft"].float()
    digit_entropy = _entropy(digit_soft).mean()
    return (
        loss
        + CONTROLLER_MI_WEIGHT * controller_mi
        + DIGIT_ENTROPY_WEIGHT * digit_entropy
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
        elif fraction < 0.85:
            multiplier = 1.0
        else:
            progress = (fraction - 0.85) / 0.15
            multiplier = 0.08 + 0.92 * 0.5 * (
                1.0 + math.cos(math.pi * progress)
            )
        for group in self.optimizer.param_groups:
            group["lr"] = group["base_lr"] * multiplier
        self.model.training_temperature = max(
            0.35, 1.25 * (0.35 / 1.25) ** fraction
        )
        self.model.canonical_hardness = min(
            1.0, 0.15 + 0.85 * fraction / 0.35
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
    batch_size=128,
    eval_batch_size=512,
)
