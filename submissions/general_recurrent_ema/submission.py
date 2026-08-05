"""Temporal EMA-teacher consistency for a generic recurrent reasoner.

Student and teacher receive the same untouched prompt-derived registers.  Both
branches are computed inside one forward pass during training.  The endpoint
prediction is always the student's and retains a direct gradient path.  A
frozen teacher supplies representation and readout targets, and is updated only
after optimizer steps by the evaluator-called scheduler using an exponential
moving average of student parameters.

The transition contains no arithmetic features or process labels.  T controls
only how many times the same complete learned transition is applied.
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
MAX_OUTER_STEPS = 64
BASE_LR = 1.0e-3
EMA_DECAY = 0.995
CONSISTENCY_TEMPERATURE = 2.0
DEEP_WEIGHT = 0.45
REPRESENTATION_WEIGHT = 0.05
READOUT_WEIGHT = 0.15


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
    """One general token-mixing block, tied across refinement segments."""

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


class RecurrentTransition(nn.Module):
    """Refine generic answer and scratch slots through a discrete interface."""

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
        self.student_transition = RecurrentTransition(self.digits)
        self.teacher_transition = RecurrentTransition(self.digits)
        self.teacher_transition.load_state_dict(self.student_transition.state_dict())
        self.teacher_transition.requires_grad_(False)
        self.training_outer_steps = 3
        self.training_temperature = 1.0
        self.eval_temperature = 0.10
        self.detach_segments = True
        self.consistency_scale = 0.0

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

    @staticmethod
    def _select_terminal(
        previous: list[Tensor],
        current: tuple[Tensor, ...],
        terminal: Tensor,
    ) -> list[Tensor]:
        return [
            torch.where(terminal, value, old)
            for value, old in zip(current, previous)
        ]

    def _run_transition(
        self,
        transition: RecurrentTransition,
        register: Tensor,
        modulus: Tensor,
        t_values: Tensor,
        temperature: float,
        detach_segments: bool,
    ) -> tuple[Tensor, list[Tensor], list[Tensor]]:
        batch = register.shape[0]
        terminal_phases = [
            torch.zeros(
                batch, self.digits, NUM_DIGITS,
                device=register.device, dtype=register.dtype,
            )
            for _ in range(SEGMENTS)
        ]
        terminal_states = [
            torch.zeros(
                batch, 2 * self.digits, WIDTH,
                device=register.device, dtype=register.dtype,
            )
            for _ in range(SEGMENTS)
        ]
        for outer_step in range(self.training_outer_steps):
            source = register.detach() if outer_step > 0 else register
            candidate, phases, states = transition(
                source, modulus, temperature, detach_segments
            )
            terminal = (t_values == outer_step + 1)[:, None, None]
            terminal_phases = self._select_terminal(
                terminal_phases, phases, terminal
            )
            terminal_states = self._select_terminal(
                terminal_states, states, terminal
            )
            active = (t_values > outer_step)[:, None, None]
            register = torch.where(active, candidate, register)
        return register, terminal_phases, terminal_states

    @torch.no_grad()
    def update_teacher(self, decay: float) -> None:
        student_parameters = dict(self.student_transition.named_parameters())
        for name, teacher in self.teacher_transition.named_parameters():
            teacher.lerp_(student_parameters[name], 1.0 - decay)
        student_buffers = dict(self.student_transition.named_buffers())
        for name, teacher in self.teacher_transition.named_buffers():
            teacher.copy_(student_buffers[name])

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, object]]:
        _, prompt = input_ids.shape
        if attention_mask is None:
            attention_mask = input_ids != PAD
        valid = attention_mask.bool()
        modulus, initial_register, t_values = self._parse(input_ids, valid)
        lengths = valid.sum(dim=1)

        if not self.training:
            register = initial_register
            for outer_step in range(MAX_OUTER_STEPS):
                candidate, _, _ = self.student_transition(
                    register, modulus, self.eval_temperature, False
                )
                active = (t_values > outer_step)[:, None, None]
                register = torch.where(active, candidate, register)
            logits = self._place_logits(
                register.clamp_min(1e-8).log(), lengths, prompt
            )
            return logits, {}

        student_register, student_phases, student_states = self._run_transition(
            self.student_transition,
            initial_register,
            modulus,
            t_values,
            self.training_temperature,
            self.detach_segments,
        )
        with torch.no_grad():
            _, teacher_phases, teacher_states = self._run_transition(
                self.teacher_transition,
                initial_register,
                modulus,
                t_values,
                self.training_temperature,
                False,
            )
        logits = self._place_logits(
            student_register.clamp_min(1e-8).log(), lengths, prompt
        )
        return logits, {
            "student_phases": tuple(
                self._place_logits(phase, lengths, prompt)
                for phase in student_phases
            ),
            "teacher_phases": tuple(
                self._place_logits(phase, lengths, prompt)
                for phase in teacher_phases
            ),
            "student_states": tuple(student_states),
            "teacher_states": tuple(teacher_states),
            "supervised_rows": t_values <= self.training_outer_steps,
            "consistency_scale": self.consistency_scale,
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


def _teacher_representation_loss(
    student: Tensor, teacher: Tensor, rows: Tensor
) -> Tensor:
    student = F.normalize(student.float(), dim=-1)
    teacher = F.normalize(teacher.float(), dim=-1)
    losses = 1.0 - (student * teacher).sum(dim=-1)
    weights = rows[:, None].to(losses.dtype)
    return (losses * weights).sum() / (
        weights.sum().clamp_min(1.0) * losses.shape[1]
    )


def _teacher_readout_loss(
    student: Tensor, teacher: Tensor, valid: Tensor
) -> Tensor:
    temperature = CONSISTENCY_TEMPERATURE
    student_log = F.log_softmax(student.float() / temperature, dim=-1)
    teacher_probability = F.softmax(teacher.float() / temperature, dim=-1)
    losses = F.kl_div(
        student_log, teacher_probability, reduction="none"
    ).sum(dim=-1) * temperature**2
    weights = valid.to(losses.dtype)
    return (losses * weights).sum() / weights.sum().clamp_min(1.0)


def token_training_loss(batch: TokenLossBatch) -> Tensor:
    endpoint = _masked_cross_entropy(batch.logits, batch.labels, batch.valid_mask)
    auxiliary = batch.auxiliary
    supervised_rows = auxiliary["supervised_rows"]
    supervised_valid = batch.valid_mask & supervised_rows[:, None]
    student_phases = tuple(
        _target_aligned(phase, batch) for phase in auxiliary["student_phases"]
    )
    teacher_phases = tuple(
        _target_aligned(phase, batch) for phase in auxiliary["teacher_phases"]
    )
    deep = torch.stack([
        _masked_cross_entropy(phase, batch.labels, supervised_valid)
        for phase in student_phases
    ]).mean()
    representation = torch.stack([
        _teacher_representation_loss(student, teacher, supervised_rows)
        for student, teacher in zip(
            auxiliary["student_states"], auxiliary["teacher_states"]
        )
    ]).mean()
    readout = torch.stack([
        _teacher_readout_loss(student, teacher, supervised_valid)
        for student, teacher in zip(student_phases, teacher_phases)
    ]).mean()
    scale = float(auxiliary["consistency_scale"])
    return endpoint + DEEP_WEIGHT * deep + scale * (
        REPRESENTATION_WEIGHT * representation + READOUT_WEIGHT * readout
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
        self.updates = 0

    def step(self) -> None:
        self.updates += 1
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
        self.model.consistency_scale = min(fraction / 0.20, 1.0)
        decay = min(EMA_DECAY, 1.0 - 1.0 / (self.updates + 1.0))
        self.model.update_teacher(decay)


def build_model(spec: ModelSpec) -> Model:
    model = Model(spec)
    assert_model_state(model, spec)
    return model


def build_optimizer(model: Model, spec: OptimizerSpec) -> OptimizerBundle:
    decay: list[Tensor] = []
    no_decay: list[Tensor] = []
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
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
