"""Position-only recurrent mixer with decoupled learned local feedback.

Digit values determine the messages, but never the routing weights. Every
attention head learns one relative-position kernel shared across examples,
refinement depth, and outer T. Each refinement state also predicts the real
endpoint digit through a separately learned local classifier. Stop-gradient
paths decouple feature and classifier updates, and an orthogonality loss keeps
the feedback coordinate conditioned without a custom backward.
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
HEADS = 4
REFINEMENT_STEPS = 4
MAX_OUTER_STEPS = 64
BASE_LR = 2.0e-3
LOCAL_FEEDBACK_WEIGHT = 0.20
LOCAL_HEAD_WEIGHT = 0.05
LOCAL_ORTHOGONAL_WEIGHT = 0.01
LOCAL_LOGIT_SCALE = 2.0


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


class PositionOnlyMixer(nn.Module):
    """Global learned relative routing whose weights cannot see content."""

    def __init__(self, length: int) -> None:
        super().__init__()
        self.length = length
        self.relative_bias = nn.Parameter(torch.empty(HEADS, 2 * length - 1))
        nn.init.normal_(self.relative_bias, std=0.02)
        self.value = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.output = nn.Linear(HIDDEN, HIDDEN, bias=False)

    def forward(self, state: Tensor) -> Tensor:
        positions = torch.arange(self.length, device=state.device)
        offsets = positions[:, None] - positions[None, :] + self.length - 1
        routing = F.softmax(self.relative_bias[:, offsets], dim=-1)
        values = self.value(state).reshape(
            state.shape[0], self.length, HEADS, HIDDEN // HEADS
        ).transpose(1, 2)
        mixed = torch.einsum("hij,bhjd->bhid", routing, values)
        mixed = mixed.transpose(1, 2).reshape_as(state)
        return self.output(mixed)


class ProductPositionSquare(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = width
        self.columns = 2 * width - 1
        self.pair_embedding = nn.Parameter(
            torch.empty(NUM_DIGITS, NUM_DIGITS, HIDDEN)
        )
        nn.init.normal_(self.pair_embedding, std=HIDDEN**-0.5)
        self.modulus_projection = nn.Linear(NUM_DIGITS, HIDDEN, bias=False)
        self.register_projection = nn.Linear(NUM_DIGITS, HIDDEN, bias=False)
        self.input_projection = nn.Linear(3 * HIDDEN, HIDDEN, bias=False)
        self.mix_norm = RMSNorm(HIDDEN)
        self.mixer = PositionOnlyMixer(self.columns)
        self.mlp_norm = RMSNorm(HIDDEN)
        self.mlp = nn.Sequential(
            nn.Linear(HIDDEN, 2 * HIDDEN, bias=False),
            nn.SiLU(),
            nn.Linear(2 * HIDDEN, HIDDEN, bias=False),
        )
        self.mix_gate = nn.Parameter(torch.empty(HIDDEN))
        self.mlp_gate = nn.Parameter(torch.empty(HIDDEN))
        nn.init.normal_(self.mix_gate, mean=-2.0, std=0.02)
        nn.init.normal_(self.mlp_gate, mean=-2.0, std=0.02)
        self.output_norm = RMSNorm(HIDDEN)
        self.output = nn.Linear(HIDDEN, NUM_DIGITS)
        local_classifier = torch.empty(NUM_DIGITS, HIDDEN)
        nn.init.orthogonal_(local_classifier)
        self.local_classifier = nn.Parameter(local_classifier)

    def _product_columns(self, register: Tensor) -> Tensor:
        batch = register.shape[0]
        pair_features = torch.einsum(
            "bid,bje,deh->bijh", register, register, self.pair_embedding
        )
        significance = (
            torch.arange(self.width, device=register.device)[:, None]
            + torch.arange(self.width, device=register.device)[None, :]
        ).reshape(1, self.width * self.width, 1)
        significance = significance.expand(batch, -1, HIDDEN)
        columns = torch.zeros(
            batch,
            self.columns,
            HIDDEN,
            device=register.device,
            dtype=pair_features.dtype,
        )
        return columns.scatter_add(
            1, significance, pair_features.reshape(batch, -1, HIDDEN)
        )

    def forward(
        self, register: Tensor, modulus: Tensor, temperature: float
    ) -> tuple[
        Tensor, tuple[Tensor, ...], tuple[tuple[Tensor, Tensor], ...]
    ]:
        product = self._product_columns(register)
        padding = self.columns - self.width
        modulus_columns = F.pad(modulus, (0, 0, 0, padding))
        register_columns = F.pad(register, (0, 0, 0, padding))
        state = self.input_projection(
            torch.cat(
                (
                    product,
                    self.modulus_projection(modulus_columns),
                    self.register_projection(register_columns),
                ),
                dim=-1,
            )
        )
        phases: list[Tensor] = []
        local_phases: list[tuple[Tensor, Tensor]] = []
        for _ in range(REFINEMENT_STEPS):
            state = state + torch.sigmoid(self.mix_gate) * self.mixer(
                self.mix_norm(state)
            )
            state = state + torch.sigmoid(self.mlp_gate) * self.mlp(
                self.mlp_norm(state)
            )
            normalized = self.output_norm(state[:, : self.width])
            phases.append(self.output(normalized))
            local_state = F.rms_norm(state[:, : self.width], (HIDDEN,))
            local_phases.append(
                (
                    LOCAL_LOGIT_SCALE
                    * F.linear(
                        local_state, self.local_classifier.detach()
                    ),
                    LOCAL_LOGIT_SCALE
                    * F.linear(
                        local_state.detach(), self.local_classifier
                    ),
                )
            )
        return (
            F.softmax(phases[-1] / temperature, dim=-1),
            tuple(phases),
            tuple(local_phases),
        )


class Model(nn.Module):
    num_loops = 1

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.width = max(2, (spec.max_seq_len - 4) // 2)
        self.square = ProductPositionSquare(self.width)
        self.training_temperature = 1.2
        self.eval_temperature = 0.12

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
            batch, self.width, 1, device=input_ids.device
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
        return (
            self._register(input_ids, digits & after_n & ~after_x),
            self._register(input_ids, digits & after_x & ~after_t),
            self._time_steps(input_ids, valid),
        )

    def _place(self, digit_logits_lsd: Tensor, lengths: Tensor, prompt: int) -> Tensor:
        digit_logits = digit_logits_lsd.flip(1)
        destination = lengths[:, None] - self.width + torch.arange(
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
        self, input_ids: Tensor, attention_mask: Tensor | None = None
    ) -> tuple[Tensor, dict[str, object]]:
        _, prompt = input_ids.shape
        if attention_mask is None:
            attention_mask = input_ids != PAD
        valid = attention_mask.bool()
        modulus, register, t_values = self._parse(input_ids, valid)
        loops = 3 if self.training else int(t_values.max().item())
        temperature = (
            self.training_temperature if self.training else self.eval_temperature
        )
        first_phases: tuple[Tensor, ...] = ()
        first_local_phases: tuple[tuple[Tensor, Tensor], ...] = ()
        for outer_step in range(loops):
            transition_input = (
                register.detach()
                if self.training and outer_step > 0
                else register
            )
            candidate, phases, local_phases = self.square(
                transition_input, modulus, temperature
            )
            if outer_step == 0:
                first_phases = phases
                first_local_phases = local_phases
            register = torch.where(
                (t_values > outer_step)[:, None, None], candidate, register
            )
        lengths = valid.sum(dim=1)
        return self._place(register.clamp_min(1e-8).log(), lengths, prompt), {
            "t_values": t_values,
            "phase_logits": tuple(
                self._place(phase, lengths, prompt) for phase in first_phases
            ),
            "local_phase_logits": tuple(
                tuple(
                    self._place(channel, lengths, prompt)
                    for channel in phase
                )
                for phase in first_local_phases
            ),
            "local_orthogonality": self._local_orthogonality(),
        }

    def _local_orthogonality(self) -> Tensor:
        classifier = F.normalize(
            self.square.local_classifier.float(), dim=-1
        )
        gram = classifier @ classifier.transpose(0, 1)
        identity = torch.eye(
            NUM_DIGITS, device=gram.device, dtype=gram.dtype
        )
        return (gram - identity).square().mean()


def _target_aligned(full_logits: Tensor, batch: TokenLossBatch) -> Tensor:
    if batch.target_positions is None:
        return full_logits[:, : batch.logits.shape[1]]
    rows = torch.arange(full_logits.shape[0], device=full_logits.device)[:, None]
    return full_logits[rows, batch.target_positions.clamp_min(0)]


def _masked_ce(logits: Tensor, labels: Tensor, mask: Tensor) -> Tensor:
    losses = F.cross_entropy(
        logits.transpose(1, 2), labels, ignore_index=-100, reduction="none"
    )
    weights = mask.float()
    return (losses * weights).sum() / weights.sum().clamp_min(1.0)


def token_training_loss(batch: TokenLossBatch) -> Tensor:
    valid = batch.valid_mask
    t_values = batch.auxiliary["t_values"]
    row_weight = torch.where(
        t_values == 1,
        torch.full_like(t_values, 4.0, dtype=torch.float32),
        torch.ones_like(t_values, dtype=torch.float32),
    )
    loss = _masked_ce(
        batch.logits, batch.labels, valid.float() * row_weight[:, None]
    )
    t1 = valid & (t_values == 1)[:, None]
    phases = [
        _masked_ce(_target_aligned(item, batch), batch.labels, t1)
        for item in batch.auxiliary["phase_logits"]
    ]
    if phases:
        loss = loss + 0.35 * torch.stack(phases).mean()
    feature_terms: list[Tensor] = []
    head_terms: list[Tensor] = []
    for feature_logits, head_logits in batch.auxiliary["local_phase_logits"]:
        feature_terms.append(
            _masked_ce(
                _target_aligned(feature_logits, batch), batch.labels, t1
            )
        )
        head_terms.append(
            _masked_ce(
                _target_aligned(head_logits, batch), batch.labels, t1
            )
        )
    if feature_terms:
        loss = loss + LOCAL_FEEDBACK_WEIGHT * torch.stack(feature_terms).mean()
        loss = loss + LOCAL_HEAD_WEIGHT * torch.stack(head_terms).mean()
    loss = (
        loss
        + LOCAL_ORTHOGONAL_WEIGHT
        * batch.auxiliary["local_orthogonality"]
    )
    return loss


class WallClockSchedule:
    def __init__(
        self, optimizer: torch.optim.Optimizer, model: Model, seconds: float
    ) -> None:
        self.optimizer = optimizer
        self.model = model
        self.started = time.monotonic()
        self.budget = max(float(seconds) * 0.98, 1.0)

    def step(self) -> None:
        fraction = min((time.monotonic() - self.started) / self.budget, 1.0)
        if fraction < 0.05:
            multiplier = 0.1 + 18.0 * fraction
        elif fraction < 0.85:
            multiplier = 1.0
        else:
            progress = (fraction - 0.85) / 0.15
            multiplier = 0.08 + 0.92 * 0.5 * (1.0 + math.cos(math.pi * progress))
        for group in self.optimizer.param_groups:
            group["lr"] = group["base_lr"] * multiplier
        self.model.training_temperature = max(
            0.20, 1.20 * (0.20 / 1.20) ** fraction
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
    eval_batch_size=256,
)
