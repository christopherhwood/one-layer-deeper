"""Product-aware ripple squaring with evaluator-owned PCGrad supervision.

This is the benchmark-compatible descendant of the successful sequential-ripple
diagnostic.  It keeps a positional digit register and weight-tied recurrent
scans, exposes the full digit-pair outer product so multiplication does not have
to be rediscovered, and asks the learned scans to perform carry, comparison, and
modular reduction. The first evaluator pass differentiates only the real final
endpoint; the second differentiates only legitimate T=1 refinement endpoints.
PCGrad projects away only the component of the local gradient that conflicts
with the global gradient. Both forwards and backwards remain evaluator-owned.
"""

from __future__ import annotations

import math
import time

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from benchmark import (
    BackwardPassContext,
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
REFINEMENT_STEPS = 4
MAX_OUTER_STEPS = 64
BASE_LR = 1.5e-3
LOCAL_GRADIENT_SCALE = 0.35


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


class ProductRippleSquare(nn.Module):
    """One learned modular square over LSD-first digit distributions."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = width
        self.columns = 2 * width - 1

        # Every (x_i, x_j) pair is represented explicitly, but its contribution
        # is learned. Grouping by i+j supplies only positional multiplication
        # structure, not a hard-coded digit product or reduction algorithm.
        self.pair_embedding = nn.Parameter(
            torch.empty(NUM_DIGITS, NUM_DIGITS, HIDDEN)
        )
        self.modulus_projection = nn.Linear(NUM_DIGITS, HIDDEN, bias=False)
        self.register_projection = nn.Linear(NUM_DIGITS, HIDDEN, bias=False)
        self.input_projection = nn.Linear(3 * HIDDEN, HIDDEN, bias=False)
        nn.init.normal_(self.pair_embedding, std=HIDDEN**-0.5)

        # A bidirectional sequential scan provides the carry/borrow substrate:
        # low-to-high propagates carries, high-to-low propagates comparison and
        # quotient information. The same scan is tied across refinement rounds.
        self.scan_norm = RMSNorm(HIDDEN)
        self.scan = nn.GRU(
            HIDDEN,
            HIDDEN,
            batch_first=True,
            bidirectional=True,
        )
        self.scan_out = nn.Linear(2 * HIDDEN, HIDDEN, bias=False)
        self.mlp_norm = RMSNorm(HIDDEN)
        self.mlp = nn.Sequential(
            nn.Linear(HIDDEN, 2 * HIDDEN, bias=False),
            nn.SiLU(),
            nn.Linear(2 * HIDDEN, HIDDEN, bias=False),
        )
        self.scan_gate = nn.Parameter(torch.full((HIDDEN,), -2.0))
        self.mlp_gate = nn.Parameter(torch.full((HIDDEN,), -2.0))
        self.output_norm = RMSNorm(HIDDEN)
        self.output = nn.Linear(HIDDEN, NUM_DIGITS)

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
        self,
        register: Tensor,
        modulus: Tensor,
        temperature: float,
    ) -> tuple[Tensor, tuple[Tensor, ...]]:
        batch = register.shape[0]
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

        phase_logits: list[Tensor] = []
        for _ in range(REFINEMENT_STEPS):
            scanned, _ = self.scan(self.scan_norm(state))
            state = state + torch.sigmoid(self.scan_gate) * self.scan_out(scanned)
            state = state + torch.sigmoid(self.mlp_gate) * self.mlp(
                self.mlp_norm(state)
            )
            phase_logits.append(self.output(self.output_norm(state[:, : self.width])))
        probabilities = F.softmax(phase_logits[-1] / temperature, dim=-1)
        return probabilities, tuple(phase_logits)


class Model(nn.Module):
    num_loops = 1

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.max_length = spec.max_seq_len
        self.width = max(2, (spec.max_seq_len - 5) // 2)
        self.square = ProductRippleSquare(self.width)
        self.training_outer_steps = 3
        self.training_temperature = 1.2
        self.eval_temperature = 0.12
        self.loss_mode = "global"

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

    def _right_aligned_register(self, input_ids: Tensor, mask: Tensor) -> Tensor:
        batch = input_ids.shape[0]
        digit_values = (input_ids - DIGIT_OFFSET).clamp(0, NUM_DIGITS - 1)
        source = F.one_hot(digit_values, NUM_DIGITS).to(torch.float32)
        source = source * mask.unsqueeze(-1)
        count = mask.sum(dim=1)
        rank = mask.long().cumsum(dim=1) - 1
        destination = (self.width - count[:, None] + rank).clamp(0, self.width - 1)
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
        occupied.scatter_add_(1, destination.unsqueeze(-1), mask.unsqueeze(-1).float())
        register[:, :, 0] = register[:, :, 0] + (1.0 - occupied.squeeze(-1))
        # Arithmetic scans are least-significant digit first.
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
        n_mask = digits & after_n & ~after_x
        x_mask = digits & after_x & ~after_t
        return (
            self._right_aligned_register(input_ids, n_mask),
            self._right_aligned_register(input_ids, x_mask),
            self._time_steps(input_ids, valid),
        )

    def _place_logits(self, digit_logits_lsd: Tensor, input_lengths: Tensor, prompt: int) -> Tensor:
        digit_logits = digit_logits_lsd.flip(1)
        destination = input_lengths[:, None] - self.width + torch.arange(
            self.width, device=digit_logits.device
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
        _, prompt_length = input_ids.shape
        if attention_mask is None:
            attention_mask = input_ids != PAD
        valid = attention_mask.bool()
        modulus, register, t_values = self._parse(input_ids, valid)
        loops = self.training_outer_steps if self.training else MAX_OUTER_STEPS
        temperature = self.training_temperature if self.training else self.eval_temperature
        first_square_phases: tuple[Tensor, ...] = ()
        for outer_step in range(loops):
            # Truncated recurrence: T=1 teaches the complete square transition;
            # later endpoints train the same tied transition without requiring
            # gradients to cross every preceding square.
            transition_input = (
                register.detach()
                if self.training and outer_step > 0
                else register
            )
            candidate, phases = self.square(
                transition_input, modulus, temperature
            )
            if outer_step == 0:
                first_square_phases = phases
            active = (t_values > outer_step)[:, None, None]
            register = torch.where(active, candidate, register)

        lengths = valid.sum(dim=1)
        logits = self._place_logits(
            register.clamp_min(1e-8).log(), lengths, prompt_length
        )
        phase_logits = tuple(
            self._place_logits(phase, lengths, prompt_length)
            for phase in first_square_phases
        )
        return logits, {
            "t_values": t_values,
            "phase_logits": phase_logits,
            "loss_mode": self.loss_mode,
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


def token_training_loss(batch: TokenLossBatch) -> Tensor:
    valid = batch.valid_mask
    t_values = batch.auxiliary["t_values"]
    row_weight = torch.where(
        t_values == 1,
        torch.full_like(t_values, 4.0, dtype=torch.float32),
        torch.ones_like(t_values, dtype=torch.float32),
    )
    token_mask = valid.float() * row_weight[:, None]
    losses = F.cross_entropy(
        batch.logits.transpose(1, 2),
        batch.labels,
        ignore_index=-100,
        reduction="none",
    )
    endpoint = (losses * token_mask).sum() / token_mask.sum().clamp_min(1.0)

    if batch.auxiliary["loss_mode"] == "global":
        return endpoint

    # On T=1, every refinement phase has the same legitimate x^2 mod N target.
    # This shortens credit assignment without inventing carry/borrow labels.
    t1_mask = (t_values == 1)[:, None] & valid
    phase_terms = [
        _masked_loss(
            _target_aligned(full_logits, batch), batch.labels, t1_mask
        )
        for full_logits in batch.auxiliary["phase_logits"]
    ]
    return torch.stack(phase_terms).mean()


class PCGradAdamW(torch.optim.AdamW):
    """Protect the global endpoint gradient from a conflicting local gradient."""

    def __init__(self, groups, model: Model, **kwargs) -> None:
        super().__init__(groups, **kwargs)
        self.model = model
        self.global_gradients: list[Tensor | None] = []

    def _parameters(self) -> list[Tensor]:
        return [
            parameter
            for group in self.param_groups
            for parameter in group["params"]
        ]

    @torch.no_grad()
    def between_backward_passes(self, context: BackwardPassContext) -> None:
        if context.pass_index != 1 or context.total_passes != 2:
            raise RuntimeError("PCGrad requires exactly two evaluator-owned passes")
        if self.model.loss_mode != "global":
            raise RuntimeError("first PCGrad pass must use the global endpoint loss")
        self.global_gradients = [
            None if parameter.grad is None else parameter.grad.detach().clone()
            for parameter in self._parameters()
        ]
        self.model.loss_mode = "local"

    @torch.no_grad()
    def step(self, closure=None):
        parameters = self._parameters()
        if len(self.global_gradients) != len(parameters):
            self.model.loss_mode = "global"
            raise RuntimeError("global endpoint gradients were not captured")

        gradient_dot = torch.zeros((), device=parameters[0].device)
        global_norm_square = torch.zeros_like(gradient_dot)
        for parameter, global_gradient in zip(parameters, self.global_gradients):
            local_gradient = parameter.grad
            if global_gradient is None:
                continue
            global_norm_square.add_(global_gradient.float().square().sum())
            if local_gradient is not None:
                gradient_dot.add_(
                    (global_gradient.float() * local_gradient.float()).sum()
                )

        # This is zero when objectives agree. When they conflict it is the
        # coefficient of the local component parallel to the global gradient.
        projection = gradient_dot.clamp_max(0.0) / global_norm_square.clamp_min(1e-12)
        for parameter, global_gradient in zip(parameters, self.global_gradients):
            local_gradient = parameter.grad
            if global_gradient is None:
                if local_gradient is not None:
                    local_gradient.mul_(LOCAL_GRADIENT_SCALE)
                continue
            combined = global_gradient.clone()
            if local_gradient is not None:
                projected_local = local_gradient - projection.to(
                    local_gradient.dtype
                ) * global_gradient
                combined.add_(projected_local, alpha=LOCAL_GRADIENT_SCALE)
            parameter.grad = combined

        try:
            return super().step(closure)
        finally:
            self.global_gradients = []
            self.model.loss_mode = "global"


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
    optimizer = PCGradAdamW(
        [
            {"params": decay, "weight_decay": 0.1, "base_lr": BASE_LR},
            {"params": no_decay, "weight_decay": 0.0, "base_lr": BASE_LR},
        ],
        model,
        lr=BASE_LR * 0.1,
        betas=(0.9, 0.95),
        eps=1e-8,
        fused=spec.device_type == "cuda",
    )
    return OptimizerBundle(
        optimizer=optimizer,
        scheduler=WallClockSchedule(optimizer, model, spec.training_time_seconds),
        backward_passes_per_step=2,
        between_backward_passes=optimizer.between_backward_passes,
    )


SUBMISSION = Submission(
    build_model=build_model,
    build_optimizer=build_optimizer,
    token_training_loss=token_training_loss,
    batch_size=32,
    eval_batch_size=256,
)
