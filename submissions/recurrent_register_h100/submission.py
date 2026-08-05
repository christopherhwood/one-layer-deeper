"""Digit-register recurrent model for repeated modular squaring.

The learned outer cell is applied T times.  T is used only as GPU-side control
flow; the learned path sees N and the current residue, but not the T tokens.
One outer cell contains four learned refinement phases over right-aligned digit
registers.  No arithmetic transition or weight table is hard-coded.
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

WIDTH = 192
HEADS = 6
EXPANSION = 3
DIRECT_LAYERS = 2
REFINEMENT_PHASES = 4
MAX_OUTER_STEPS = 64
BASE_LR = 1.5e-3


class Config:
    def __init__(self, vocab_size: int, max_seq_len: int) -> None:
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len


class RMSNorm(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.shape[-1],), self.weight)


class DirectBlock(nn.Module):
    """Small bidirectional prompt block used as a statistical residual path."""

    def __init__(self) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(WIDTH)
        self.qkv = nn.Linear(WIDTH, 3 * WIDTH, bias=False)
        self.attn_out = nn.Linear(WIDTH, WIDTH, bias=False)
        self.mlp_norm = RMSNorm(WIDTH)
        self.up = nn.Linear(WIDTH, 2 * EXPANSION * WIDTH, bias=False)
        self.down = nn.Linear(EXPANSION * WIDTH, WIDTH, bias=False)

    def forward(self, x: Tensor, valid: Tensor) -> Tensor:
        batch, length, _ = x.shape
        h = self.attn_norm(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q = q.view(batch, length, HEADS, -1).transpose(1, 2)
        k = k.view(batch, length, HEADS, -1).transpose(1, 2)
        v = v.view(batch, length, HEADS, -1).transpose(1, 2)
        mask = valid[:, None, None, :]
        mixed = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        mixed = mixed.transpose(1, 2).contiguous().view(batch, length, WIDTH)
        x = x + self.attn_out(mixed)
        left, right = self.up(self.mlp_norm(x)).chunk(2, dim=-1)
        return x + self.down(F.silu(left) * right)


class RegisterPhase(nn.Module):
    """One learned local/global refinement of a decimal residue register."""

    def __init__(self) -> None:
        super().__init__()
        self.state_norm = RMSNorm(WIDTH)
        self.register_in = nn.Linear(WIDTH, WIDTH, bias=False)
        self.modulus_in = nn.Linear(WIDTH, WIDTH, bias=False)
        self.local = nn.Conv1d(
            WIDTH, WIDTH, kernel_size=3, padding=1, groups=WIDTH, bias=False
        )
        self.qkv = nn.Linear(WIDTH, 3 * WIDTH, bias=False)
        self.attn_out = nn.Linear(WIDTH, WIDTH, bias=False)
        self.mlp_norm = RMSNorm(WIDTH)
        self.up = nn.Linear(WIDTH, 2 * EXPANSION * WIDTH, bias=False)
        self.down = nn.Linear(EXPANSION * WIDTH, WIDTH, bias=False)
        self.digit_norm = RMSNorm(WIDTH)
        self.digit_head = nn.Linear(WIDTH, NUM_DIGITS)
        self.replace_gate = nn.Linear(WIDTH, 1)
        nn.init.constant_(self.replace_gate.bias, -0.5)

    def forward(
        self,
        hidden: Tensor,
        register_embedding: Tensor,
        modulus_embedding: Tensor,
        temperature: float,
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch, cells, _ = hidden.shape
        source = self.state_norm(hidden)
        source = (
            source
            + self.register_in(register_embedding)
            + self.modulus_in(modulus_embedding)
        )
        local = self.local(source.transpose(1, 2)).transpose(1, 2)
        q, k, v = self.qkv(source).chunk(3, dim=-1)
        q = q.view(batch, cells, HEADS, -1).transpose(1, 2)
        k = k.view(batch, cells, HEADS, -1).transpose(1, 2)
        v = v.view(batch, cells, HEADS, -1).transpose(1, 2)
        global_mix = F.scaled_dot_product_attention(q, k, v)
        global_mix = global_mix.transpose(1, 2).contiguous().view(
            batch, cells, WIDTH
        )
        hidden = hidden + local + self.attn_out(global_mix)
        left, right = self.up(self.mlp_norm(hidden)).chunk(2, dim=-1)
        hidden = hidden + self.down(F.silu(left) * right)
        digit_logits = self.digit_head(self.digit_norm(hidden)).float()
        candidate = F.softmax(digit_logits / temperature, dim=-1)
        gate = torch.sigmoid(self.replace_gate(hidden).float())
        return hidden, candidate, gate


class SquaringCell(nn.Module):
    """A complete learned residue transition, tied over outer T steps."""

    def __init__(self) -> None:
        super().__init__()
        self.residue_digits = nn.Parameter(torch.empty(NUM_DIGITS, WIDTH))
        self.modulus_digits = nn.Parameter(torch.empty(NUM_DIGITS, WIDTH))
        nn.init.normal_(self.residue_digits, std=WIDTH**-0.5)
        nn.init.normal_(self.modulus_digits, std=WIDTH**-0.5)
        self.initial = nn.Linear(2 * WIDTH, WIDTH, bias=False)
        self.phases = nn.ModuleList(RegisterPhase() for _ in range(REFINEMENT_PHASES))

    def forward(
        self,
        register: Tensor,
        modulus: Tensor,
        place_encoding: Tensor,
        temperature: float,
    ) -> tuple[Tensor, tuple[Tensor, ...]]:
        residue_embedding = register.to(self.residue_digits.dtype) @ self.residue_digits
        modulus_embedding = modulus.to(self.modulus_digits.dtype) @ self.modulus_digits
        hidden = self.initial(torch.cat((residue_embedding, modulus_embedding), dim=-1))
        hidden = hidden + place_encoding
        phase_probabilities: list[Tensor] = []
        for phase in self.phases:
            residue_embedding = register.to(self.residue_digits.dtype) @ self.residue_digits
            hidden, candidate, gate = phase(
                hidden,
                residue_embedding,
                modulus_embedding,
                temperature,
            )
            register = (1.0 - gate) * register + gate * candidate
            register = register / register.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            phase_probabilities.append(register)
        return register, tuple(phase_probabilities)


def _sinusoidal_places(cells: int, width: int) -> Tensor:
    """Length-independent place features, indexed from the least-significant digit."""
    places = torch.arange(cells - 1, -1, -1, dtype=torch.float32).unsqueeze(1)
    half = width // 2
    scale = torch.exp(
        torch.arange(half, dtype=torch.float32)
        * (-math.log(10_000.0) / max(half - 1, 1))
    )
    angles = places * scale.unsqueeze(0)
    result = torch.cat((angles.sin(), angles.cos()), dim=1)
    if result.shape[1] < width:
        result = F.pad(result, (0, width - result.shape[1]))
    return result


class Model(nn.Module):
    num_loops = REFINEMENT_PHASES

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.cells = spec.max_seq_len
        self.token_embedding = nn.Embedding(spec.vocab_size, WIDTH)
        self.prompt_position = nn.Embedding(spec.max_seq_len, WIDTH)
        self.direct_blocks = nn.ModuleList(
            DirectBlock() for _ in range(DIRECT_LAYERS)
        )
        self.direct_norm = RMSNorm(WIDTH)
        self.direct_head = nn.Linear(WIDTH, spec.vocab_size, bias=False)
        self.direct_head.weight = self.token_embedding.weight
        self.cell = SquaringCell()
        self.raw_direct_scale = nn.Parameter(torch.tensor(-1.0))
        self.raw_register_scale = nn.Parameter(torch.tensor(0.5))
        self.register_buffer(
            "place_encoding",
            _sinusoidal_places(spec.max_seq_len, WIDTH),
            persistent=True,
        )
        self.training_outer_steps = 4
        self.training_temperature = 1.25
        self.eval_temperature = 0.18

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

    def _right_aligned_register(self, input_ids: Tensor, field_mask: Tensor) -> Tensor:
        batch, _ = input_ids.shape
        digits = (input_ids - DIGIT_OFFSET).clamp(min=0, max=NUM_DIGITS - 1)
        source = F.one_hot(digits, NUM_DIGITS).float() * field_mask.unsqueeze(-1)
        count = field_mask.sum(dim=1)
        rank = field_mask.long().cumsum(dim=1) - 1
        destination = (self.cells - count[:, None] + rank).clamp(
            min=0, max=self.cells - 1
        )
        register = torch.zeros(
            batch,
            self.cells,
            NUM_DIGITS,
            device=input_ids.device,
            dtype=torch.float32,
        )
        occupied = torch.zeros(
            batch, self.cells, 1, device=input_ids.device, dtype=torch.float32
        )
        register.scatter_add_(
            1, destination.unsqueeze(-1).expand(-1, -1, NUM_DIGITS), source
        )
        occupied.scatter_add_(
            1, destination.unsqueeze(-1), field_mask.unsqueeze(-1).float()
        )
        register[:, :, 0] = register[:, :, 0] + (1.0 - occupied.squeeze(-1))
        return register

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

    def _register_vocab_logits(self, probabilities: Tensor, prompt_length: int) -> Tensor:
        digit_logits = probabilities.clamp_min(1e-8).log()
        vocab_logits = F.pad(
            digit_logits,
            (DIGIT_OFFSET, self.config.vocab_size - DIGIT_OFFSET - NUM_DIGITS),
            value=-16.0,
        )
        return vocab_logits[:, self.cells - prompt_length :]

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, object]]:
        batch, prompt_length = input_ids.shape
        if prompt_length > self.cells:
            raise ValueError("input is longer than model.config.max_seq_len")
        valid = (
            attention_mask.bool()
            if attention_mask is not None
            else input_ids != PAD
        )

        positions = torch.arange(prompt_length, device=input_ids.device)
        direct = self.token_embedding(input_ids) + self.prompt_position(positions)
        for block in self.direct_blocks:
            direct = block(direct, valid)
        direct_logits = self.direct_head(self.direct_norm(direct))

        modulus, register, t_values = self._parse(input_ids, valid)
        outer_steps = self.training_outer_steps if self.training else MAX_OUTER_STEPS
        temperature = self.training_temperature if self.training else self.eval_temperature
        first_phase_probabilities: tuple[Tensor, ...] = ()
        for outer_step in range(outer_steps):
            candidate, phase_probabilities = self.cell(
                register,
                modulus,
                self.place_encoding.to(register.dtype),
                temperature,
            )
            if outer_step == 0:
                first_phase_probabilities = phase_probabilities
            active = (t_values > outer_step)[:, None, None]
            register = torch.where(active, candidate, register)

        register_logits = self._register_vocab_logits(register, prompt_length)
        register_scale = F.softplus(self.raw_register_scale)
        direct_scale = torch.sigmoid(self.raw_direct_scale)
        logits = register_scale * register_logits + direct_scale * direct_logits
        phase_logits = tuple(
            self._register_vocab_logits(probabilities, prompt_length)
            for probabilities in first_phase_probabilities
        )
        auxiliary: dict[str, object] = {
            "t_values": t_values,
            "register_logits": register_logits,
            "phase_logits": phase_logits,
            "training_outer_steps": outer_steps,
        }
        return logits, auxiliary


def _target_aligned(full_logits: Tensor, batch: TokenLossBatch) -> Tensor:
    if batch.target_positions is None:
        return full_logits[:, : batch.logits.shape[1]]
    batch_indices = torch.arange(full_logits.shape[0], device=full_logits.device)[:, None]
    return full_logits[batch_indices, batch.target_positions.clamp_min(0)]


def _masked_token_loss(logits: Tensor, labels: Tensor, mask: Tensor) -> Tensor:
    losses = F.cross_entropy(
        logits.transpose(1, 2), labels, ignore_index=-100, reduction="none"
    )
    weights = mask.to(losses.dtype)
    return (losses * weights).sum() / weights.sum().clamp_min(1.0)


def token_training_loss(batch: TokenLossBatch) -> Tensor:
    valid = batch.valid_mask
    token_losses = F.cross_entropy(
        batch.logits.transpose(1, 2),
        batch.labels,
        ignore_index=-100,
        reduction="none",
    )
    counts = valid.sum(dim=1).clamp_min(1)
    token_term = token_losses[valid].mean()
    sequence_term = (
        (token_losses * valid).sum(dim=1) / counts.float().mean()
    ).mean()
    loss = 0.65 * token_term + 0.35 * sequence_term

    auxiliary = batch.auxiliary
    if not isinstance(auxiliary, dict):
        return loss
    t_values = auxiliary["t_values"]
    trained_outer = int(auxiliary["training_outer_steps"])
    eligible = (t_values <= trained_outer)[:, None] & valid
    register_logits = _target_aligned(auxiliary["register_logits"], batch)
    loss = loss + 0.35 * _masked_token_loss(
        register_logits, batch.labels, eligible
    )

    t1_mask = (t_values == 1)[:, None] & valid
    phase_terms = []
    for full_logits in auxiliary["phase_logits"]:
        phase_terms.append(
            _masked_token_loss(
                _target_aligned(full_logits, batch), batch.labels, t1_mask
            )
        )
    if phase_terms:
        loss = loss + 0.15 * torch.stack(phase_terms).mean()
    return loss


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
        if fraction < 0.04:
            lr_multiplier = 0.1 + 0.9 * fraction / 0.04
        elif fraction < 0.82:
            lr_multiplier = 1.0
        else:
            progress = (fraction - 0.82) / 0.18
            lr_multiplier = 0.08 + 0.92 * 0.5 * (
                1.0 + math.cos(math.pi * progress)
            )
        for group in self.optimizer.param_groups:
            group["lr"] = group["base_lr"] * lr_multiplier
        self.model.training_temperature = max(
            0.22, 1.25 * (0.22 / 1.25) ** fraction
        )


def build_model(spec: ModelSpec) -> Model:
    model = Model(spec)
    assert_model_state(model, spec)
    return model


def build_optimizer(model: Model, spec: OptimizerSpec) -> OptimizerBundle:
    if spec.training_time_seconds <= 90:
        model.training_outer_steps = 4
    else:
        model.training_outer_steps = 16
    decay: list[Tensor] = []
    no_decay: list[Tensor] = []
    for parameter in model.parameters():
        (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {
                "params": decay,
                "weight_decay": 0.05,
                "base_lr": BASE_LR,
                "lr": BASE_LR * 0.1,
            },
            {
                "params": no_decay,
                "weight_decay": 0.0,
                "base_lr": BASE_LR,
                "lr": BASE_LR * 0.1,
            },
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
    eval_batch_size=512,
)
