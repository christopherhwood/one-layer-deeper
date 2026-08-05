"""General weight-tied scratchpad reasoner for repeated modular squaring.

The model is deliberately problem-agnostic beyond reading the public T field:
it learns a recurrent transition over a prompt plus scratch tape, hides T from
that transition, and applies the same weights the requested number of times.
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


WIDTH = 128
HEADS = 4
EXPANSION = 4
BASE_LR = 8.0e-4


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


class ReasoningBlock(nn.Module):
    """Pre-norm Transformer block stabilized for repeated weight tying."""

    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.width = width
        self.heads = heads
        self.head_width = width // heads
        self.attention_norm = RMSNorm(width)
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.attention_out = nn.Linear(width, width, bias=False)
        self.mlp_norm = RMSNorm(width)
        self.up = nn.Linear(width, 2 * EXPANSION * width, bias=False)
        self.down = nn.Linear(EXPANSION * width, width, bias=False)
        self.attention_gate = nn.Parameter(torch.empty(width))
        self.mlp_gate = nn.Parameter(torch.empty(width))
        nn.init.normal_(self.attention_gate, mean=-2.0, std=0.02)
        nn.init.normal_(self.mlp_gate, mean=-2.0, std=0.02)

    def forward(self, x: Tensor, key_mask: Tensor) -> Tensor:
        batch, length, _ = x.shape
        normalized = self.attention_norm(x)
        q, k, v = self.qkv(normalized).chunk(3, dim=-1)
        q = q.view(batch, length, self.heads, self.head_width).transpose(1, 2)
        k = k.view(batch, length, self.heads, self.head_width).transpose(1, 2)
        v = v.view(batch, length, self.heads, self.head_width).transpose(1, 2)
        q = F.rms_norm(q, (self.head_width,))
        k = F.rms_norm(k, (self.head_width,))
        attended = F.scaled_dot_product_attention(
            q, k, v, attn_mask=key_mask, scale=self.head_width**-0.5
        )
        attended = attended.transpose(1, 2).contiguous().view(batch, length, self.width)
        x = x + torch.sigmoid(self.attention_gate) * self.attention_out(attended)
        left, right = self.up(self.mlp_norm(x)).chunk(2, dim=-1)
        mixed = F.silu(left) * right
        return x + torch.sigmoid(self.mlp_gate) * self.down(mixed)


class Model(nn.Module):
    num_loops = 4

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.max_length = spec.max_seq_len
        self.token_embedding = nn.Embedding(spec.vocab_size, WIDTH)
        self.position_embedding = nn.Parameter(torch.empty(2 * spec.max_seq_len, WIDTH))
        self.type_embedding = nn.Parameter(torch.empty(2, WIDTH))
        self.scratch = nn.Parameter(torch.empty(spec.max_seq_len, WIDTH))
        nn.init.normal_(self.position_embedding, std=WIDTH**-0.5)
        nn.init.normal_(self.type_embedding, std=WIDTH**-0.5)
        nn.init.normal_(self.scratch, std=WIDTH**-0.5)
        self.encoder = ReasoningBlock(WIDTH, HEADS)
        self.reasoning_a = ReasoningBlock(WIDTH, HEADS)
        self.reasoning_b = ReasoningBlock(WIDTH, HEADS)
        self.decoder = ReasoningBlock(WIDTH, HEADS)
        self.final_norm = RMSNorm(WIDTH)
        self.head = nn.Linear(WIDTH, spec.vocab_size, bias=False)
        self.head.weight = self.token_embedding.weight
        self.active_train_loops = 4
        self.maximum_train_loops = 16
        self.eval_loops = 64

    @staticmethod
    def _requested_steps(input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        value = torch.zeros(input_ids.shape[0], device=input_ids.device, dtype=torch.long)
        after_marker = torch.zeros_like(value, dtype=torch.bool)
        for position in range(input_ids.shape[1]):
            token = input_ids[:, position]
            after_marker = after_marker | ((token == 4) & attention_mask[:, position])
            is_digit = after_marker & (token >= 7) & (token <= 16)
            value = torch.where(is_digit, value * 10 + token - 7, value)
        return value.clamp(min=0, max=64)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, None]:
        batch, prompt_length = input_ids.shape
        if prompt_length > self.max_length:
            raise ValueError("input is longer than model.config.max_seq_len")
        if attention_mask is None:
            attention_mask = input_ids != 0
        attention_mask = attention_mask.bool()
        requested_steps = self._requested_steps(input_ids, attention_mask)

        # T is a compute-control signal only. The learned state transition never
        # sees it, which prevents separate lookup functions for training depths.
        time_field = (input_ids == 4).cumsum(dim=1) > 0
        reasoning_mask = attention_mask & ~time_field
        reasoning_ids = torch.where(reasoning_mask, input_ids, 0)
        prompt = self.token_embedding(reasoning_ids)
        if prompt_length < self.max_length:
            prompt = F.pad(prompt, (0, 0, 0, self.max_length - prompt_length))
            reasoning_mask = F.pad(
                reasoning_mask, (0, self.max_length - prompt_length), value=False
            )
        prompt = prompt + self.position_embedding[: self.max_length] + self.type_embedding[0]
        scratch = (
            self.scratch
            + self.position_embedding[self.max_length :]
            + self.type_embedding[1]
        ).unsqueeze(0).expand(batch, -1, -1)
        initial = torch.cat((prompt, scratch), dim=1)
        scratch_mask = torch.ones(
            (batch, self.max_length), device=input_ids.device, dtype=torch.bool
        )
        key_mask = torch.cat((reasoning_mask, scratch_mask), dim=1)[:, None, None, :]

        encoded = self.encoder(initial, key_mask)
        context = encoded[:, : self.max_length]
        work = encoded[:, self.max_length :]
        loops = self.active_train_loops if self.training else self.eval_loops
        for loop in range(loops):
            state = torch.cat((context, work), dim=1)
            candidate = self.reasoning_b(self.reasoning_a(state, key_mask), key_mask)
            active = (requested_steps > loop)[:, None, None]
            work = torch.where(active, candidate[:, self.max_length :], work)

        decoded = self.decoder(torch.cat((context, work), dim=1), key_mask)
        logits = self.head(self.final_norm(decoded[:, :prompt_length]))
        return logits, None


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
        self.budget_seconds = max(float(budget_seconds) * 0.97, 1.0)
        self.base_lrs = [BASE_LR for _ in optimizer.param_groups]

    def step(self) -> None:
        fraction = min((time.monotonic() - self.started_at) / self.budget_seconds, 1.0)
        if fraction < 0.03:
            multiplier = 0.05 + 0.95 * fraction / 0.03
        elif fraction < 0.90:
            multiplier = 1.0
        else:
            progress = (fraction - 0.90) / 0.10
            multiplier = 0.1 + 0.45 * (1.0 + math.cos(math.pi * progress))
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = base_lr * multiplier
        target = self.model.maximum_train_loops
        if fraction < 0.08:
            loops = min(4, target)
        elif fraction < 0.22:
            loops = min(8, target)
        elif fraction < 0.45:
            loops = min(12, target)
        else:
            loops = target
        self.model.active_train_loops = loops
        self.model.num_loops = loops


def _stablemax_cross_entropy(logits: Tensor, labels: Tensor) -> Tensor:
    """StableMax CE with both branches defined over their full input domain."""

    positive = torch.log1p(logits.clamp_min(0))
    negative = -torch.log1p((-logits).clamp_min(0))
    transformed = torch.where(logits >= 0, positive, negative)
    return F.cross_entropy(
        transformed.transpose(1, 2),
        labels,
        ignore_index=-100,
        reduction="none",
    )


def token_training_loss(batch: TokenLossBatch) -> Tensor:
    valid = batch.valid_mask
    losses = _stablemax_cross_entropy(batch.logits, batch.labels)
    counts = valid.sum(dim=1).clamp_min(1)
    token_term = losses[valid].mean()
    sequence_term = (losses.sum(dim=1) / counts.float().mean()).mean()
    return 0.7 * token_term + 0.3 * sequence_term


def build_model(spec: ModelSpec) -> Model:
    model = Model(spec)
    assert_model_state(model, spec)
    return model


def build_optimizer(model: Model, spec: OptimizerSpec) -> OptimizerBundle:
    model.maximum_train_loops = 8 if spec.training_time_seconds <= 90 else 16
    decay: list[Tensor] = []
    no_decay: list[Tensor] = []
    for parameter in model.parameters():
        (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    # Keep the H100 path on PyTorch's well-tested fused implementation. The
    # prior custom gradient rewrite around fused AdamW was CUDA-only risk with
    # no CPU smoke coverage and no demonstrated score benefit.
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": 1.0},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=BASE_LR * 0.05,
        betas=(0.9, 0.95),
        eps=1e-8,
        fused=spec.device_type == "cuda",
    )
    schedule = WallClockSchedule(optimizer, model, spec.training_time_seconds)
    return OptimizerBundle(optimizer=optimizer, scheduler=schedule)


SUBMISSION = Submission(
    build_model=build_model,
    build_optimizer=build_optimizer,
    token_training_loss=token_training_loss,
    batch_size=32,
    eval_batch_size=512,
)
