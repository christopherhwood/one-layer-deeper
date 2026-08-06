"""T-conditioned recurrent Transformer with per-iteration vocabulary snap-back."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from benchmark import (
    ModelSpec,
    OptimizerBundle,
    OptimizerSpec,
    Submission,
    assert_model_state,
)


D_MODEL = 256
NUM_HEADS = 8
NUM_ENCODER_BLOCKS = 2
T_MARKER_ID = 4
DIGIT_OFFSET = 7
TAU_FLOOR = 0.05
TAU_INIT = 1.0


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


class Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(D_MODEL)
        self.qkv = nn.Linear(D_MODEL, 3 * D_MODEL)
        self.out = nn.Linear(D_MODEL, D_MODEL)
        self.mixer_norm = RMSNorm(D_MODEL)
        self.up = nn.Linear(D_MODEL, 4 * D_MODEL)
        self.down = nn.Linear(4 * D_MODEL, D_MODEL)

    def forward(self, x: Tensor, attention_mask: Tensor | None) -> Tensor:
        residual = x
        x = self.attention_norm(x)
        batch, length, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(batch, length, NUM_HEADS, -1).transpose(1, 2)
        k = k.view(batch, length, NUM_HEADS, -1).transpose(1, 2)
        v = v.view(batch, length, NUM_HEADS, -1).transpose(1, 2)
        mask = None
        if attention_mask is not None:
            if attention_mask.shape == (batch, length):
                mask = attention_mask[:, None, None, :]
            elif attention_mask.shape == (batch, length, length):
                mask = attention_mask[:, None, :, :]
            else:
                raise ValueError("invalid attention_mask shape")
            mask = mask.to(device=x.device, dtype=torch.bool)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = x.transpose(1, 2).contiguous().view(batch, length, D_MODEL)
        x = residual + self.out(x)
        return x + self.down(F.gelu(self.up(self.mixer_norm(x))))


class SnapBack(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm = RMSNorm(D_MODEL)
        raw_tau = torch.log(torch.expm1(torch.tensor(TAU_INIT - TAU_FLOOR)))
        self.raw_tau = nn.Parameter(raw_tau)
        self.raw_gate = nn.Parameter(torch.zeros(D_MODEL))

    def forward(self, x: Tensor, embedding: Tensor) -> Tensor:
        tau = F.softplus(self.raw_tau) + TAU_FLOOR
        logits = self.norm(x) @ embedding.t()
        probs = F.softmax(logits / tau, dim=-1)
        snapped = (probs @ embedding).to(x.dtype)
        gate = torch.sigmoid(self.raw_gate).to(x.dtype)
        return x + gate * (snapped - x)


def extract_t_values(
    input_ids: Tensor,
    attention_mask: Tensor | None = None,
) -> Tensor:
    valid = torch.ones_like(input_ids, dtype=torch.bool)
    if attention_mask is not None:
        valid = attention_mask.to(device=input_ids.device, dtype=torch.bool)
    is_marker = (input_ids == T_MARKER_ID) & valid
    has_marker = is_marker.any(dim=1)
    marker_pos = is_marker.int().argmax(dim=1)
    positions = torch.arange(input_ids.shape[1], device=input_ids.device)
    after = positions[None, :] > marker_pos[:, None]
    is_digit = (input_ids >= DIGIT_OFFSET) & valid
    stopped = torch.cummax((after & ~is_digit).int(), dim=1).values.bool()
    take = after & is_digit & ~stopped
    order = take.cumsum(dim=1)
    count = take.sum(dim=1, keepdim=True)
    place = torch.pow(10, (count - order).clamp(min=0))
    digits = (input_ids - DIGIT_OFFSET).clamp(min=0)
    values = torch.where(take, digits * place, torch.zeros_like(digits))
    t_values = values.sum(dim=1)
    return torch.where(has_marker, t_values, torch.zeros_like(t_values))


class Model(nn.Module):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.token_embedding = nn.Embedding(spec.vocab_size, D_MODEL)
        self.position_embedding = nn.Embedding(spec.max_seq_len, D_MODEL)
        self.encoder = nn.ModuleList(Block() for _ in range(NUM_ENCODER_BLOCKS))
        self.core_block = Block()
        self.snap_back = SnapBack()
        self.decoder_block = Block()
        self.final_norm = RMSNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, spec.vocab_size, bias=False)
        self.head.weight = self.token_embedding.weight

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, None]:
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        for block in self.encoder:
            x = block(x, attention_mask)
        t_values = extract_t_values(input_ids, attention_mask)
        t_max = int(t_values.max().item())
        embedding = self.token_embedding.weight
        for step in range(t_max):
            new_x = self.core_block(x, attention_mask)
            new_x = self.snap_back(new_x, embedding)
            x = torch.where((t_values > step).view(-1, 1, 1), new_x, x)
        x = self.decoder_block(x, attention_mask)
        return self.head(self.final_norm(x)), None


def build_model(spec: ModelSpec) -> Model:
    model = Model(spec)
    assert_model_state(model, spec)
    return model


def build_optimizer(model: nn.Module, spec: OptimizerSpec) -> OptimizerBundle:
    return OptimizerBundle(
        torch.optim.AdamW(
            model.parameters(),
            lr=1e-3,
            betas=(0.9, 0.95),
            weight_decay=0.1,
            capturable=spec.device_type == "cuda",
        )
    )


SUBMISSION = Submission(build_model=build_model, build_optimizer=build_optimizer)
