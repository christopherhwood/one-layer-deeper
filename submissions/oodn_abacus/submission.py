"""Length-general modular-arithmetic model for OOD-N (abacus + looped core).

Goal: compute x^(2^T) mod N as a FUNCTION OF THE DIGITS, so it generalizes to
moduli N never seen in training (the OOD-N metric), rather than memorizing a
table keyed on specific x or N.

Design (see the three ingredients):
1. ABACUS embeddings -- each digit is tagged with its place value *within its own
   number* (units=0, tens=1, ...) via the N/X/T marker tokens, plus a segment id
   (digit-of-N / -x / -T). NO absolute sequence positions. This makes the
   representation relative, the known trick for length/size generalization of
   learned arithmetic.
2. LOOPED weight-tied transformer core -- one bidirectional block applied K times.
   Attention gives the GLOBAL magnitude comparison modular reduction needs
   ("running value >= N?"), which a local conv cannot; the loop propagates
   carry/borrow across digit positions.
3. PLACE-VALUE decoder -- each output position predicts a digit conditioned on its
   distance from the prompt end (its output place value), so the answer is
   produced length-generally and tail-aligned as the evaluator expects.

Throughput: small batch_size (the H100 manifests respawn DataLoader workers on
tiny datasets at large batch), fixed loop count (no per-step device syncs).
Rules: fully learned end-to-end; no loaded weights; no hard-coded arithmetic.
"""
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

PAD, BOS, N_MARK, X_MARK, T_MARK, ANS, EOS = 0, 1, 2, 3, 4, 5, 6
DIGIT_OFFSET = 7

D_MODEL = 256
NUM_HEADS = 4
LOOPS = 16          # looped-core depth (weight-tied), fixed -> no device syncs
MAX_PLACE = 64      # max digit-significance / output place index


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
    """Bidirectional pre-norm attention + MLP block (padding-masked)."""

    def __init__(self) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(D_MODEL)
        self.qkv = nn.Linear(D_MODEL, 3 * D_MODEL)
        self.out = nn.Linear(D_MODEL, D_MODEL)
        self.mixer_norm = RMSNorm(D_MODEL)
        self.up = nn.Linear(D_MODEL, 4 * D_MODEL)
        self.down = nn.Linear(4 * D_MODEL, D_MODEL)

    def forward(self, x: Tensor, key_padding: Tensor) -> Tensor:
        h = self.attention_norm(x)
        batch, length, _ = h.shape
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q = q.view(batch, length, NUM_HEADS, -1).transpose(1, 2)
        k = k.view(batch, length, NUM_HEADS, -1).transpose(1, 2)
        v = v.view(batch, length, NUM_HEADS, -1).transpose(1, 2)
        mask = key_padding[:, None, None, :].to(dtype=torch.bool)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        a = a.transpose(1, 2).contiguous().view(batch, length, D_MODEL)
        x = x + self.out(a)
        return x + self.down(F.gelu(self.up(self.mixer_norm(x))))


def _significance_and_segment(input_ids: Tensor) -> tuple[Tensor, Tensor]:
    """Abacus features, fully vectorized (loops over sequence length only).

    significance[b,p] = place value of a digit within its own number (units=0),
                        else 0.
    segment[b,p]      = 1/2/3 for a digit belonging to N / x / T, else 0.
    """
    batch, length = input_ids.shape
    device = input_ids.device
    is_digit = (input_ids >= DIGIT_OFFSET) & (input_ids < DIGIT_OFFSET + 10)

    # significance: reverse scan; digits are MSB-first so units digit (place 0)
    # is the last digit of a run.
    significance = torch.zeros(batch, length, dtype=torch.long, device=device)
    prev = torch.zeros(batch, dtype=torch.long, device=device)
    prev_digit = torch.zeros(batch, dtype=torch.bool, device=device)
    for p in range(length - 1, -1, -1):
        d = is_digit[:, p]
        sig = torch.where(prev_digit, prev + 1, torch.zeros_like(prev))
        significance[:, p] = torch.where(d, sig, torch.zeros_like(sig))
        prev = torch.where(d, significance[:, p], torch.zeros_like(prev))
        prev_digit = d

    # segment id: most recent marker (N=2->1, X=3->2, T=4->3) at/left of p.
    seg = torch.zeros(batch, length, dtype=torch.long, device=device)
    cur = torch.zeros(batch, dtype=torch.long, device=device)
    for p in range(length):
        tok = input_ids[:, p]
        cur = torch.where(tok == N_MARK, torch.full_like(cur, 1), cur)
        cur = torch.where(tok == X_MARK, torch.full_like(cur, 2), cur)
        cur = torch.where(tok == T_MARK, torch.full_like(cur, 3), cur)
        seg[:, p] = torch.where(is_digit[:, p], cur, torch.zeros_like(cur))
    return significance.clamp(max=MAX_PLACE - 1), seg


class Model(nn.Module):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.token_embedding = nn.Embedding(spec.vocab_size, D_MODEL)
        self.sig_embedding = nn.Embedding(MAX_PLACE, D_MODEL)     # abacus place value
        self.seg_embedding = nn.Embedding(4, D_MODEL)            # which number
        self.block = Block()                                    # looped (tied)
        self.final_norm = RMSNorm(D_MODEL)
        self.place_embedding = nn.Embedding(MAX_PLACE, D_MODEL)  # output place value
        self.decoder_norm = RMSNorm(D_MODEL)
        self.decoder = nn.Sequential(
            nn.Linear(2 * D_MODEL, 4 * D_MODEL), nn.GELU(),
            nn.Linear(4 * D_MODEL, D_MODEL), nn.GELU(),
        )
        self.head = nn.Linear(D_MODEL, spec.vocab_size, bias=False)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, None]:
        batch, length = input_ids.shape
        device = input_ids.device
        key_padding = (
            attention_mask.bool() if attention_mask is not None else input_ids != PAD
        )
        significance, segment = _significance_and_segment(input_ids)
        x = (
            self.token_embedding(input_ids)
            + self.sig_embedding(significance)
            + self.seg_embedding(segment)
        )
        for _ in range(LOOPS):                      # looped weight-tied core
            x = self.block(x, key_padding)
        x = self.final_norm(x)

        # place-value decode, tail-aligned to each row's true prompt end.
        lengths = key_padding.long().sum(1)
        idx = torch.arange(length, device=device).unsqueeze(0)
        place = (lengths.unsqueeze(1) - 1 - idx).clamp(0, MAX_PLACE - 1)
        decoded = self.decoder(
            torch.cat([self.decoder_norm(x), self.place_embedding(place)], dim=-1)
        )
        return self.head(decoded), None


def build_model(spec: ModelSpec) -> Model:
    model = Model(spec)
    assert_model_state(model, spec)
    return model


def build_optimizer(model: nn.Module, spec: OptimizerSpec) -> OptimizerBundle:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.05,
        capturable=spec.device_type == "cuda",
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20_000)
    return OptimizerBundle(optimizer, scheduler=scheduler)


SUBMISSION = Submission(
    build_model=build_model,
    build_optimizer=build_optimizer,
    batch_size=32,          # small -> many batches/epoch -> ~10x more H100 steps
    eval_batch_size=256,
)
