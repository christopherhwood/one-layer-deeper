"""Endpoint-clamped global program lattice for repeated modular squaring.

One tied product-aware reducer exposes six possible stopping depths.  They are
not supervised as intermediate answers.  Instead, each depth defines a
normalized CRF over observable output digits, and the endpoint loss exactly
marginalizes a single input-independent posterior over the six complete
programs.  Evaluation extracts one global MAP depth.  Thus no per-example
latent program or unnamed arithmetic state can become a lookup channel.
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
ATTENTION_HEADS = 4
PROGRAM_DEPTHS = 4
MAX_OUTER_STEPS = 64
TRAIN_OUTER_STEPS = 3
BASE_LR = 1.8e-3
LATTICE_WEIGHT = 0.65
PROGRAM_ENTROPY_WEIGHT = 0.025
MDL_DEPTH_COST = 0.0


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


class ProductProgramSquare(nn.Module):
    """One square with an observable-digit CRF at every tied stopping depth."""

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

        self.attention_norm = RMSNorm(HIDDEN)
        self.qkv = nn.Linear(HIDDEN, 3 * HIDDEN, bias=False)
        self.attention_out = nn.Linear(HIDDEN, HIDDEN, bias=False)
        query_position = torch.arange(self.columns)[:, None]
        key_position = torch.arange(self.columns)[None, :]
        self.register_buffer(
            "attention_mask",
            torch.stack(
                (
                    torch.ones(
                        self.columns, self.columns, dtype=torch.bool
                    ),
                    (query_position - key_position).abs() <= 1,
                    key_position <= query_position,
                    key_position >= query_position,
                ),
                dim=0,
            )[None],
            persistent=False,
        )
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
        self.edge_norm = RMSNorm(HIDDEN)
        self.edge_head = nn.Linear(
            2 * HIDDEN, NUM_DIGITS * NUM_DIGITS
        )
        self.start_scores = nn.Parameter(torch.empty(NUM_DIGITS))
        self.end_scores = nn.Parameter(torch.empty(NUM_DIGITS))
        nn.init.normal_(self.start_scores, std=0.02)
        nn.init.normal_(self.end_scores, std=0.02)

    def _product_columns(self, register: Tensor) -> Tensor:
        batch = register.shape[0]
        pair_features = torch.einsum(
            "bid,bje,deh->bijh", register, register, self.pair_embedding
        )
        position = torch.arange(self.width, device=register.device)
        significance = (position[:, None] + position[None, :]).reshape(
            1, self.width * self.width, 1
        )
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
    ) -> tuple[
        tuple[Tensor, ...],
        tuple[tuple[Tensor, Tensor, Tensor, Tensor], ...],
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

        probabilities: list[Tensor] = []
        crfs: list[tuple[Tensor, Tensor, Tensor, Tensor]] = []
        for _ in range(PROGRAM_DEPTHS):
            normalized = self.attention_norm(state)
            qkv = self.qkv(normalized).reshape(
                state.shape[0],
                self.columns,
                3,
                ATTENTION_HEADS,
                HIDDEN // ATTENTION_HEADS,
            ).permute(2, 0, 3, 1, 4)
            query, key, value = qkv.unbind(dim=0)
            attended = F.scaled_dot_product_attention(
                query, key, value, attn_mask=self.attention_mask
            )
            attended = attended.transpose(1, 2).reshape(
                state.shape[0], self.columns, HIDDEN
            )
            state = state + torch.sigmoid(
                self.scan_gate
            ) * self.attention_out(attended)
            state = state + torch.sigmoid(self.mlp_gate) * self.mlp(
                self.mlp_norm(state)
            )

            digit_logits = self.output(
                self.output_norm(state[:, : self.width])
            )
            probabilities.append(
                F.softmax(digit_logits / temperature, dim=-1)
            )
            output_state = self.edge_norm(state[:, : self.width]).flip(1)
            adjacent = torch.cat(
                (output_state[:, :-1], output_state[:, 1:]), dim=-1
            )
            edges = self.edge_head(adjacent).reshape(
                register.shape[0],
                self.width - 1,
                NUM_DIGITS,
                NUM_DIGITS,
            )
            crfs.append(
                (
                    digit_logits.flip(1),
                    edges,
                    self.start_scores,
                    self.end_scores,
                )
            )
        return tuple(probabilities), tuple(crfs)


class Model(nn.Module):
    num_loops = PROGRAM_DEPTHS

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.max_length = spec.max_seq_len
        self.width = max(2, (spec.max_seq_len - 5) // 2)
        self.square = ProductProgramSquare(self.width)
        self.program_logits = nn.Parameter(torch.zeros(PROGRAM_DEPTHS))
        self.register_buffer(
            "program_cost",
            MDL_DEPTH_COST * torch.arange(PROGRAM_DEPTHS).float(),
            persistent=False,
        )
        self.training_temperature = 1.2
        self.program_temperature = 1.5
        self.program_entropy_scale = 0.0

    def _program_log_weights(self) -> Tensor:
        temperature = self.program_temperature if self.training else 0.1
        return F.log_softmax(
            (self.program_logits - self.program_cost) / temperature,
            dim=0,
        )

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
            1,
            destination.unsqueeze(-1).expand(-1, -1, NUM_DIGITS),
            source,
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
            (
                DIGIT_OFFSET,
                self.config.vocab_size - DIGIT_OFFSET - NUM_DIGITS,
            ),
            value=-16.0,
        )

    @staticmethod
    def _viterbi(
        unary: Tensor, edges: Tensor, start: Tensor, end: Tensor
    ) -> Tensor:
        score = unary[:, 0] + start
        backpointers: list[Tensor] = []
        for position in range(1, unary.shape[1]):
            candidates = score[:, :, None] + edges[:, position - 1]
            score, previous = candidates.max(dim=1)
            score = score + unary[:, position]
            backpointers.append(previous)
        current = (score + end).argmax(dim=-1)
        path = [current]
        for previous in reversed(backpointers):
            current = previous.gather(1, current[:, None]).squeeze(1)
            path.append(current)
        return torch.stack(tuple(reversed(path)), dim=1)

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
        loops = (
            TRAIN_OUTER_STEPS
            if self.training
            else int(t_values.max().item())
        )
        log_weights = self._program_log_weights()
        weights = log_weights.exp()
        temperature = self.training_temperature if self.training else 0.12

        terminal_crfs: list[tuple[Tensor, Tensor, Tensor, Tensor]] | None = None
        for outer_step in range(loops):
            transition_input = (
                register.detach()
                if self.training and outer_step > 0
                else register
            )
            phase_probabilities, phase_crfs = self.square(
                transition_input, modulus, temperature
            )
            candidate = torch.stack(phase_probabilities, dim=0).mul(
                weights[:, None, None, None]
            ).sum(dim=0)
            if terminal_crfs is None:
                terminal_crfs = []
                for unary, edges, start, end in phase_crfs:
                    terminal_crfs.append(
                        (
                            torch.zeros_like(unary),
                            torch.zeros_like(edges),
                            start,
                            end,
                        )
                    )
            terminal = t_values == outer_step + 1
            for depth, crf in enumerate(phase_crfs):
                prior = terminal_crfs[depth]
                terminal_crfs[depth] = (
                    torch.where(terminal[:, None, None], crf[0], prior[0]),
                    torch.where(
                        terminal[:, None, None, None], crf[1], prior[1]
                    ),
                    crf[2],
                    crf[3],
                )
            active = (t_values > outer_step)[:, None, None]
            register = torch.where(active, candidate, register)

        if terminal_crfs is None:
            raise RuntimeError("at least one outer square is required")
        if self.training:
            digit_logits_lsd = register.clamp_min(1e-8).log()
        else:
            depth = int(log_weights.argmax().item())
            path = self._viterbi(*terminal_crfs[depth])
            path_logits = torch.full(
                (*path.shape, NUM_DIGITS),
                -16.0,
                device=path.device,
                dtype=register.dtype,
            )
            path_logits.scatter_(2, path.unsqueeze(-1), 0.0)
            digit_logits_lsd = path_logits.flip(1)

        lengths = valid.sum(dim=1)
        logits = self._place_logits(
            digit_logits_lsd, lengths, prompt_length
        )
        return logits, {
            "t_values": t_values,
            "program_crfs": tuple(terminal_crfs),
            "program_log_weights": log_weights,
            "program_entropy_scale": self.program_entropy_scale,
            "crf_width": self.width,
        }


def _crf_log_partition(
    unary: Tensor, edges: Tensor, start: Tensor, end: Tensor
) -> Tensor:
    score = unary[:, 0].float() + start.float()
    for position in range(1, unary.shape[1]):
        score = unary[:, position].float() + torch.logsumexp(
            score[:, :, None] + edges[:, position - 1].float(), dim=1
        )
    return torch.logsumexp(score + end.float(), dim=-1)


def _partial_suffix_nll_rows(
    crf: tuple[Tensor, Tensor, Tensor, Tensor],
    width: int,
    labels: Tensor,
    valid: Tensor,
) -> Tensor:
    unary, edges, start, end = crf
    count = valid.sum(dim=1)
    rank = valid.long().cumsum(dim=1) - 1
    destination = (width - count[:, None] + rank).clamp(0, width - 1)
    assignment = F.one_hot(destination, width).bool() & valid.unsqueeze(-1)
    observed = assignment.any(dim=1)
    target_digits = (labels - DIGIT_OFFSET).clamp(0, NUM_DIGITS - 1)
    selected = (
        assignment.unsqueeze(-1)
        & F.one_hot(target_digits, NUM_DIGITS).bool().unsqueeze(2)
    ).any(dim=1)
    allowed = ~observed.unsqueeze(-1) | selected
    constrained = unary.float().masked_fill(~allowed, -1e4)
    return _crf_log_partition(
        unary, edges, start, end
    ) - _crf_log_partition(constrained, edges, start, end)


def token_training_loss(batch: TokenLossBatch) -> Tensor:
    valid = batch.valid_mask
    t_values = batch.auxiliary["t_values"]
    row_weight = torch.where(
        t_values == 1,
        torch.full_like(t_values, 4.0, dtype=torch.float32),
        torch.ones_like(t_values, dtype=torch.float32),
    )
    endpoint = F.cross_entropy(
        batch.logits.transpose(1, 2),
        batch.labels,
        ignore_index=-100,
        reduction="none",
    )
    token_weight = valid.float() * row_weight[:, None]
    loss = (
        (endpoint * token_weight).sum()
        / token_weight.sum().clamp_min(1.0)
    )

    width = int(batch.auxiliary["crf_width"])
    nll = torch.stack(
        tuple(
            _partial_suffix_nll_rows(crf, width, batch.labels, valid)
            for crf in batch.auxiliary["program_crfs"]
        ),
        dim=1,
    )
    log_weights = batch.auxiliary["program_log_weights"].float()
    mixture_nll = -torch.logsumexp(log_weights[None] - nll, dim=1)
    lattice = (mixture_nll * row_weight).sum() / row_weight.sum()

    probability = log_weights.exp()
    entropy = -(probability * log_weights).sum()
    entropy_scale = float(batch.auxiliary["program_entropy_scale"])
    return (
        loss
        + LATTICE_WEIGHT * lattice
        + PROGRAM_ENTROPY_WEIGHT * entropy_scale * entropy
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
            0.20, 1.20 * (0.20 / 1.20) ** fraction
        )
        self.model.program_temperature = max(
            0.18, 1.50 * (0.18 / 1.50) ** fraction
        )
        self.model.program_entropy_scale = min(
            1.0, max(0.0, (fraction - 0.55) / 0.25)
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
            {"params": decay, "weight_decay": 0.08, "base_lr": BASE_LR},
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
