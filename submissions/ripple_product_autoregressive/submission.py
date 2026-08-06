"""Product-aware recurrent reducer with an exact autoregressive endpoint tree.

The arithmetic encoder remains fully learned. Its endpoint decoder is a tied
autoregressive GRU over actual output digits. During training the model scores
every possible output prefix in parallel, independently of labels; the loss
then marginalizes unobserved leading digits and selects only evaluator-supplied
endpoint paths. Evaluation uses the same decoder greedily. No process state,
generated target, arithmetic validity mask, or nested model call is used.
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
HIDDEN = 128
DECODER = 32
REFINEMENT_STEPS = 4
MAX_OUTER_STEPS = 64
BASE_LR = 1.5e-3
AR_WEIGHT = 0.5


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


class ProductAutoregressiveSquare(nn.Module):
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
        self.scan_norm = RMSNorm(HIDDEN)
        self.scan = nn.GRU(
            HIDDEN, HIDDEN, batch_first=True, bidirectional=True
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

        self.context_projection = nn.Linear(HIDDEN, DECODER, bias=False)
        self.root = nn.Parameter(torch.empty(DECODER))
        self.root_projection = nn.Linear(HIDDEN, DECODER, bias=False)
        self.digit_embedding = nn.Parameter(torch.empty(NUM_DIGITS, DECODER))
        self.decoder_cell = nn.GRUCell(2 * DECODER, DECODER)
        self.decoder_norm = RMSNorm(DECODER)
        self.decoder_output = nn.Linear(DECODER, NUM_DIGITS)
        nn.init.normal_(self.root, std=DECODER**-0.5)
        nn.init.normal_(self.digit_embedding, std=DECODER**-0.5)

    def _product_columns(self, register: Tensor) -> Tensor:
        batch = register.shape[0]
        pair_features = torch.einsum(
            "bid,bje,deh->bijh", register, register, self.pair_embedding
        )
        positions = torch.arange(self.width, device=register.device)
        significance = (positions[:, None] + positions[None, :]).reshape(
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

    def _encode(self, register: Tensor, modulus: Tensor) -> Tensor:
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
        for _ in range(REFINEMENT_STEPS):
            scanned, _ = self.scan(self.scan_norm(state))
            state = state + torch.sigmoid(self.scan_gate) * self.scan_out(scanned)
            state = state + torch.sigmoid(self.mlp_gate) * self.mlp(
                self.mlp_norm(state)
            )
        return state[:, : self.width].flip(1)

    def _initial_hidden(self, context: Tensor) -> Tensor:
        return self.root + self.root_projection(context.mean(dim=1))

    def _training_tree(
        self, context: Tensor
    ) -> tuple[Tensor, tuple[Tensor, ...]]:
        batch = context.shape[0]
        hidden = self._initial_hidden(context)[:, None]
        decoder_context = self.context_projection(context)
        prefix_probability = torch.ones(
            batch, 1, device=context.device, dtype=context.dtype
        )
        marginals: list[Tensor] = []
        tables: list[Tensor] = []
        for position in range(self.width):
            position_context = decoder_context[:, position, None]
            readout = hidden + position_context
            logits = self.decoder_output(self.decoder_norm(readout))
            probability = F.softmax(logits, dim=-1)
            tables.append(logits)
            marginals.append(
                (prefix_probability[:, :, None] * probability).sum(dim=1)
            )
            if position + 1 < self.width:
                prefixes = hidden.shape[1]
                parent = hidden[:, :, None].expand(-1, -1, NUM_DIGITS, -1)
                digit = self.digit_embedding[None, None].expand(
                    batch, prefixes, -1, -1
                )
                local_context = decoder_context[:, position, None, None].expand(
                    -1, prefixes, NUM_DIGITS, -1
                )
                decoder_input = torch.cat((digit, local_context), dim=-1)
                hidden = self.decoder_cell(
                    decoder_input.reshape(-1, 2 * DECODER),
                    parent.reshape(-1, DECODER),
                ).reshape(batch, prefixes * NUM_DIGITS, DECODER)
                prefix_probability = (
                    prefix_probability[:, :, None] * probability
                ).reshape(batch, prefixes * NUM_DIGITS)
        return torch.stack(marginals, dim=1).flip(1), tuple(tables)

    def _greedy(self, context: Tensor) -> Tensor:
        hidden = self._initial_hidden(context)
        decoder_context = self.context_projection(context)
        digits: list[Tensor] = []
        for position in range(self.width):
            logits = self.decoder_output(
                self.decoder_norm(hidden + decoder_context[:, position])
            )
            digit = logits.argmax(dim=-1)
            digits.append(digit)
            if position + 1 < self.width:
                decoder_input = torch.cat(
                    (self.digit_embedding[digit], decoder_context[:, position]), dim=-1
                )
                hidden = self.decoder_cell(decoder_input, hidden)
        path_msd = torch.stack(digits, dim=1)
        return F.one_hot(path_msd.flip(1), NUM_DIGITS).to(context.dtype)

    def forward(
        self, register: Tensor, modulus: Tensor
    ) -> tuple[Tensor, tuple[Tensor, ...]]:
        context = self._encode(register, modulus)
        if self.training:
            return self._training_tree(context)
        return self._greedy(context), ()


class Model(nn.Module):
    num_loops = 1

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.width = max(2, (spec.max_seq_len - 5) // 2)
        if self.width > 5:
            raise ValueError("exact autoregressive tree is bounded to five digits")
        self.square = ProductAutoregressiveSquare(self.width)
        self.training_outer_steps = 3

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
        values = (input_ids - DIGIT_OFFSET).clamp(0, NUM_DIGITS - 1)
        source = F.one_hot(values, NUM_DIGITS).float() * mask.unsqueeze(-1)
        count = mask.sum(dim=1)
        rank = mask.long().cumsum(dim=1) - 1
        destination = (self.width - count[:, None] + rank).clamp(0, self.width - 1)
        register = torch.zeros(
            batch, self.width, NUM_DIGITS, device=input_ids.device
        )
        occupied = torch.zeros(batch, self.width, 1, device=input_ids.device)
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
            self._right_aligned_register(input_ids, digits & after_n & ~after_x),
            self._right_aligned_register(input_ids, digits & after_x & ~after_t),
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
            digit_logits.shape[0], prompt, NUM_DIGITS,
            device=digit_logits.device, dtype=digit_logits.dtype
        )
        canvas = canvas.scatter(
            1, destination.unsqueeze(-1).expand(-1, -1, NUM_DIGITS), digit_logits
        )
        return F.pad(
            canvas,
            (DIGIT_OFFSET, self.config.vocab_size - DIGIT_OFFSET - NUM_DIGITS),
            value=-16.0,
        )

    def forward(
        self, input_ids: Tensor, attention_mask: Tensor | None = None
    ) -> tuple[Tensor, dict[str, object]]:
        _, prompt_length = input_ids.shape
        if attention_mask is None:
            attention_mask = input_ids != PAD
        valid = attention_mask.bool()
        modulus, register, t_values = self._parse(input_ids, valid)
        loops = self.training_outer_steps if self.training else int(t_values.max().item())
        terminal_tables: list[Tensor] | None = None
        for outer_step in range(loops):
            transition_input = register.detach() if self.training and outer_step else register
            candidate, tables = self.square(transition_input, modulus)
            if self.training:
                if terminal_tables is None:
                    terminal_tables = [torch.zeros_like(table) for table in tables]
                terminal = t_values == outer_step + 1
                terminal_tables = [
                    torch.where(terminal[:, None, None], table, selected)
                    for table, selected in zip(tables, terminal_tables)
                ]
            register = torch.where(
                (t_values > outer_step)[:, None, None], candidate, register
            )
        lengths = valid.sum(dim=1)
        digit_logits_lsd = register.clamp_min(1e-8).log()
        logits = self._place_logits(digit_logits_lsd, lengths, prompt_length)
        return logits, {
            "t_values": t_values,
            "width": self.width,
            "tables": () if terminal_tables is None else tuple(terminal_tables),
        }


def _observations(batch: TokenLossBatch, width: int) -> tuple[Tensor, Tensor]:
    valid = batch.valid_mask
    count = valid.sum(dim=1)
    rank = valid.long().cumsum(dim=1) - 1
    destination = (width - count[:, None] + rank).clamp(0, width - 1)
    assignment = F.one_hot(destination, width).bool() & valid.unsqueeze(-1)
    observed = assignment.any(dim=1)
    digits = (batch.labels - DIGIT_OFFSET).clamp(0, NUM_DIGITS - 1)
    selected = (
        assignment.unsqueeze(-1)
        & F.one_hot(digits, NUM_DIGITS).bool().unsqueeze(2)
    ).any(dim=1)
    return observed, selected


def _autoregressive_nll(batch: TokenLossBatch) -> Tensor:
    tables = batch.auxiliary["tables"]
    observed, selected = _observations(batch, int(batch.auxiliary["width"]))
    batch_size = observed.shape[0]
    prefix_score = torch.zeros(
        batch_size, 1, device=observed.device, dtype=torch.float32
    )
    allowed_prefix = torch.ones(
        batch_size, 1, device=observed.device, dtype=torch.bool
    )
    for position, table in enumerate(tables):
        local = F.log_softmax(table.float(), dim=-1)
        prefix_score = (prefix_score[:, :, None] + local).flatten(1)
        allowed_digit = ~observed[:, position, None] | selected[:, position]
        allowed_prefix = (
            allowed_prefix[:, :, None] & allowed_digit[:, None]
        ).flatten(1)
    constrained = prefix_score.masked_fill(~allowed_prefix, -1e4)
    nll = -torch.logsumexp(constrained, dim=-1)
    row_weight = torch.where(
        batch.auxiliary["t_values"] == 1,
        torch.full_like(nll, 4.0),
        torch.ones_like(nll),
    )
    return (nll * row_weight).sum() / row_weight.sum().clamp_min(1.0)


def token_training_loss(batch: TokenLossBatch) -> Tensor:
    t_values = batch.auxiliary["t_values"]
    row_weight = torch.where(
        t_values == 1,
        torch.full_like(t_values, 4.0, dtype=torch.float32),
        torch.ones_like(t_values, dtype=torch.float32),
    )
    weights = batch.valid_mask.float() * row_weight[:, None]
    losses = F.cross_entropy(
        batch.logits.transpose(1, 2), batch.labels,
        ignore_index=-100, reduction="none"
    )
    endpoint = (losses * weights).sum() / weights.sum().clamp_min(1.0)
    return endpoint + AR_WEIGHT * _autoregressive_nll(batch)


class WallClockSchedule:
    def __init__(self, optimizer: torch.optim.Optimizer, budget_seconds: float) -> None:
        self.optimizer = optimizer
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
        scheduler=WallClockSchedule(optimizer, spec.training_time_seconds),
    )


SUBMISSION = Submission(
    build_model=build_model,
    build_optimizer=build_optimizer,
    token_training_loss=token_training_loss,
    batch_size=32,
    eval_batch_size=512,
)
