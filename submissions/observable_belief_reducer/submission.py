"""Observable belief-state reducer for repeated modular squaring.

The only recurrent reduction state is a normalized first-order distribution
over the actual output digits. Learned evidence updates consume product/modulus
columns and revise that distribution in place. There is no persistent hidden
controller, free latent alphabet, carry state, quotient state, or arithmetic
transition table. Endpoint labels score the final belief and, for T=1 rows,
each evidence-prefix belief with the same proper chain likelihood.
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
EVIDENCE_WIDTH = 96
MAX_OUTER_STEPS = 64
BASE_LR = 1.5e-3
CHAIN_WEIGHT = 0.2
PREFIX_WEIGHT = 0.25


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


def _chain_marginals(start_logp: Tensor, transition_logp: Tensor) -> Tensor:
    marginals = [start_logp.exp()]
    for position in range(transition_logp.shape[1]):
        next_probability = torch.einsum(
            "bi,bij->bj", marginals[-1], transition_logp[:, position].exp()
        )
        marginals.append(next_probability)
    return torch.stack(marginals, dim=1)


class ObservableBeliefSquare(nn.Module):
    """One learned square whose evolving state is an output-digit Markov chain."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = width
        self.columns = 2 * width - 1
        self.pair_embedding = nn.Parameter(
            torch.empty(NUM_DIGITS, NUM_DIGITS, EVIDENCE_WIDTH)
        )
        nn.init.normal_(self.pair_embedding, std=EVIDENCE_WIDTH**-0.5)
        self.modulus_projection = nn.Linear(
            NUM_DIGITS, EVIDENCE_WIDTH, bias=False
        )
        self.register_projection = nn.Linear(
            NUM_DIGITS, EVIDENCE_WIDTH, bias=False
        )
        self.evidence_position = nn.Linear(2, EVIDENCE_WIDTH, bias=False)
        self.output_position = nn.Linear(2, EVIDENCE_WIDTH, bias=False)
        self.evidence_norm = RMSNorm(EVIDENCE_WIDTH)
        self.evidence_mlp = nn.Sequential(
            nn.Linear(EVIDENCE_WIDTH, 2 * EVIDENCE_WIDTH, bias=False),
            nn.SiLU(),
            nn.Linear(2 * EVIDENCE_WIDTH, EVIDENCE_WIDTH, bias=False),
        )
        self.start_prior = nn.Parameter(torch.zeros(NUM_DIGITS))
        self.transition_prior = nn.Parameter(
            torch.zeros(width - 1, NUM_DIGITS, NUM_DIGITS)
        )
        self.start_update = nn.Sequential(
            nn.Linear(EVIDENCE_WIDTH + NUM_DIGITS, 2 * EVIDENCE_WIDTH),
            nn.SiLU(),
            nn.Linear(2 * EVIDENCE_WIDTH, NUM_DIGITS),
        )
        self.transition_update = nn.Sequential(
            nn.Linear(
                2 * EVIDENCE_WIDTH + NUM_DIGITS * NUM_DIGITS,
                2 * EVIDENCE_WIDTH,
            ),
            nn.SiLU(),
            nn.Linear(2 * EVIDENCE_WIDTH, NUM_DIGITS * NUM_DIGITS),
        )
        self.start_gate = nn.Parameter(torch.tensor(-1.0))
        self.transition_gate = nn.Parameter(torch.tensor(-1.0))

    def _positions(self, count: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        if count == 1:
            position = torch.zeros(1, device=device, dtype=dtype)
        else:
            position = torch.linspace(0.0, 1.0, count, device=device, dtype=dtype)
        return torch.stack((position, position.square()), dim=-1)

    def _evidence(self, register: Tensor, modulus: Tensor) -> Tensor:
        batch = register.shape[0]
        pair_features = torch.einsum(
            "bid,bje,deh->bijh", register, register, self.pair_embedding
        )
        digit_position = torch.arange(self.width, device=register.device)
        significance = (digit_position[:, None] + digit_position[None, :]).reshape(
            1, self.width * self.width, 1
        )
        significance = significance.expand(batch, -1, EVIDENCE_WIDTH)
        product = torch.zeros(
            batch,
            self.columns,
            EVIDENCE_WIDTH,
            device=register.device,
            dtype=pair_features.dtype,
        )
        product = product.scatter_add(
            1, significance, pair_features.reshape(batch, -1, EVIDENCE_WIDTH)
        )
        padding = self.columns - self.width
        modulus_columns = F.pad(modulus, (0, 0, 0, padding))
        register_columns = F.pad(register, (0, 0, 0, padding))
        position = self.evidence_position(
            self._positions(self.columns, register.device, register.dtype)
        )
        evidence = (
            product
            + self.modulus_projection(modulus_columns)
            + self.register_projection(register_columns)
            + position
        )
        return evidence + self.evidence_mlp(self.evidence_norm(evidence))

    def forward(
        self, register: Tensor, modulus: Tensor
    ) -> tuple[
        Tensor,
        tuple[tuple[Tensor, Tensor], ...],
        tuple[Tensor, Tensor],
    ]:
        batch = register.shape[0]
        evidence_columns = self._evidence(register, modulus)
        start_logp = F.log_softmax(self.start_prior, dim=-1).expand(batch, -1)
        transition_logp = F.log_softmax(
            self.transition_prior, dim=-1
        ).expand(batch, -1, -1, -1)
        output_position = self.output_position(
            self._positions(
                self.width - 1, register.device, evidence_columns.dtype
            )
        )
        beliefs: list[tuple[Tensor, Tensor]] = []
        for column in range(self.columns):
            evidence = evidence_columns[:, column]
            start_delta = self.start_update(
                torch.cat((evidence, start_logp.exp()), dim=-1)
            )
            start_logp = F.log_softmax(
                start_logp + torch.sigmoid(self.start_gate) * start_delta,
                dim=-1,
            )

            repeated_evidence = evidence[:, None].expand(
                -1, self.width - 1, -1
            )
            transition_input = torch.cat(
                (
                    repeated_evidence,
                    output_position[None].expand(batch, -1, -1),
                    transition_logp.exp().flatten(-2),
                ),
                dim=-1,
            )
            transition_delta = self.transition_update(transition_input).reshape(
                batch, self.width - 1, NUM_DIGITS, NUM_DIGITS
            )
            transition_logp = F.log_softmax(
                transition_logp
                + torch.sigmoid(self.transition_gate) * transition_delta,
                dim=-1,
            )
            beliefs.append((start_logp, transition_logp))

        marginals_msd = _chain_marginals(start_logp, transition_logp)
        return marginals_msd.flip(1), tuple(beliefs), (
            start_logp,
            transition_logp,
        )


class Model(nn.Module):
    num_loops = 1

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.width = max(2, (spec.max_seq_len - 5) // 2)
        self.square = ObservableBeliefSquare(self.width)
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
        digit_values = (input_ids - DIGIT_OFFSET).clamp(0, NUM_DIGITS - 1)
        source = F.one_hot(digit_values, NUM_DIGITS).float() * mask.unsqueeze(-1)
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
        n_mask = digits & after_n & ~after_x
        x_mask = digits & after_x & ~after_t
        return (
            self._right_aligned_register(input_ids, n_mask),
            self._right_aligned_register(input_ids, x_mask),
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

    @staticmethod
    def _viterbi(start_logp: Tensor, transition_logp: Tensor) -> Tensor:
        score = start_logp
        backpointers: list[Tensor] = []
        for position in range(transition_logp.shape[1]):
            candidates = score[:, :, None] + transition_logp[:, position]
            score, previous = candidates.max(dim=1)
            backpointers.append(previous)
        current = score.argmax(dim=-1)
        path = [current]
        for previous in reversed(backpointers):
            current = previous.gather(1, current[:, None]).squeeze(1)
            path.append(current)
        return torch.stack(tuple(reversed(path)), dim=1)

    def forward(
        self, input_ids: Tensor, attention_mask: Tensor | None = None
    ) -> tuple[Tensor, dict[str, object]]:
        _, prompt_length = input_ids.shape
        if attention_mask is None:
            attention_mask = input_ids != PAD
        valid = attention_mask.bool()
        modulus, register, t_values = self._parse(input_ids, valid)
        loops = self.training_outer_steps if self.training else int(t_values.max().item())
        first_beliefs: tuple[tuple[Tensor, Tensor], ...] = ()
        terminal_start: Tensor | None = None
        terminal_transition: Tensor | None = None
        for outer_step in range(loops):
            transition_input = register.detach() if self.training and outer_step else register
            candidate, beliefs, chain = self.square(transition_input, modulus)
            if outer_step == 0:
                first_beliefs = beliefs
                terminal_start = torch.zeros_like(chain[0])
                terminal_transition = torch.zeros_like(chain[1])
            terminal = t_values == outer_step + 1
            if terminal_start is None or terminal_transition is None:
                raise RuntimeError("belief chain was not initialized")
            terminal_start = torch.where(terminal[:, None], chain[0], terminal_start)
            terminal_transition = torch.where(
                terminal[:, None, None, None], chain[1], terminal_transition
            )
            register = torch.where(
                (t_values > outer_step)[:, None, None], candidate, register
            )
        if terminal_start is None or terminal_transition is None:
            raise RuntimeError("at least one square is required")

        lengths = valid.sum(dim=1)
        if self.training:
            digit_logits_lsd = register.clamp_min(1e-8).log()
        else:
            path_msd = self._viterbi(terminal_start, terminal_transition)
            path_lsd = path_msd.flip(1)
            digit_logits_lsd = torch.full(
                (*path_lsd.shape, NUM_DIGITS),
                -16.0,
                device=path_lsd.device,
                dtype=terminal_start.dtype,
            )
            digit_logits_lsd.scatter_(2, path_lsd.unsqueeze(-1), 0.0)
        logits = self._place_logits(digit_logits_lsd, lengths, prompt_length)
        return logits, {
            "t_values": t_values,
            "width": self.width,
            "beliefs": first_beliefs,
            "terminal_chain": (terminal_start, terminal_transition),
        }


def _observations(batch: TokenLossBatch, width: int) -> tuple[Tensor, Tensor]:
    valid = batch.valid_mask
    count = valid.sum(dim=1)
    rank = valid.long().cumsum(dim=1) - 1
    destination = (width - count[:, None] + rank).clamp(0, width - 1)
    assignment = F.one_hot(destination, width).bool() & valid.unsqueeze(-1)
    observed = assignment.any(dim=1)
    target_digits = (batch.labels - DIGIT_OFFSET).clamp(0, NUM_DIGITS - 1)
    selected = (
        assignment.unsqueeze(-1)
        & F.one_hot(target_digits, NUM_DIGITS).bool().unsqueeze(2)
    ).any(dim=1)
    return observed, selected


def _chain_nll(
    chain: tuple[Tensor, Tensor],
    observed: Tensor,
    selected: Tensor,
    row_mask: Tensor | None = None,
) -> Tensor:
    start_logp, transition_logp = chain
    negative = torch.finfo(torch.float32).min
    alpha = start_logp.float().masked_fill(
        observed[:, 0, None] & ~selected[:, 0], negative
    )
    for position in range(1, observed.shape[1]):
        alpha = torch.logsumexp(
            alpha[:, :, None] + transition_logp[:, position - 1].float(), dim=1
        )
        alpha = alpha.masked_fill(
            observed[:, position, None] & ~selected[:, position], negative
        )
    nll = -torch.logsumexp(alpha, dim=-1)
    if row_mask is not None:
        weights = row_mask.to(nll.dtype)
        return (nll * weights).sum() / weights.sum().clamp_min(1.0)
    return nll.mean()


def token_training_loss(batch: TokenLossBatch) -> Tensor:
    valid = batch.valid_mask
    t_values = batch.auxiliary["t_values"]
    row_weight = torch.where(
        t_values == 1,
        torch.full_like(t_values, 4.0, dtype=torch.float32),
        torch.ones_like(t_values, dtype=torch.float32),
    )
    token_weight = valid.float() * row_weight[:, None]
    losses = F.cross_entropy(
        batch.logits.transpose(1, 2),
        batch.labels,
        ignore_index=-100,
        reduction="none",
    )
    endpoint = (losses * token_weight).sum() / token_weight.sum().clamp_min(1.0)
    observed, selected = _observations(batch, int(batch.auxiliary["width"]))
    final_chain = _chain_nll(batch.auxiliary["terminal_chain"], observed, selected)
    t1 = t_values == 1
    prefix_terms = [
        _chain_nll(chain, observed, selected, t1)
        for chain in batch.auxiliary["beliefs"]
    ]
    prefix = torch.stack(prefix_terms).mean()
    return endpoint + CHAIN_WEIGHT * final_chain + PREFIX_WEIGHT * prefix


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
