"""Observable belief-state reduction from endpoint-only supervision.

The square transition scans learned product/modulus evidence columns. Its only
recurrent reduction state is a normalized first-order distribution over the
actual output digits: an initial digit distribution and row-normalized adjacent
digit transitions. A tied learned evidence update changes that belief after
each column, then exact forward/backward normalization returns a canonical
belief for the next update. There is no persistent hidden controller, latent
state alphabet, EMA teacher, arithmetic process label, or hard-coded reducer.

Every T=1 prefix belief is scored against the supplied endpoint. Since log loss
is strictly proper, its population optimum is the endpoint distribution given
the evidence prefix. The final belief is the submitted prediction and retains
a direct endpoint gradient on every row.
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
MAX_OUTER_STEPS = 64
BASE_LR = 2.0e-3
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


def _normalize_chain(
    unary: Tensor,
    edges: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return canonical log initial/transitions and node log marginals."""

    width = unary.shape[1]
    forward = [unary[:, 0].float()]
    for position in range(1, width):
        forward.append(
            unary[:, position].float()
            + torch.logsumexp(
                forward[-1][:, :, None] + edges[:, position - 1].float(),
                dim=1,
            )
        )
    backward: list[Tensor] = [torch.empty(0)] * width
    backward[-1] = torch.zeros_like(forward[-1])
    for position in range(width - 2, -1, -1):
        backward[position] = torch.logsumexp(
            edges[:, position].float()
            + unary[:, position + 1].float()[:, None, :]
            + backward[position + 1][:, None, :],
            dim=2,
        )
    log_partition = torch.logsumexp(forward[-1], dim=-1)
    node_log = torch.stack(
        [
            forward[position] + backward[position] - log_partition[:, None]
            for position in range(width)
        ],
        dim=1,
    )
    log_initial = F.log_softmax(node_log[:, 0], dim=-1)
    transitions: list[Tensor] = []
    for position in range(width - 1):
        pair_log = (
            forward[position][:, :, None]
            + edges[:, position].float()
            + unary[:, position + 1].float()[:, None, :]
            + backward[position + 1][:, None, :]
            - log_partition[:, None, None]
        )
        conditional = pair_log - node_log[:, position, :, None]
        transitions.append(F.log_softmax(conditional, dim=-1))
    if transitions:
        log_transition = torch.stack(transitions, dim=1)
    else:
        log_transition = unary.new_empty(
            unary.shape[0], 0, NUM_DIGITS, NUM_DIGITS, dtype=torch.float32
        )
    return log_initial, log_transition, node_log


def _canonical_node_log(
    log_initial: Tensor,
    log_transition: Tensor,
) -> Tensor:
    nodes = [F.log_softmax(log_initial.float(), dim=-1)]
    for position in range(log_transition.shape[1]):
        nodes.append(
            torch.logsumexp(
                nodes[-1][:, :, None]
                + F.log_softmax(log_transition[:, position].float(), dim=-1),
                dim=1,
            )
        )
    return torch.stack(nodes, dim=1)


class ObservableBeliefSquare(nn.Module):
    """One square whose complete recurrent state is an output-chain belief."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = width
        self.columns = 2 * width - 1
        self.pair_embedding = nn.Parameter(
            torch.empty(NUM_DIGITS, NUM_DIGITS, HIDDEN)
        )
        nn.init.normal_(self.pair_embedding, std=HIDDEN**-0.5)
        self.modulus_projection = nn.Linear(NUM_DIGITS, HIDDEN, bias=False)
        self.belief_projection = nn.Linear(NUM_DIGITS, HIDDEN, bias=False)
        self.position_projection = nn.Linear(4, HIDDEN, bias=False)
        self.update_norm = RMSNorm(HIDDEN)
        self.update_mlp = nn.Sequential(
            nn.Linear(HIDDEN, 2 * HIDDEN, bias=False),
            nn.SiLU(),
            nn.Linear(2 * HIDDEN, HIDDEN, bias=False),
        )
        self.unary_head = nn.Linear(HIDDEN, NUM_DIGITS)
        self.edge_head = nn.Linear(2 * HIDDEN, NUM_DIGITS * NUM_DIGITS)
        self.update_gate = nn.Parameter(torch.tensor(-1.0))

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

    def _position_features(
        self,
        column: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        positions = torch.arange(self.width, device=device, dtype=dtype)
        position_scale = float(max(self.width - 1, 1))
        column_scale = float(max(self.columns - 1, 1))
        position = positions / position_scale
        column_value = torch.full_like(position, column / column_scale)
        relative = position - column_value
        return torch.stack(
            (position, column_value, relative, relative.abs()), dim=-1
        )

    def forward(
        self,
        register: Tensor,
        modulus: Tensor,
    ) -> tuple[
        Tensor,
        tuple[tuple[Tensor, Tensor], ...],
        tuple[Tensor, Tensor],
    ]:
        batch = register.shape[0]
        product = self._product_columns(register)
        modulus_features = self.modulus_projection(modulus)
        log_initial = register.new_full(
            (batch, NUM_DIGITS), -math.log(NUM_DIGITS)
        )
        log_transition = register.new_full(
            (batch, self.width - 1, NUM_DIGITS, NUM_DIGITS),
            -math.log(NUM_DIGITS),
        )
        prefix_beliefs: list[tuple[Tensor, Tensor]] = []
        gate = torch.sigmoid(self.update_gate)

        for column in range(self.columns):
            node_log = _canonical_node_log(log_initial, log_transition)
            belief_features = self.belief_projection(node_log.exp())
            evidence = product[:, column]
            if column < self.width:
                evidence = evidence + modulus_features[:, column]
            position = self.position_projection(
                self._position_features(
                    column, evidence.device, evidence.dtype
                )
            )
            features = belief_features + evidence[:, None, :] + position[None]
            features = features + self.update_mlp(self.update_norm(features))
            unary_delta = self.unary_head(features)
            edge_delta = self.edge_head(
                torch.cat((features[:, :-1], features[:, 1:]), dim=-1)
            ).reshape(batch, self.width - 1, NUM_DIGITS, NUM_DIGITS)

            unary = unary_delta * gate
            unary[:, 0] = unary[:, 0] + log_initial
            edges = edge_delta * gate + log_transition
            log_initial, log_transition, node_log = _normalize_chain(
                unary, edges
            )
            prefix_beliefs.append((log_initial, log_transition))

        return node_log.exp(), tuple(prefix_beliefs), (
            log_initial,
            log_transition,
        )


class Model(nn.Module):
    num_loops = 1

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.max_length = spec.max_seq_len
        self.width = max(2, (spec.max_seq_len - 5) // 2)
        self.square = ObservableBeliefSquare(self.width)
        self.training_outer_steps = 3

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

    def _right_aligned_register(self, input_ids: Tensor, mask: Tensor) -> Tensor:
        batch = input_ids.shape[0]
        digit_values = (input_ids - DIGIT_OFFSET).clamp(0, NUM_DIGITS - 1)
        source = F.one_hot(digit_values, NUM_DIGITS).to(torch.float32)
        source = source * mask.unsqueeze(-1)
        count = mask.sum(dim=1)
        rank = mask.long().cumsum(dim=1) - 1
        destination = (self.width - count[:, None] + rank).clamp(
            0, self.width - 1
        )
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
        occupied.scatter_add_(
            1, destination.unsqueeze(-1), mask.unsqueeze(-1).float()
        )
        register[:, :, 0] = register[:, :, 0] + (1.0 - occupied.squeeze(-1))
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

    @staticmethod
    def _viterbi(log_initial: Tensor, log_transition: Tensor) -> Tensor:
        score = log_initial
        backpointers: list[Tensor] = []
        for position in range(log_transition.shape[1]):
            candidates = score[:, :, None] + log_transition[:, position]
            score, previous = candidates.max(dim=1)
            backpointers.append(previous)
        current = score.argmax(dim=-1)
        path = [current]
        for previous in reversed(backpointers):
            current = previous.gather(1, current[:, None]).squeeze(1)
            path.append(current)
        return torch.stack(tuple(reversed(path)), dim=1)

    def _run_squares(
        self,
        register: Tensor,
        modulus: Tensor,
        t_values: Tensor,
        loops: int,
    ) -> tuple[
        tuple[Tensor, Tensor],
        tuple[tuple[Tensor, Tensor], ...],
    ]:
        terminal_initial = register.new_zeros(register.shape[0], NUM_DIGITS)
        terminal_transition = register.new_zeros(
            register.shape[0], self.width - 1, NUM_DIGITS, NUM_DIGITS
        )
        first_prefixes: tuple[tuple[Tensor, Tensor], ...] = ()
        for outer_step in range(loops):
            transition_input = (
                register.detach()
                if self.training and outer_step > 0
                else register
            )
            candidate, prefixes, chain = self.square(
                transition_input, modulus
            )
            if outer_step == 0:
                first_prefixes = prefixes
            terminal = t_values == outer_step + 1
            terminal_initial = torch.where(
                terminal[:, None], chain[0], terminal_initial
            )
            terminal_transition = torch.where(
                terminal[:, None, None, None], chain[1], terminal_transition
            )
            active = (t_values > outer_step)[:, None, None]
            register = torch.where(active, candidate, register)
        return (terminal_initial, terminal_transition), first_prefixes

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
            self.training_outer_steps
            if self.training
            else int(t_values.max().item())
        )
        final_chain, first_prefixes = self._run_squares(
            register, modulus, t_values, loops
        )
        if self.training:
            digit_logits_lsd = _canonical_node_log(*final_chain)
        else:
            path = self._viterbi(*final_chain)
            digit_logits_lsd = torch.full(
                (*path.shape, NUM_DIGITS),
                -16.0,
                device=path.device,
                dtype=final_chain[0].dtype,
            )
            digit_logits_lsd.scatter_(2, path.unsqueeze(-1), 0.0)
        lengths = valid.sum(dim=1)
        logits = self._place_logits(
            digit_logits_lsd, lengths, prompt_length
        )
        return logits, {
            "t_values": t_values,
            "final_chain": final_chain,
            "prefix_chains": first_prefixes,
            "belief_width": self.width,
        }


def _partial_chain_nll(
    chain: tuple[Tensor, Tensor],
    batch: TokenLossBatch,
    row_mask: Tensor | None = None,
) -> Tensor:
    log_initial, log_transition = chain
    width = int(batch.auxiliary["belief_width"])
    valid = batch.valid_mask
    count = valid.sum(dim=1)
    rank = valid.long().cumsum(dim=1) - 1
    destination = (count[:, None] - 1 - rank).clamp(0, width - 1)
    assignment = F.one_hot(destination, width).bool() & valid.unsqueeze(-1)
    observed = assignment.any(dim=1)
    target_digits = (batch.labels - DIGIT_OFFSET).clamp(0, NUM_DIGITS - 1)
    target_one_hot = F.one_hot(target_digits, NUM_DIGITS).bool()
    selected = (
        assignment.unsqueeze(-1) & target_one_hot.unsqueeze(2)
    ).any(dim=1)
    allowed = ~observed.unsqueeze(-1) | selected
    constraint = log_initial.new_zeros(
        log_initial.shape[0], width, NUM_DIGITS, dtype=torch.float32
    )
    constraint = constraint.masked_fill(~allowed, -1e4)

    score = F.log_softmax(log_initial.float(), dim=-1) + constraint[:, 0]
    for position in range(width - 1):
        transition = F.log_softmax(
            log_transition[:, position].float(), dim=-1
        )
        score = (
            torch.logsumexp(score[:, :, None] + transition, dim=1)
            + constraint[:, position + 1]
        )
    losses = -torch.logsumexp(score, dim=-1)
    if row_mask is None:
        return losses.mean()
    weights = row_mask.to(losses.dtype)
    return (losses * weights).sum() / weights.sum().clamp_min(1.0)


def token_training_loss(batch: TokenLossBatch) -> Tensor:
    final_loss = _partial_chain_nll(batch.auxiliary["final_chain"], batch)
    t1_rows = batch.auxiliary["t_values"] == 1
    prefix_terms = [
        _partial_chain_nll(chain, batch, t1_rows)
        for chain in batch.auxiliary["prefix_chains"]
    ]
    if not prefix_terms:
        return final_loss
    return final_loss + PREFIX_WEIGHT * torch.stack(prefix_terms).mean()


class WallClockSchedule:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        budget_seconds: float,
    ) -> None:
        self.optimizer = optimizer
        self.started_at = time.monotonic()
        self.budget_seconds = max(float(budget_seconds) * 0.98, 1.0)

    def step(self) -> None:
        fraction = min(
            (time.monotonic() - self.started_at) / self.budget_seconds, 1.0
        )
        if fraction < 0.05:
            multiplier = 0.1 + 0.9 * fraction / 0.05
        else:
            progress = (fraction - 0.05) / 0.95
            multiplier = 0.08 + 0.92 * 0.5 * (
                1.0 + math.cos(math.pi * progress)
            )
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
        scheduler=WallClockSchedule(optimizer, spec.training_time_seconds),
    )


SUBMISSION = Submission(
    build_model=build_model,
    build_optimizer=build_optimizer,
    token_training_loss=token_training_loss,
    batch_size=32,
    eval_batch_size=512,
)
