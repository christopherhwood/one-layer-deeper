"""Product-aware ripple/EMA squaring with a terminal observed-digit trigram CRF.

The student is the tracked product-aware H100 candidate: digit-pair features
are grouped by significance and a tied bidirectional recurrent scan learns the
complete modular-square transition.  During training, a frozen EMA copy runs
on the same untouched digit registers in the same forward pass.  The ordinary
student endpoint remains the submitted prediction with a direct gradient path;
the only added signal is softened endpoint agreement with the temporal teacher.
The recurrent register remains unchanged and factorized. A label-independent,
globally normalized chain scores only the final fixed-width output digits. It
retains the unary and pairwise terms of the surviving first-order decoder and
adds a zero-initialized, input-conditioned trigram residual. Training uses only
partial endpoint suffixes; evaluation uses second-order Viterbi. Every chain
state is an observed decimal digit, never a latent arithmetic state.
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
REFINEMENT_STEPS = 4
MAX_OUTER_STEPS = 64
BASE_LR = 1.5e-3
EMA_DECAY = 0.995
EMA_WEIGHT = 0.15
CONSISTENCY_TEMPERATURE = 2.0
CRF_WEIGHT = 0.2
TRIGRAM_RANK = 16


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
        self.pair_embedding = nn.Parameter(
            torch.empty(NUM_DIGITS, NUM_DIGITS, HIDDEN)
        )
        self.modulus_projection = nn.Linear(NUM_DIGITS, HIDDEN, bias=False)
        self.register_projection = nn.Linear(NUM_DIGITS, HIDDEN, bias=False)
        self.input_projection = nn.Linear(3 * HIDDEN, HIDDEN, bias=False)
        nn.init.normal_(self.pair_embedding, std=HIDDEN**-0.5)
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
        self.edge_norm = RMSNorm(HIDDEN)
        self.edge_head = nn.Linear(2 * HIDDEN, NUM_DIGITS * NUM_DIGITS)
        self.start_scores = nn.Parameter(torch.empty(NUM_DIGITS))
        self.end_scores = nn.Parameter(torch.empty(NUM_DIGITS))
        nn.init.normal_(self.start_scores, std=0.02)
        nn.init.normal_(self.end_scores, std=0.02)

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
    ) -> tuple[Tensor, tuple[Tensor, ...], tuple[Tensor, ...]]:
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
        phase_states: list[Tensor] = []
        for _ in range(REFINEMENT_STEPS):
            scanned, _ = self.scan(self.scan_norm(state))
            state = state + torch.sigmoid(self.scan_gate) * self.scan_out(scanned)
            state = state + torch.sigmoid(self.mlp_gate) * self.mlp(
                self.mlp_norm(state)
            )
            output_state = state[:, : self.width]
            phase_states.append(output_state)
            phase_logits.append(self.output(self.output_norm(output_state)))
        probabilities = F.softmax(phase_logits[-1] / temperature, dim=-1)
        return probabilities, tuple(phase_logits), tuple(phase_states)

    def crf_energies(
        self,
        reducer_state_lsd: Tensor,
        unary_lsd: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Build one fixed-width observed-digit energy from a selected state."""

        output_state = self.edge_norm(reducer_state_lsd).flip(1)
        adjacent = torch.cat(
            (output_state[:, :-1], output_state[:, 1:]), dim=-1
        )
        edges = self.edge_head(adjacent).reshape(
            reducer_state_lsd.shape[0], self.width - 1, NUM_DIGITS, NUM_DIGITS
        )
        unary = unary_lsd.flip(1)
        return unary, edges, self.start_scores, self.end_scores, output_state


class Model(nn.Module):
    num_loops = 1

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.max_length = spec.max_seq_len
        self.width = max(2, (spec.max_seq_len - 5) // 2)
        self.square = ProductRippleSquare(self.width)
        self.teacher_square = ProductRippleSquare(self.width)
        self.teacher_square.load_state_dict(self.square.state_dict())
        self.teacher_square.requires_grad_(False)
        # Student-only terminal decoder. The zero final projection makes the
        # initial model exactly the retained first-order observed-digit CRF.
        self.trigram_context = nn.Linear(3 * HIDDEN, TRIGRAM_RANK)
        self.trigram_head = nn.Linear(
            TRIGRAM_RANK, NUM_DIGITS**3, bias=False
        )
        nn.init.zeros_(self.trigram_head.weight)
        self.training_outer_steps = 3
        self.training_temperature = 1.2
        self.eval_temperature = 0.12
        self.consistency_scale = 0.0

    def _trigram_energies(self, output_state_msd: Tensor) -> Tensor:
        batch, width, _ = output_state_msd.shape
        if width < 3:
            return output_state_msd.new_zeros(
                batch, 0, NUM_DIGITS, NUM_DIGITS, NUM_DIGITS
            )
        triples = torch.cat(
            (
                output_state_msd[:, :-2],
                output_state_msd[:, 1:-1],
                output_state_msd[:, 2:],
            ),
            dim=-1,
        )
        factors = self.trigram_head(
            F.silu(self.trigram_context(triples))
        )
        return factors.reshape(
            batch,
            width - 2,
            NUM_DIGITS,
            NUM_DIGITS,
            NUM_DIGITS,
        )

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

    def _run_square(
        self,
        square: ProductRippleSquare,
        register: Tensor,
        modulus: Tensor,
        t_values: Tensor,
        loops: int,
        temperature: float,
        detach_outer: bool,
    ) -> tuple[
        Tensor,
        tuple[Tensor, ...],
        Tensor,
        Tensor,
    ]:
        first_phases: tuple[Tensor, ...] = ()
        terminal_unary: Tensor | None = None
        terminal_state: Tensor | None = None
        for outer_step in range(loops):
            transition_input = (
                register.detach() if detach_outer and outer_step > 0 else register
            )
            candidate, phases, states = square(
                transition_input, modulus, temperature
            )
            if outer_step == 0:
                first_phases = phases
                terminal_unary = torch.zeros_like(phases[-1])
                terminal_state = torch.zeros_like(states[-1])
            terminal = (t_values == outer_step + 1)
            if terminal_unary is None or terminal_state is None:
                raise RuntimeError("terminal reducer state was not initialized")
            terminal_unary = torch.where(
                terminal[:, None, None], phases[-1], terminal_unary
            )
            terminal_state = torch.where(
                terminal[:, None, None], states[-1], terminal_state
            )
            active = (t_values > outer_step)[:, None, None]
            register = torch.where(active, candidate, register)
        if terminal_unary is None or terminal_state is None:
            raise RuntimeError("at least one outer square is required")
        return (
            register,
            first_phases,
            terminal_unary,
            terminal_state,
        )

    @staticmethod
    def _first_order_viterbi(
        unary: Tensor,
        edges: Tensor,
        start: Tensor,
        end: Tensor,
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

    @staticmethod
    def _viterbi(
        unary: Tensor,
        edges: Tensor,
        trigrams: Tensor,
        start: Tensor,
        end: Tensor,
    ) -> Tensor:
        width = unary.shape[1]
        if width < 3:
            return Model._first_order_viterbi(unary, edges, start, end)

        # Pair state (a,b) remembers only the two preceding observed digits.
        score = (
            unary[:, 0, :, None]
            + start[None, :, None]
            + edges[:, 0]
            + unary[:, 1, None, :]
        )
        backpointers: list[Tensor] = []
        for position in range(2, width):
            candidates = score[:, :, :, None] + trigrams[:, position - 2]
            score, previous = candidates.max(dim=1)
            score = (
                score
                + edges[:, position - 1]
                + unary[:, position, None, :]
            )
            backpointers.append(previous)

        batch_rows = torch.arange(unary.shape[0], device=unary.device)
        terminal = (score + end[None, None, :]).flatten(1).argmax(dim=-1)
        left = torch.div(terminal, NUM_DIGITS, rounding_mode="floor")
        right = terminal.remainder(NUM_DIGITS)
        reverse_path = [right, left]
        for previous in reversed(backpointers):
            prior = previous[batch_rows, left, right]
            reverse_path.append(prior)
            right = left
            left = prior
        return torch.stack(tuple(reversed(reverse_path)), dim=1)

    @torch.no_grad()
    def update_teacher(self, decay: float) -> None:
        student_parameters = dict(self.square.named_parameters())
        for name, teacher in self.teacher_square.named_parameters():
            teacher.lerp_(student_parameters[name], 1.0 - decay)
        student_buffers = dict(self.square.named_buffers())
        for name, teacher in self.teacher_square.named_buffers():
            teacher.copy_(student_buffers[name])

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, object]]:
        _, prompt_length = input_ids.shape
        if attention_mask is None:
            attention_mask = input_ids != PAD
        valid = attention_mask.bool()
        modulus, initial_register, t_values = self._parse(input_ids, valid)
        loops = (
            self.training_outer_steps
            if self.training
            else int(t_values.max().item())
        )
        temperature = self.training_temperature if self.training else self.eval_temperature
        (
            register,
            first_square_phases,
            terminal_unary,
            terminal_state,
        ) = self._run_square(
            self.square,
            initial_register,
            modulus,
            t_values,
            loops,
            temperature,
            self.training,
        )
        # The edge head is applied once after selecting each row's requested
        # final reducer state. This is algebraically identical to selecting
        # already-computed per-step edges because the head is row-local.
        unary, edges, start, end, output_state = self.square.crf_energies(
            terminal_state, terminal_unary
        )
        trigrams = self._trigram_energies(output_state)
        crf = (unary, edges, trigrams, start, end)
        lengths = valid.sum(dim=1)
        if self.training:
            digit_logits_lsd = register.clamp_min(1e-8).log()
        else:
            path = self._viterbi(*crf)
            path_logits = torch.full(
                (*path.shape, NUM_DIGITS),
                -16.0,
                device=path.device,
                dtype=register.dtype,
            )
            path_logits.scatter_(2, path.unsqueeze(-1), 0.0)
            digit_logits_lsd = path_logits.flip(1)
        logits = self._place_logits(digit_logits_lsd, lengths, prompt_length)
        phase_logits = tuple(
            self._place_logits(phase, lengths, prompt_length)
            for phase in first_square_phases
        )
        auxiliary: dict[str, object] = {
            "t_values": t_values,
            "phase_logits": phase_logits,
            "crf": crf,
            "crf_width": self.width,
        }
        if self.training:
            with torch.no_grad():
                teacher_register, _, _, _ = self._run_square(
                    self.teacher_square,
                    initial_register,
                    modulus,
                    t_values,
                    self.training_outer_steps,
                    self.training_temperature,
                    False,
                )
            auxiliary["teacher_logits"] = self._place_logits(
                teacher_register.clamp_min(1e-8).log(), lengths, prompt_length
            )
            auxiliary["consistency_scale"] = self.consistency_scale
        return logits, auxiliary


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


def _teacher_consistency(student: Tensor, teacher: Tensor, valid: Tensor) -> Tensor:
    temperature = CONSISTENCY_TEMPERATURE
    student_log = F.log_softmax(student.float() / temperature, dim=-1)
    teacher_probability = F.softmax(teacher.float() / temperature, dim=-1)
    losses = F.kl_div(
        student_log, teacher_probability, reduction="none"
    ).sum(dim=-1) * temperature**2
    weights = valid.to(losses.dtype)
    return (losses * weights).sum() / weights.sum().clamp_min(1.0)


def _crf_log_partition(
    unary: Tensor,
    edges: Tensor,
    trigrams: Tensor,
    start: Tensor,
    end: Tensor,
) -> Tensor:
    if unary.shape[1] < 3:
        score = unary[:, 0].float() + start.float()
        for position in range(1, unary.shape[1]):
            score = unary[:, position].float() + torch.logsumexp(
                score[:, :, None] + edges[:, position - 1].float(), dim=1
            )
        return torch.logsumexp(score + end.float(), dim=-1)

    score = (
        unary[:, 0, :, None].float()
        + start[None, :, None].float()
        + edges[:, 0].float()
        + unary[:, 1, None, :].float()
    )
    for position in range(2, unary.shape[1]):
        score = (
            unary[:, position, None, :].float()
            + edges[:, position - 1].float()
            + torch.logsumexp(
                score[:, :, :, None] + trigrams[:, position - 2].float(),
                dim=1,
            )
        )
    return torch.logsumexp(score + end[None, None, :].float(), dim=(1, 2))


def _first_order_log_partition(
    unary: Tensor,
    edges: Tensor,
    start: Tensor,
    end: Tensor,
) -> Tensor:
    """Reference recurrence retained for numerical nesting checks."""

    score = unary[:, 0].float() + start.float()
    for position in range(1, unary.shape[1]):
        score = unary[:, position].float() + torch.logsumexp(
            score[:, :, None] + edges[:, position - 1].float(), dim=1
        )
    return torch.logsumexp(score + end.float(), dim=-1)


def _partial_suffix_crf_nll(
    batch: TokenLossBatch,
    crf: tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
) -> Tensor:
    unary, edges, trigrams, start, end = crf
    width = unary.shape[1]
    valid = batch.valid_mask
    count = valid.sum(dim=1)
    rank = valid.long().cumsum(dim=1) - 1
    destination = (width - count[:, None] + rank).clamp(0, width - 1)
    assignment = F.one_hot(destination, width).bool() & valid.unsqueeze(-1)
    observed = assignment.any(dim=1)
    target_digits = (batch.labels - DIGIT_OFFSET).clamp(0, NUM_DIGITS - 1)
    target_one_hot = F.one_hot(target_digits, NUM_DIGITS).bool()
    selected = (
        assignment.unsqueeze(-1) & target_one_hot.unsqueeze(2)
    ).any(dim=1)
    allowed = ~observed.unsqueeze(-1) | selected
    constrained_unary = unary.float().masked_fill(~allowed, -1e4)
    partition = _crf_log_partition(unary, edges, trigrams, start, end)
    constrained = _crf_log_partition(
        constrained_unary, edges, trigrams, start, end
    )
    return (partition - constrained).mean()


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
    loss = (losses * token_mask).sum() / token_mask.sum().clamp_min(1.0)

    t1_mask = (t_values == 1)[:, None] & valid
    phase_terms = [
        _masked_loss(
            _target_aligned(full_logits, batch), batch.labels, t1_mask
        )
        for full_logits in batch.auxiliary["phase_logits"]
    ]
    if phase_terms:
        loss = loss + 0.35 * torch.stack(phase_terms).mean()

    teacher_logits = _target_aligned(batch.auxiliary["teacher_logits"], batch)
    consistency = _teacher_consistency(batch.logits, teacher_logits, valid)
    scale = float(batch.auxiliary["consistency_scale"])
    crf = _partial_suffix_crf_nll(batch, batch.auxiliary["crf"])
    return loss + EMA_WEIGHT * scale * consistency + CRF_WEIGHT * crf


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
        self.updates = 0

    def step(self) -> None:
        self.updates += 1
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
        self.model.consistency_scale = min(fraction / 0.20, 1.0)
        decay = min(EMA_DECAY, 1.0 - 1.0 / (self.updates + 1.0))
        self.model.update_teacher(decay)


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
        scheduler=WallClockSchedule(optimizer, model, spec.training_time_seconds),
    )


SUBMISSION = Submission(
    build_model=build_model,
    build_optimizer=build_optimizer,
    token_training_loss=token_training_loss,
    batch_size=32,
    eval_batch_size=512,
)
