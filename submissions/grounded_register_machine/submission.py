"""Grounded global-program register machine for repeated modular squaring.

Every example executes the same short learned program.  A program phase makes
only observable generic choices: which canonical tape to read twice, which
canonical tape to write, and which direction to scan.  One finite-state local
cell is tied across every digit, program phase, and outer square.  The answer
digit tape is the submitted output and the next square's input; a scratch digit
tape is required to return to the canonical all-zero state at the boundary.

Digit-pair evidence is learned and grouped only by positional significance.
The program, digit transition, and finite controller are randomly initialized
and endpoint-trained.  No multiplication table, carry, quotient, comparison,
or modular-reduction instruction is supplied.
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
READ_TAPES = 4
WRITE_TAPES = 2
DIRECTIONS = 2
CONTROL_STATES = 8
HIDDEN = 64
PROGRAM_STEPS = 6
MAX_OUTER_STEPS = 64
TRAIN_OUTER_STEPS = 3
BASE_LR = 1.8e-3
WORKSPACE_WEIGHT = 0.03
PROGRAM_ENTROPY_WEIGHT = 0.01
CONTROL_MI_WEIGHT = 0.005


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


def _straight_through_choice(
    logits: Tensor,
    temperature: float,
    hardness: float,
) -> tuple[Tensor, Tensor]:
    soft = F.softmax(logits.float() / temperature, dim=-1).to(logits.dtype)
    hard = F.one_hot(soft.argmax(dim=-1), soft.shape[-1]).to(soft.dtype)
    return soft + float(hardness) * (hard - soft).detach(), soft


class LocalTapeCell(nn.Module):
    """A tied finite-state digit transducer used by every program scan."""

    def __init__(self) -> None:
        super().__init__()
        input_width = (
            3 * NUM_DIGITS + HIDDEN + CONTROL_STATES + 2 + DIRECTIONS
        )
        self.input = nn.Linear(input_width, 2 * HIDDEN)
        self.norm = RMSNorm(2 * HIDDEN)
        self.body = nn.Sequential(
            nn.Linear(2 * HIDDEN, 2 * HIDDEN, bias=False),
            nn.SiLU(),
            nn.Linear(2 * HIDDEN, HIDDEN, bias=False),
            nn.SiLU(),
        )
        self.digit = nn.Linear(HIDDEN, NUM_DIGITS)
        self.control = nn.Linear(HIDDEN, CONTROL_STATES)
        self.identity_scale = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        read_a: Tensor,
        read_b: Tensor,
        destination: Tensor,
        product: Tensor,
        control: Tensor,
        boundary: Tensor,
        direction: Tensor,
        temperature: float,
        hardness: float,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        hidden = self.body(
            self.norm(
                self.input(
                    torch.cat(
                        (
                            read_a,
                            read_b,
                            destination,
                            product,
                            control,
                            boundary,
                            direction,
                        ),
                        dim=-1,
                    )
                )
            )
        )
        residual = F.softplus(self.identity_scale)
        digit_logits = self.digit(hidden) + residual * destination
        control_logits = self.control(hidden) + residual * control
        next_digit, digit_soft = _straight_through_choice(
            digit_logits, temperature, hardness
        )
        next_control, control_soft = _straight_through_choice(
            control_logits, temperature, hardness
        )
        return next_digit, next_control, digit_logits, control_soft


class GroundedProgramSquare(nn.Module):
    """One square executed by one input-independent program over digit tapes."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = width
        self.columns = 2 * width - 1
        self.pair_embedding = nn.Parameter(
            torch.empty(NUM_DIGITS, NUM_DIGITS, HIDDEN)
        )
        nn.init.normal_(self.pair_embedding, std=HIDDEN**-0.5)
        self.column_norm = RMSNorm(HIDDEN)
        self.cell = LocalTapeCell()

        # The program is global learned state, never a function of the prompt.
        self.read_a_logits = nn.Parameter(
            torch.empty(PROGRAM_STEPS, READ_TAPES)
        )
        self.read_b_logits = nn.Parameter(
            torch.empty(PROGRAM_STEPS, READ_TAPES)
        )
        self.write_logits = nn.Parameter(
            torch.empty(PROGRAM_STEPS, WRITE_TAPES)
        )
        self.direction_logits = nn.Parameter(
            torch.empty(PROGRAM_STEPS, DIRECTIONS)
        )
        nn.init.normal_(self.read_a_logits, std=0.02)
        nn.init.normal_(self.read_b_logits, std=0.02)
        nn.init.normal_(self.write_logits, std=0.02)
        nn.init.normal_(self.direction_logits, std=0.02)

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
        columns = columns.scatter_add(
            1, significance, pair_features.reshape(batch, -1, HIDDEN)
        )
        return self.column_norm(columns)

    @staticmethod
    def _read(tapes: Tensor, choice: Tensor) -> Tensor:
        return torch.einsum("t,tbwd->bwd", choice, tapes)

    def _both_directions(
        self,
        read_a: Tensor,
        read_b: Tensor,
        destination: Tensor,
        product: Tensor,
        boundary: Tensor,
        temperature: float,
        hardness: float,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Evaluate both scan orders in one doubled-batch sequential pass."""

        batch = destination.shape[0]
        paired_a = torch.cat((read_a, read_a.flip(1)), dim=0)
        paired_b = torch.cat((read_b, read_b.flip(1)), dim=0)
        paired_destination = torch.cat(
            (destination, destination.flip(1)), dim=0
        )
        paired_product = torch.cat((product, product.flip(1)), dim=0)
        paired_boundary = torch.cat((boundary, boundary.flip(1)), dim=0)
        direction = destination.new_zeros(2 * batch, DIRECTIONS)
        direction[:batch, 0] = 1.0
        direction[batch:, 1] = 1.0
        control = destination.new_zeros(2 * batch, CONTROL_STATES)
        control[:, 0] = 1.0

        digits: list[Tensor] = []
        logits: list[Tensor] = []
        control_soft_states: list[Tensor] = []
        for position in range(self.columns):
            digit, control, digit_logits, control_soft = self.cell(
                paired_a[:, position],
                paired_b[:, position],
                paired_destination[:, position],
                paired_product[:, position],
                control,
                paired_boundary[:, position],
                direction,
                temperature,
                hardness,
            )
            digits.append(digit)
            logits.append(digit_logits)
            control_soft_states.append(control_soft)

        digit_tape = torch.stack(digits, dim=1)
        logit_tape = torch.stack(logits, dim=1)
        control_soft = torch.stack(control_soft_states, dim=1)
        forward_digits = digit_tape[:batch]
        reverse_digits = digit_tape[batch:].flip(1)
        forward_logits = logit_tape[:batch]
        reverse_logits = logit_tape[batch:].flip(1)
        return (
            torch.stack((forward_digits, reverse_digits), dim=0),
            torch.stack((forward_logits, reverse_logits), dim=0),
            control_soft,
        )

    def forward(
        self,
        register: Tensor,
        modulus: Tensor,
        state_temperature: float,
        state_hardness: float,
        program_temperature: float,
        program_hardness: float,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        batch = register.shape[0]
        padding = self.columns - self.width
        source = F.pad(register, (0, 0, 0, padding))
        modulus_tape = F.pad(modulus, (0, 0, 0, padding))
        product = self._product_columns(register)
        answer = source
        scratch = register.new_zeros(
            batch, self.columns, NUM_DIGITS
        )
        scratch[:, :, 0] = 1.0
        answer_logits = answer.clamp_min(1e-6).log()
        scratch_logits = scratch.clamp_min(1e-6).log()
        boundary = register.new_zeros(batch, self.columns, 2)
        boundary[:, 0, 0] = 1.0
        boundary[:, -1, 1] = 1.0

        program_entropy_terms: list[Tensor] = []
        control_soft_states: list[Tensor] = []
        for phase in range(PROGRAM_STEPS):
            read_a, read_a_soft = _straight_through_choice(
                self.read_a_logits[phase],
                program_temperature,
                program_hardness,
            )
            read_b, read_b_soft = _straight_through_choice(
                self.read_b_logits[phase],
                program_temperature,
                program_hardness,
            )
            write, write_soft = _straight_through_choice(
                self.write_logits[phase],
                program_temperature,
                program_hardness,
            )
            direction, direction_soft = _straight_through_choice(
                self.direction_logits[phase],
                program_temperature,
                program_hardness,
            )
            program_entropy_terms.extend(
                _entropy(choice)
                for choice in (
                    read_a_soft,
                    read_b_soft,
                    write_soft,
                    direction_soft,
                )
            )

            tapes = torch.stack((answer, scratch, source, modulus_tape), dim=0)
            first_read = self._read(tapes, read_a)
            second_read = self._read(tapes, read_b)
            destination = write[0] * answer + write[1] * scratch
            scan_digits, scan_logits, control_soft = self._both_directions(
                first_read,
                second_read,
                destination,
                product,
                boundary,
                state_temperature,
                state_hardness,
            )
            control_soft_states.append(control_soft)
            selected_digits = (
                direction[0] * scan_digits[0]
                + direction[1] * scan_digits[1]
            )
            selected_logits = (
                direction[0] * scan_logits[0]
                + direction[1] * scan_logits[1]
            )
            answer = write[0] * selected_digits + write[1] * answer
            scratch = write[1] * selected_digits + write[0] * scratch
            answer_logits = (
                write[0] * selected_logits + write[1] * answer_logits
            )
            scratch_logits = (
                write[1] * selected_logits + write[0] * scratch_logits
            )

        return (
            answer[:, : self.width],
            answer_logits[:, : self.width],
            scratch_logits,
            torch.stack(program_entropy_terms),
            torch.stack(control_soft_states, dim=1),
        )


class Model(nn.Module):
    num_loops = PROGRAM_STEPS

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.max_length = spec.max_seq_len
        self.width = max(2, (spec.max_seq_len - 5) // 2)
        self.square = GroundedProgramSquare(self.width)
        self.state_temperature = 1.25
        self.program_temperature = 1.50
        self.state_hardness = 0.05
        self.program_hardness = 0.05
        self.workspace_scale = 0.0
        self.program_scale = 0.0

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
            batch, self.width, 1, device=input_ids.device, dtype=torch.float32
        )
        register.scatter_add_(
            1, destination.unsqueeze(-1).expand(-1, -1, NUM_DIGITS), source
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
        n_mask = digits & after_n & ~after_x
        x_mask = digits & after_x & ~after_t
        return (
            self._register(input_ids, n_mask),
            self._register(input_ids, x_mask),
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
        state_hardness = self.state_hardness if self.training else 1.0
        program_hardness = self.program_hardness if self.training else 1.0
        state_temperature = self.state_temperature if self.training else 0.20
        program_temperature = (
            self.program_temperature if self.training else 0.10
        )

        terminal_logits = torch.zeros_like(register)
        terminal_workspace: Tensor | None = None
        first_program: Tensor | None = None
        first_control: Tensor | None = None
        for outer_step in range(loops):
            transition_input = (
                register.detach()
                if self.training and outer_step > 0
                else register
            )
            (
                candidate,
                candidate_logits,
                workspace_logits,
                program_soft,
                control_soft,
            ) = self.square(
                transition_input,
                modulus,
                state_temperature,
                state_hardness,
                program_temperature,
                program_hardness,
            )
            if outer_step == 0:
                first_program = program_soft
                first_control = control_soft
                terminal_workspace = torch.zeros_like(workspace_logits)
            terminal = (t_values == outer_step + 1)[:, None, None]
            terminal_logits = torch.where(
                terminal, candidate_logits, terminal_logits
            )
            if terminal_workspace is None:
                raise RuntimeError("workspace was not initialized")
            terminal_workspace = torch.where(
                terminal, workspace_logits, terminal_workspace
            )
            active = (t_values > outer_step)[:, None, None]
            register = torch.where(active, candidate, register)

        if (
            first_program is None
            or first_control is None
            or terminal_workspace is None
        ):
            raise RuntimeError("at least one square is required")
        lengths = valid.sum(dim=1)
        logits = self._place_logits(
            terminal_logits, lengths, prompt_length
        )
        return logits, {
            "t_values": t_values,
            "workspace_logits": terminal_workspace,
            "program_entropy": first_program,
            "control_soft": first_control,
            "workspace_scale": self.workspace_scale,
            "program_scale": self.program_scale,
        }


def _entropy(probability: Tensor) -> Tensor:
    probability = probability.float()
    return -(
        probability.clamp_min(1e-8).log() * probability
    ).sum(dim=-1)


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
    endpoint_mask = valid.float() * row_weight[:, None]
    loss = (
        (endpoint * endpoint_mask).sum()
        / endpoint_mask.sum().clamp_min(1.0)
    )

    workspace_logits = batch.auxiliary["workspace_logits"].float()
    blank = torch.zeros(
        workspace_logits.shape[:-1],
        device=workspace_logits.device,
        dtype=torch.long,
    )
    workspace = F.cross_entropy(
        workspace_logits.flatten(0, -2), blank.flatten()
    )

    program_entropy = batch.auxiliary["program_entropy"].float().mean()
    control = batch.auxiliary["control_soft"].float()
    local_control_entropy = _entropy(control).mean()
    aggregate_control = control.mean(dim=(0, 1, 2))
    control_mi = local_control_entropy - _entropy(aggregate_control)
    return (
        loss
        + WORKSPACE_WEIGHT
        * float(batch.auxiliary["workspace_scale"])
        * workspace
        + PROGRAM_ENTROPY_WEIGHT
        * float(batch.auxiliary["program_scale"])
        * program_entropy
        + CONTROL_MI_WEIGHT * control_mi
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

        crystallization = min(
            1.0, max(0.0, (fraction - 0.78) / 0.17)
        )
        self.model.state_temperature = 1.25 * (
            0.30 / 1.25
        ) ** crystallization
        self.model.program_temperature = 1.50 * (
            0.20 / 1.50
        ) ** crystallization
        self.model.state_hardness = 0.05 + 0.95 * crystallization
        self.model.program_hardness = 0.05 + 0.95 * crystallization
        self.model.workspace_scale = min(
            1.0, max(0.0, (fraction - 0.20) / 0.25)
        )
        self.model.program_scale = min(
            1.0, max(0.0, (fraction - 0.78) / 0.17)
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
            {"params": decay, "weight_decay": 0.05, "base_lr": BASE_LR},
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
