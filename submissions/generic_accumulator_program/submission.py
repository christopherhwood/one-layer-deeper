"""Endpoint-trained posterior over a small generic modular accumulator ISA.

Four instruction slots each learn one of sixteen choices: a generic opcode
(no-op, add, subtract, multiply) and a right operand (input, one, zero,
modulus).  The accumulator begins at the current recurrent value.  One global
factorized posterior is shared across every example, modulus, instruction,
and recurrence depth.  The forward pass enumerates the complete finite
version space during training and executes only the learned joint MAP program
during evaluation.

No complete program is seeded or stored.  Every posterior logit is randomly
initialized and updated by ordinary SGD from evaluator endpoints.  Training
uses exact marginal likelihood over all programs; there is no genetic search,
program archive, optimizer-side adoption, intermediate arithmetic target, or
participant-controlled training loop.
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
SLOTS = 4
OPCODES = 4
OPERANDS = 4
CHOICES = OPCODES * OPERANDS
PROGRAMS = CHOICES**SLOTS
MAX_OUTER_STEPS = 64
POSTERIOR_LR = 0.3
GLOBAL_WEIGHT = 1.0


class Config:
    def __init__(self, vocab_size: int, max_seq_len: int) -> None:
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len


class Model(nn.Module):
    num_loops = SLOTS

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.decimal_width = max(2, (spec.max_seq_len - 4) // 2)
        self.instruction_logits = nn.Parameter(
            torch.randn(SLOTS, CHOICES) * 0.02
        )
        axes = [torch.arange(CHOICES) for _ in range(SLOTS)]
        self.register_buffer(
            "programs",
            torch.cartesian_prod(*axes),
            persistent=True,
        )

    @staticmethod
    def _parse_number(input_ids: Tensor, mask: Tensor) -> Tensor:
        value = torch.zeros(
            input_ids.shape[0], device=input_ids.device, dtype=torch.long
        )
        for position in range(input_ids.shape[1]):
            digit = (input_ids[:, position] - DIGIT_OFFSET).clamp(0, 9)
            value = torch.where(mask[:, position], value * 10 + digit, value)
        return value

    def _parse(self, input_ids: Tensor, valid: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        digits = (
            valid
            & (input_ids >= DIGIT_OFFSET)
            & (input_ids < DIGIT_OFFSET + NUM_DIGITS)
        )
        after_n = (input_ids == N_MARK).cumsum(dim=1) > 0
        after_x = (input_ids == X_MARK).cumsum(dim=1) > 0
        after_t = (input_ids == T_MARK).cumsum(dim=1) > 0
        modulus = self._parse_number(input_ids, digits & after_n & ~after_x)
        value = self._parse_number(input_ids, digits & after_x & ~after_t)
        time_steps = self._parse_number(
            input_ids, digits & after_t
        ).clamp(min=1, max=MAX_OUTER_STEPS)
        return modulus, value, time_steps

    def _program_log_weights(self) -> Tensor:
        fields = F.log_softmax(self.instruction_logits, dim=-1)
        slots = torch.arange(SLOTS, device=fields.device)[:, None]
        return fields[slots, self.programs.T].sum(dim=0)

    @staticmethod
    def _operand(
        code: Tensor, source: Tensor, modulus: Tensor, state: Tensor
    ) -> Tensor:
        del state
        shape = source.shape
        one = torch.ones(shape, device=source.device, dtype=source.dtype)
        zero = torch.zeros(shape, device=source.device, dtype=source.dtype)
        return torch.where(
            code == 0,
            source,
            torch.where(code == 1, one, torch.where(code == 2, zero, modulus)),
        )

    def _execute_programs(
        self, source: Tensor, modulus: Tensor, depth: int
    ) -> Tensor:
        program = self.programs
        state = source[None].expand(PROGRAMS, -1)
        source_full = source[None].expand_as(state)
        modulus_full = modulus[None].expand_as(state)
        for _ in range(depth):
            for slot in range(SLOTS):
                choice = program[:, slot, None]
                opcode = choice // OPERANDS
                rhs = self._operand(
                    choice.remainder(OPERANDS), source_full, modulus_full, state
                )
                added = torch.remainder(state + rhs, modulus_full)
                subtracted = torch.remainder(state - rhs, modulus_full)
                multiplied = torch.remainder(state * rhs, modulus_full)
                state = torch.where(
                    opcode == 0,
                    state,
                    torch.where(
                        opcode == 1,
                        added,
                        torch.where(opcode == 2, subtracted, multiplied),
                    ),
                )
            source_full = state
        return state

    def _execute_map(
        self, source: Tensor, modulus: Tensor, time_steps: Tensor
    ) -> Tensor:
        selected = self.programs[self._program_log_weights().argmax()]
        state = source
        terminal = torch.zeros_like(source)
        for outer_step in range(int(time_steps.max().item())):
            previous = state
            for slot in range(SLOTS):
                choice = selected[slot]
                opcode = choice // OPERANDS
                code = choice.remainder(OPERANDS)
                rhs = torch.where(
                    code == 0,
                    previous,
                    torch.where(
                        code == 1,
                        torch.ones_like(previous),
                        torch.where(
                            code == 2, torch.zeros_like(previous), modulus
                        ),
                    ),
                )
                added = torch.remainder(state + rhs, modulus)
                subtracted = torch.remainder(state - rhs, modulus)
                multiplied = torch.remainder(state * rhs, modulus)
                state = torch.where(
                    opcode == 0,
                    state,
                    torch.where(
                        opcode == 1,
                        added,
                        torch.where(opcode == 2, subtracted, multiplied),
                    ),
                )
            terminal = torch.where(time_steps == outer_step + 1, state, terminal)
            state = torch.where(time_steps > outer_step, state, previous)
        return terminal

    def _digits(self, value: Tensor) -> Tensor:
        powers = 10 ** torch.arange(
            self.decimal_width, device=value.device
        )
        return (value[:, None] // powers[None]).remainder(10)

    def _hard_digit_logits(self, value: Tensor) -> Tensor:
        digits = self._digits(value)
        logits = self.instruction_logits.new_full(
            (value.shape[0], self.decimal_width, NUM_DIGITS), -16.0
        )
        logits.scatter_(2, digits.unsqueeze(-1), 0.0)
        return logits

    def _mixture_digit_logits(
        self, outputs: Tensor, log_weights: Tensor
    ) -> Tensor:
        probability = log_weights.softmax(dim=0)
        powers = 10 ** torch.arange(
            self.decimal_width, device=outputs.device
        )
        digits = (outputs[:, :, None] // powers[None, None]).remainder(10)
        mixture = probability.new_zeros(
            outputs.shape[1], self.decimal_width, NUM_DIGITS
        )
        for position in range(self.decimal_width):
            mixture[:, position].scatter_add_(
                1,
                digits[:, :, position].T,
                probability[:, None].expand(-1, outputs.shape[1]).T,
            )
        return mixture.clamp_min(1e-9).log()

    def _place_logits(self, digits_lsd: Tensor, lengths: Tensor, prompt: int) -> Tensor:
        digits = digits_lsd.flip(1)
        destination = lengths[:, None] - self.decimal_width + torch.arange(
            self.decimal_width, device=digits.device
        )[None]
        destination = destination.clamp(0, prompt - 1)
        canvas = digits.new_zeros(digits.shape[0], prompt, NUM_DIGITS)
        canvas = canvas.scatter(
            1,
            destination.unsqueeze(-1).expand(-1, -1, NUM_DIGITS),
            digits,
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
    ) -> tuple[Tensor, dict[str, Tensor | int]]:
        if attention_mask is None:
            attention_mask = input_ids != PAD
        valid = attention_mask.bool()
        modulus, source, time_steps = self._parse(input_ids, valid)
        lengths = valid.sum(dim=1)
        if not self.training:
            terminal = self._execute_map(source, modulus, time_steps)
            logits = self._place_logits(
                self._hard_digit_logits(terminal), lengths, input_ids.shape[1]
            )
            return logits, {"time_steps": time_steps}

        depth = int(time_steps.min().item())
        rows = (time_steps == depth).nonzero(as_tuple=False).flatten()
        outputs = self._execute_programs(source[rows], modulus[rows], depth)
        log_weights = self._program_log_weights()
        selected_logits = self._mixture_digit_logits(outputs, log_weights)
        digit_logits = self.instruction_logits.new_zeros(
            input_ids.shape[0], self.decimal_width, NUM_DIGITS
        )
        digit_logits = digit_logits.index_copy(0, rows, selected_logits)
        logits = self._place_logits(digit_logits, lengths, input_ids.shape[1])
        return logits, {
            "program_outputs": outputs,
            "program_log_weights": log_weights,
            "selected_rows": rows,
            "time_steps": time_steps,
        }


def _target_integer(labels: Tensor, valid: Tensor) -> Tensor:
    value = torch.zeros(labels.shape[0], device=labels.device, dtype=torch.long)
    for position in range(labels.shape[1]):
        digit = (labels[:, position] - DIGIT_OFFSET).clamp(0, 9)
        value = torch.where(valid[:, position], value * 10 + digit, value)
    return value


def token_training_loss(batch: TokenLossBatch) -> Tensor:
    rows = batch.auxiliary["selected_rows"]
    row_mask = torch.zeros(
        batch.labels.shape[0], device=batch.labels.device, dtype=torch.float32
    ).index_fill(0, rows, 1.0)
    losses = F.cross_entropy(
        batch.logits.transpose(1, 2),
        batch.labels,
        ignore_index=-100,
        reduction="none",
    )
    token_weight = batch.valid_mask.to(losses.dtype) * row_mask[:, None]
    endpoint = (losses * token_weight).sum() / token_weight.sum().clamp_min(1.0)

    target = _target_integer(batch.labels, batch.valid_mask)[rows]
    outputs = batch.auxiliary["program_outputs"]
    correct = (outputs == target[None]).to(torch.float32)
    # Exact whole-answer likelihood: a mismatch gets a small but finite
    # probability, keeping the differentiable scalar well-conditioned.
    log_likelihood = (correct * -1e-4 + (1.0 - correct) * -12.0).sum(dim=1)
    global_nll = -torch.logsumexp(
        batch.auxiliary["program_log_weights"] + log_likelihood,
        dim=0,
    )
    return endpoint + GLOBAL_WEIGHT * global_nll


class WallClockSchedule:
    def __init__(
        self, optimizer: torch.optim.Optimizer, budget_seconds: float
    ) -> None:
        self.optimizer = optimizer
        self.started_at = time.monotonic()
        self.budget = max(float(budget_seconds) * 0.98, 1.0)

    def step(self) -> None:
        fraction = min((time.monotonic() - self.started_at) / self.budget, 1.0)
        multiplier = 1.0
        if fraction > 0.9:
            progress = (fraction - 0.9) / 0.1
            multiplier = 0.1 + 0.45 * (1.0 + math.cos(math.pi * progress))
        for group in self.optimizer.param_groups:
            group["lr"] = group["base_lr"] * multiplier


def build_model(spec: ModelSpec) -> Model:
    model = Model(spec)
    assert_model_state(model, spec)
    return model


def build_optimizer(model: Model, spec: OptimizerSpec) -> OptimizerBundle:
    optimizer = torch.optim.SGD(
        [{
            "params": [model.instruction_logits],
            "lr": POSTERIOR_LR,
            "base_lr": POSTERIOR_LR,
        }],
        lr=POSTERIOR_LR,
    )
    return OptimizerBundle(
        optimizer=optimizer,
        scheduler=WallClockSchedule(optimizer, spec.training_time_seconds),
    )


SUBMISSION = Submission(
    build_model=build_model,
    build_optimizer=build_optimizer,
    token_training_loss=token_training_loss,
    batch_size=64,
    eval_batch_size=512,
)
