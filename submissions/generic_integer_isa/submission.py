"""Endpoint-trained posterior over a generic integer accumulator ISA.

Five instruction slots select from sixteen ordinary calculator operations.
Modulo is an explicit learned instruction; the architecture does not reduce
automatically after arithmetic and does not encode a recurrence formula.
One factorized global posterior is learned from evaluator endpoints by exact
marginal likelihood over the complete 16^5 straight-line program space.
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
SLOTS = 5
CHOICES = 16
PROGRAMS = CHOICES**SLOTS
MAX_OUTER_STEPS = 64
CAP = (1 << 62) - 1
POSTERIOR_LR = 0.3


class Config:
    def __init__(self, vocab_size: int, max_seq_len: int) -> None:
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len


def _safe_add(left: Tensor, right: Tensor) -> Tensor:
    return left + torch.minimum(right, CAP - left)


def _safe_multiply(left: Tensor, right: Tensor) -> Tensor:
    safe_right = torch.minimum(right, CAP // left.clamp_min(1))
    return left * safe_right


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
            "programs", torch.cartesian_prod(*axes), persistent=True
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
        modulus = self._parse_number(input_ids, digits & after_n & ~after_x)
        source = self._parse_number(input_ids, digits & after_x & ~after_t)
        steps = self._parse_number(input_ids, digits & after_t).clamp(
            1, MAX_OUTER_STEPS
        )
        return modulus, source, steps

    def _program_log_weights(self) -> Tensor:
        fields = F.log_softmax(self.instruction_logits, dim=-1)
        slots = torch.arange(SLOTS, device=fields.device)[:, None]
        return fields[slots, self.programs.T].sum(dim=0)

    @staticmethod
    def _instruction(
        state: Tensor, source: Tensor, modulus: Tensor, code: Tensor
    ) -> Tensor:
        one = torch.ones_like(state)
        denominator_source = source.clamp_min(1)
        denominator_modulus = modulus.clamp_min(1)
        choices = (
            state,
            _safe_add(state, source),
            (state - source).clamp_min(0),
            _safe_multiply(state, source),
            _safe_add(state, one),
            (state - one).clamp_min(0),
            _safe_add(state, state),
            torch.div(state, denominator_source, rounding_mode="floor"),
            state.remainder(denominator_source),
            state.remainder(denominator_modulus),
            torch.bitwise_xor(state, source),
            torch.bitwise_and(state, source),
            torch.minimum(state, source),
            torch.maximum(state, source),
            source,
            torch.zeros_like(state),
        )
        output = choices[0]
        for index in range(1, CHOICES):
            output = torch.where(code == index, choices[index], output)
        return output

    def _execute_programs(
        self, source: Tensor, modulus: Tensor, depth: int
    ) -> Tensor:
        state = source[None].expand(PROGRAMS, -1)
        modulus_full = modulus[None].expand_as(state)
        for _ in range(depth):
            previous = state
            for slot in range(SLOTS):
                state = self._instruction(
                    state,
                    previous,
                    modulus_full,
                    self.programs[:, slot, None],
                )
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
                code = selected[slot]
                one = torch.ones_like(state)
                choices = (
                    state,
                    _safe_add(state, previous),
                    (state - previous).clamp_min(0),
                    _safe_multiply(state, previous),
                    _safe_add(state, one),
                    (state - one).clamp_min(0),
                    _safe_add(state, state),
                    torch.div(state, previous.clamp_min(1), rounding_mode="floor"),
                    state.remainder(previous.clamp_min(1)),
                    state.remainder(modulus.clamp_min(1)),
                    torch.bitwise_xor(state, previous),
                    torch.bitwise_and(state, previous),
                    torch.minimum(state, previous),
                    torch.maximum(state, previous),
                    previous,
                    torch.zeros_like(state),
                )
                next_state = choices[0]
                for index in range(1, CHOICES):
                    next_state = torch.where(code == index, choices[index], next_state)
                state = next_state
            terminal = torch.where(time_steps == outer_step + 1, state, terminal)
            state = torch.where(time_steps > outer_step, state, previous)
        return terminal

    def _digits(self, value: Tensor) -> Tensor:
        powers = 10 ** torch.arange(self.decimal_width, device=value.device)
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
        powers = 10 ** torch.arange(self.decimal_width, device=outputs.device)
        digits = (outputs[:, :, None] // powers[None, None]).remainder(10)
        mixture = probability.new_zeros(
            outputs.shape[1], self.decimal_width, NUM_DIGITS
        )
        weights = probability[:, None].expand(-1, outputs.shape[1]).T
        for position in range(self.decimal_width):
            mixture[:, position].scatter_add_(
                1, digits[:, :, position].T, weights
            )
        return mixture.clamp_min(1e-9).log()

    def _place_logits(
        self, digit_logits_lsd: Tensor, lengths: Tensor, prompt: int
    ) -> Tensor:
        digits = digit_logits_lsd.flip(1)
        destination = lengths[:, None] - self.decimal_width + torch.arange(
            self.decimal_width, device=digits.device
        )[None]
        destination = destination.clamp(0, prompt - 1)
        canvas = digits.new_zeros(
            digits.shape[0], prompt, NUM_DIGITS
        ).scatter(
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
    ) -> tuple[Tensor, dict[str, object]]:
        if attention_mask is None:
            attention_mask = input_ids != PAD
        valid = attention_mask.bool()
        modulus, source, time_steps = self._parse(input_ids, valid)
        lengths = valid.sum(dim=1)
        if not self.training:
            output = self._execute_map(source, modulus, time_steps)
            return self._place_logits(
                self._hard_digit_logits(output), lengths, input_ids.shape[1]
            ), {"time_steps": time_steps}

        depth = int(time_steps.min().item())
        rows = (time_steps == depth).nonzero(as_tuple=False).flatten()
        outputs = self._execute_programs(source[rows], modulus[rows], depth)
        log_weights = self._program_log_weights()
        selected = self._mixture_digit_logits(outputs, log_weights)
        digit_logits = self.instruction_logits.new_zeros(
            input_ids.shape[0], self.decimal_width, NUM_DIGITS
        ).index_copy(0, rows, selected)
        return self._place_logits(
            digit_logits, lengths, input_ids.shape[1]
        ), {
            "program_outputs": outputs,
            "program_log_weights": log_weights,
            "selected_rows": rows,
        }


def _target_integer(labels: Tensor, valid: Tensor) -> Tensor:
    value = torch.zeros(labels.shape[0], device=labels.device, dtype=torch.long)
    for position in range(labels.shape[1]):
        digit = (labels[:, position] - DIGIT_OFFSET).clamp(0, 9)
        value = torch.where(valid[:, position], value * 10 + digit, value)
    return value


def token_training_loss(batch: TokenLossBatch) -> Tensor:
    rows = batch.auxiliary["selected_rows"]
    target = _target_integer(batch.labels, batch.valid_mask)[rows]
    outputs = batch.auxiliary["program_outputs"]
    correct = (outputs == target[None]).to(torch.float32)
    log_likelihood = (correct * -1e-4 + (1.0 - correct) * -12.0).sum(1)
    return -torch.logsumexp(
        batch.auxiliary["program_log_weights"] + log_likelihood, dim=0
    )


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
    batch_size=8,
    eval_batch_size=512,
)
