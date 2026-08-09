"""Endpoint-trained autoregressive writer for short integer programs.

The writer samples one instruction at a time from a learned first-order
autoregressive distribution.  Sampled programs are executed on the current
endpoint batch and the best endpoint-consistent sample supplies an ordinary
score-function/cross-entropy gradient through its sequence log probability.
No complete program space, program table, archive, or winning program is
stored or enumerated.
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
SAMPLES = 16_384
MAX_OUTER_STEPS = 64
CAP = (1 << 62) - 1
WRITER_LR = 1.0
ENTROPY_WEIGHT = 1e-4


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
        self.start_logits = nn.Parameter(torch.randn(CHOICES) * 0.02)
        self.transition_logits = nn.Parameter(
            torch.randn(SLOTS - 1, CHOICES, CHOICES) * 0.02
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

    @staticmethod
    def _gumbel_sample(log_probabilities: Tensor) -> Tensor:
        uniform = torch.rand_like(log_probabilities).clamp_(1e-6, 1.0 - 1e-6)
        gumbel = -torch.log(-torch.log(uniform))
        return (log_probabilities + gumbel).argmax(dim=-1)

    def _sample_programs(self) -> tuple[Tensor, Tensor]:
        start = F.log_softmax(self.start_logits.float(), dim=-1)
        first_distribution = start[None].expand(SAMPLES, -1)
        previous = self._gumbel_sample(first_distribution)
        instructions = [previous]
        sequence_log_probability = start[previous]
        transitions = F.log_softmax(self.transition_logits.float(), dim=-1)
        for slot in range(1, SLOTS):
            conditional = transitions[slot - 1, previous]
            current = self._gumbel_sample(conditional)
            sequence_log_probability = (
                sequence_log_probability
                + transitions[slot - 1, previous, current]
            )
            instructions.append(current)
            previous = current
        return torch.stack(instructions, dim=1), sequence_log_probability

    def _map_program(self) -> Tensor:
        start = F.log_softmax(self.start_logits.float(), dim=-1)
        transitions = F.log_softmax(self.transition_logits.float(), dim=-1)
        scores = start
        backpointers = []
        for slot in range(1, SLOTS):
            candidates = scores[:, None] + transitions[slot - 1]
            scores, previous = candidates.max(dim=0)
            backpointers.append(previous)
        current = scores.argmax()
        reversed_program = [current]
        for previous in reversed(backpointers):
            current = previous[current]
            reversed_program.append(current)
        return torch.stack(list(reversed(reversed_program)))

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
        self,
        programs: Tensor,
        source: Tensor,
        modulus: Tensor,
        depth: int,
    ) -> Tensor:
        state = source[None].expand(programs.shape[0], -1)
        modulus_full = modulus[None].expand_as(state)
        for _ in range(depth):
            previous = state
            for slot in range(SLOTS):
                state = self._instruction(
                    state,
                    previous,
                    modulus_full,
                    programs[:, slot, None],
                )
        return state

    def _execute_map(
        self, source: Tensor, modulus: Tensor, time_steps: Tensor
    ) -> Tensor:
        program = self._map_program()
        state = source
        terminal = torch.zeros_like(source)
        for outer_step in range(int(time_steps.max().item())):
            previous = state
            for slot in range(SLOTS):
                state = self._instruction(
                    state, previous, modulus, program[slot]
                )
            terminal = torch.where(time_steps == outer_step + 1, state, terminal)
            state = torch.where(time_steps > outer_step, state, previous)
        return terminal

    def _digits(self, value: Tensor) -> Tensor:
        powers = 10 ** torch.arange(self.decimal_width, device=value.device)
        return (value[:, None] // powers[None]).remainder(10)

    def _hard_digit_logits(self, value: Tensor) -> Tensor:
        digits = self._digits(value)
        logits = self.start_logits.new_full(
            (value.shape[0], self.decimal_width, NUM_DIGITS), -16.0
        )
        logits.scatter_(2, digits.unsqueeze(-1), 0.0)
        return logits

    def _sample_digit_logits(self, outputs: Tensor) -> Tensor:
        powers = 10 ** torch.arange(self.decimal_width, device=outputs.device)
        digits = (outputs[:, :, None] // powers[None, None]).remainder(10)
        mixture = self.start_logits.new_zeros(
            outputs.shape[1], self.decimal_width, NUM_DIGITS
        )
        weights = self.start_logits.new_full(
            (outputs.shape[1], outputs.shape[0]), 1.0 / outputs.shape[0]
        )
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
            ), {"time_steps": time_steps, "program": self._map_program()}

        depth = int(time_steps.min().item())
        rows = (time_steps == depth).nonzero(as_tuple=False).flatten()
        programs, sequence_log_probability = self._sample_programs()
        outputs = self._execute_programs(
            programs, source[rows], modulus[rows], depth
        )
        selected = self._sample_digit_logits(outputs)
        digit_logits = self.start_logits.new_zeros(
            input_ids.shape[0], self.decimal_width, NUM_DIGITS
        ).index_copy(0, rows, selected)
        return self._place_logits(
            digit_logits, lengths, input_ids.shape[1]
        ), {
            "program_outputs": outputs,
            "programs": programs,
            "sequence_log_probability": sequence_log_probability,
            "selected_rows": rows,
            "decimal_width": self.decimal_width,
            "start_logits": self.start_logits,
            "transition_logits": self.transition_logits,
        }


def _target_integer(labels: Tensor, valid: Tensor) -> Tensor:
    value = torch.zeros(labels.shape[0], device=labels.device, dtype=torch.long)
    for position in range(labels.shape[1]):
        digit = (labels[:, position] - DIGIT_OFFSET).clamp(0, 9)
        value = torch.where(valid[:, position], value * 10 + digit, value)
    return value


def token_training_loss(batch: TokenLossBatch) -> Tensor:
    auxiliary = batch.auxiliary
    rows = auxiliary["selected_rows"]
    target = _target_integer(batch.labels, batch.valid_mask)[rows]
    outputs = auxiliary["program_outputs"]
    width = auxiliary["decimal_width"]
    powers = 10 ** torch.arange(width, device=outputs.device)
    output_digits = (outputs[:, :, None] // powers[None, None]).remainder(10)
    target_digits = (target[:, None] // powers[None]).remainder(10)
    exact_rows = (outputs == target[None]).sum(dim=1)
    matching_digits = (output_digits == target_digits[None]).sum(dim=(1, 2))
    distance = (outputs - target[None]).abs().float()
    closeness = -torch.log1p(distance).mean(dim=1)
    quality = (
        exact_rows.float() * (target.numel() * width + 1)
        + matching_digits.float()
        + 0.01 * closeness
    )
    sequence_log_probability = auxiliary["sequence_log_probability"]
    winner = (quality + 1e-4 * sequence_log_probability.detach()).argmax()
    policy_loss = -sequence_log_probability[winner]

    start_log = F.log_softmax(auxiliary["start_logits"].float(), dim=-1)
    transition_log = F.log_softmax(
        auxiliary["transition_logits"].float(), dim=-1
    )
    start_probability = start_log.exp()
    transition_probability = transition_log.exp()
    negative_entropy = (start_probability * start_log).mean()
    negative_entropy = negative_entropy + (
        transition_probability * transition_log
    ).mean()
    return policy_loss + ENTROPY_WEIGHT * negative_entropy


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
            "params": [model.start_logits, model.transition_logits],
            "lr": WRITER_LR,
            "base_lr": WRITER_LR,
        }],
        lr=WRITER_LR,
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
