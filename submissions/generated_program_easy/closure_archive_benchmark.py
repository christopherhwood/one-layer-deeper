"""Benchmark-native bridge for the general semantic-closure archive."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from benchmark import (
    ModelSpec,
    OptimizerBundle,
    OptimizerSpec,
    Submission,
    TokenLossBatch,
    assert_model_state,
)
from submissions.generated_program_easy.generality_probe import (
    _prefix_fraction,
)
from submissions.generated_program_easy.register_machine_synthesis import (
    CAP,
    safe_multiply,
)
from submissions.generated_program_easy.semantic_closure_synthesis import (
    OPERATIONS,
    Expression,
    build_archive,
)


PAD = 0
N_MARK = 2
X_MARK = 3
T_MARK = 4
DIGIT_OFFSET = 7
NUM_DIGITS = 10
SLOTS = 4
MAX_OUTER_STEPS = 64


def _compile(expression: Expression) -> tuple[list[int], list[int], list[int], int]:
    opcode: list[int] = []
    source_a: list[int] = []
    source_b: list[int] = []
    cache: dict[Expression, int] = {}
    terminal = {"x": 0, "N": 1, "0": 2, "1": 3}

    def visit(node: Expression) -> int:
        if node.left is None:
            return terminal[node.operation]
        if node in cache:
            return cache[node]
        left = visit(node.left)
        assert node.right is not None
        right = visit(node.right)
        destination = 4 + len(opcode)
        opcode.append(OPERATIONS.index(node.operation))
        source_a.append(left)
        source_b.append(right)
        cache[node] = destination
        return destination

    output = visit(expression)
    if len(opcode) > SLOTS:
        raise AssertionError(f"expression requires {len(opcode)} slots: {expression}")
    while len(opcode) < SLOTS:
        opcode.append(0)
        source_a.append(2)
        source_b.append(2)
    return opcode, source_a, source_b, output


def _archive_tensors():
    _, _, levels, archive = build_archive(SLOTS)
    compiled = [_compile(expression) for expression in archive.values()]
    opcode = torch.tensor([row[0] for row in compiled], dtype=torch.long)
    source_a = torch.tensor([row[1] for row in compiled], dtype=torch.long)
    source_b = torch.tensor([row[2] for row in compiled], dtype=torch.long)
    output = torch.tensor([row[3] for row in compiled], dtype=torch.long)
    boundaries = torch.tensor(
        [sum(len(level) for level in levels[: index + 1]) for index in range(len(levels))],
        dtype=torch.long,
    )
    return opcode, source_a, source_b, output, boundaries


class Config:
    def __init__(self, vocab_size: int, max_seq_len: int) -> None:
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len


class Model(nn.Module):
    num_loops = 1

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.decimal_width = max(2, (spec.max_seq_len - 4) // 2)
        opcode, source_a, source_b, output, boundaries = _archive_tensors()
        self.register_buffer("opcode", opcode)
        self.register_buffer("source_a", source_a)
        self.register_buffer("source_b", source_b)
        self.register_buffer("output_register", output)
        self.register_buffer("level_boundaries", boundaries)
        self.register_buffer("growth_level", torch.zeros((), dtype=torch.long))
        self.register_buffer("active_count", boundaries[0].clone())
        self.program_logits = nn.Parameter(torch.zeros(opcode.shape[0]))

    @staticmethod
    def _parse_number(input_ids: Tensor, mask: Tensor) -> Tensor:
        value = torch.zeros(input_ids.shape[0], device=input_ids.device, dtype=torch.long)
        for position in range(input_ids.shape[1]):
            digit = (input_ids[:, position] - DIGIT_OFFSET).clamp(0, 9)
            value = torch.where(mask[:, position], value * 10 + digit, value)
        return value

    def _parse(self, input_ids: Tensor, valid: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        digits = valid & (input_ids >= DIGIT_OFFSET) & (
            input_ids < DIGIT_OFFSET + NUM_DIGITS
        )
        after_n = (input_ids == N_MARK).cumsum(dim=1) > 0
        after_x = (input_ids == X_MARK).cumsum(dim=1) > 0
        after_t = (input_ids == T_MARK).cumsum(dim=1) > 0
        modulus = self._parse_number(input_ids, digits & after_n & ~after_x)
        value = self._parse_number(input_ids, digits & after_x & ~after_t)
        time_steps = self._parse_number(input_ids, digits & after_t).clamp(
            min=1, max=MAX_OUTER_STEPS
        )
        return modulus, value, time_steps

    @staticmethod
    def _gather(registers: Tensor, source: Tensor) -> Tensor:
        return registers.gather(
            2, source[:, None, None].expand(-1, registers.shape[1], 1)
        ).squeeze(2)

    def _execute_once(self, value: Tensor, modulus: Tensor, count: int) -> Tensor:
        if value.ndim == 1:
            value = value[None].expand(count, -1)
        if modulus.ndim == 1:
            modulus = modulus[None].expand(count, -1)
        registers = torch.stack(
            (
                value,
                modulus,
                torch.zeros_like(value),
                torch.ones_like(value),
            ),
            dim=2,
        )
        for slot in range(SLOTS):
            left = self._gather(registers, self.source_a[:count, slot])
            right = self._gather(registers, self.source_b[:count, slot])
            denominator = right.clamp_min(1)
            candidates = torch.stack(
                (
                    (left + right).clamp(max=CAP),
                    (left - right).clamp(min=0),
                    safe_multiply(left, right),
                    torch.div(left, denominator, rounding_mode="floor"),
                    left.remainder(denominator),
                    torch.bitwise_and(left, right),
                    torch.minimum(left, right),
                ),
                dim=2,
            )
            selected = candidates.gather(
                2,
                self.opcode[:count, slot, None, None].expand(
                    -1, registers.shape[1], 1
                ),
            ).squeeze(2)
            registers = torch.cat((registers, selected[:, :, None]), dim=2)
        return self._gather(registers, self.output_register[:count])

    def _execute_selected(self, value: Tensor, modulus: Tensor, selected: int) -> Tensor:
        registers = [value, modulus, torch.zeros_like(value), torch.ones_like(value)]
        for slot in range(SLOTS):
            left = registers[int(self.source_a[selected, slot])]
            right = registers[int(self.source_b[selected, slot])]
            denominator = right.clamp_min(1)
            operation = int(self.opcode[selected, slot])
            candidates = (
                (left + right).clamp(max=CAP),
                (left - right).clamp(min=0),
                safe_multiply(left, right),
                torch.div(left, denominator, rounding_mode="floor"),
                left.remainder(denominator),
                torch.bitwise_and(left, right),
                torch.minimum(left, right),
            )
            registers.append(candidates[operation])
        return registers[int(self.output_register[selected])]

    def _place_logits(self, value: Tensor, lengths: Tensor, prompt: int) -> Tensor:
        powers = 10 ** torch.arange(self.decimal_width, device=value.device)
        digits = ((value[:, None] // powers[None]).remainder(10)).flip(1)
        destination = lengths[:, None] - self.decimal_width + torch.arange(
            self.decimal_width, device=value.device
        )[None]
        destination = destination.clamp(0, prompt - 1)
        canvas = torch.zeros(value.shape[0], prompt, NUM_DIGITS, device=value.device)
        canvas.scatter_(
            1,
            destination[:, :, None].expand(-1, -1, NUM_DIGITS),
            torch.nn.functional.one_hot(digits, NUM_DIGITS).float(),
        )
        return torch.nn.functional.pad(
            canvas.clamp_min(1e-7).log(),
            (DIGIT_OFFSET, self.config.vocab_size - DIGIT_OFFSET - NUM_DIGITS),
            value=-16.0,
        )

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None):
        if attention_mask is None:
            attention_mask = input_ids != PAD
        valid = attention_mask.bool()
        modulus, value, time_steps = self._parse(input_ids, valid)
        count = int(self.active_count)
        active_logits = self.program_logits[:count]
        if not self.training:
            selected = int(active_logits.argmax())
            state = value
            terminal = torch.zeros_like(value)
            for step in range(int(time_steps.max())):
                candidate = self._execute_selected(state, modulus, selected)
                terminal = torch.where(time_steps == step + 1, candidate, terminal)
                state = torch.where(time_steps > step, candidate, state)
            return self._place_logits(
                terminal, valid.sum(dim=1), input_ids.shape[1]
            ), {"selected_program": selected}

        depth = int(time_steps.min())
        state = value
        for _ in range(depth):
            state = self._execute_once(state, modulus, count)
        logits = self.program_logits.new_zeros(
            input_ids.shape[0], input_ids.shape[1], self.config.vocab_size
        )
        return logits, {
            "program_outputs": state,
            "program_log_weights": active_logits.log_softmax(dim=0),
            "modulus": modulus,
            "supervision_mask": time_steps == depth,
        }


def _target_integer(labels: Tensor, valid: Tensor) -> Tensor:
    value = torch.zeros(labels.shape[0], device=labels.device, dtype=torch.long)
    for position in range(labels.shape[1]):
        digit = (labels[:, position] - DIGIT_OFFSET).clamp(0, 9)
        value = torch.where(valid[:, position], value * 10 + digit, value)
    return value


def token_training_loss(batch: TokenLossBatch) -> Tensor:
    output = batch.auxiliary["program_outputs"]
    target = _target_integer(batch.labels, batch.valid_mask)
    supervision = batch.auxiliary["supervision_mask"].float()
    exact_error = (output != target[None]).float()
    prefix_error = 1.0 - _prefix_fraction(output, target)
    modulus = batch.auxiliary["modulus"][None].clamp_min(2)
    distance = (output.remainder(modulus) - target[None].remainder(modulus)).abs()
    distance = torch.minimum(distance, modulus - distance)
    numeric_error = (distance.float() / (modulus // 2).clamp_min(1)).clamp(max=1.0)
    width = max(1, int(max(int(output.max()), int(target.max()))).bit_length())
    positions = torch.arange(width, device=output.device)
    bit_error = (
        ((output[:, :, None] >> positions) & 1)
        != ((target[None, :, None] >> positions) & 1)
    ).float().mean(dim=2)
    penalty = 8.0 * exact_error + 2.0 * prefix_error + numeric_error + bit_error
    per_program = (penalty * supervision[None]).sum(dim=1) / supervision.sum().clamp_min(1)
    return (batch.auxiliary["program_log_weights"].exp() * per_program).sum()


class GrowthOptimizer:
    def __init__(self, model: Model) -> None:
        self.model = model
        self.inner = torch.optim.AdamW([model.program_logits], lr=0.5, weight_decay=0.0)
        self.param_groups = self.inner.param_groups
        self.state = self.inner.state
        self.optimizers = [self.inner]

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.inner.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return self.inner.state_dict()

    @torch.no_grad()
    def step(self, closure=None):
        result = self.inner.step(closure)
        level = int(self.model.growth_level)
        if level + 1 < self.model.level_boundaries.numel():
            self.model.growth_level.add_(1)
            self.model.active_count.copy_(self.model.level_boundaries[level + 1])
            self.model.program_logits.zero_()
            for value in self.inner.state.get(self.model.program_logits, {}).values():
                if torch.is_tensor(value) and value.shape == self.model.program_logits.shape:
                    value.zero_()
        return result


def build_model(spec: ModelSpec) -> Model:
    model = Model(spec)
    assert_model_state(model, spec)
    return model


def build_optimizer(model: Model, spec: OptimizerSpec) -> OptimizerBundle:
    del spec
    return OptimizerBundle(optimizer=GrowthOptimizer(model))


SUBMISSION = Submission(
    build_model=build_model,
    build_optimizer=build_optimizer,
    token_training_loss=token_training_loss,
    batch_size=8,
    eval_batch_size=512,
)
