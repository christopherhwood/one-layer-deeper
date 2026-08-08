"""Standalone benchmark-native growth of straight-line recurrence programs.

The persistent state contains capacity for programs but no nonterminal program
at construction time.  Optimizer steps retain endpoint-ranked parents, append
one instruction on GPU, and overwrite the remaining inactive/pruned rows.
"""

from __future__ import annotations

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
OPERATIONS = ("MOVE", "ADD", "SUB", "MUL", "DIV", "MOD", "AND", "MIN")
POPULATION = 65_536
SLOTS = 10
MAX_PARENTS = 512
CHILDREN_PER_PARENT = 126
CHAIN_CHILDREN = 64
ARBITRARY_OUTPUT_CHILDREN = 32
SCORE_PASSES_PER_GROWTH = 2
MAX_OUTER_STEPS = 64
CAP = (1 << 62) - 1


def safe_multiply(left: Tensor, right: Tensor) -> Tensor:
    """Nonnegative saturating int64 multiply without evaluating overflow."""
    safe_right = torch.minimum(right, CAP // left.clamp_min(1))
    return left * safe_right


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
        self.bit_width = min(62, max(8, 4 * self.decimal_width))
        # Inactive capacity uses harmless sentinel syntax.  Only the first four
        # rows are live, and they are exactly the four terminal return values.
        self.register_buffer("opcode", torch.zeros(POPULATION, SLOTS, dtype=torch.long))
        self.register_buffer("source_a", torch.full((POPULATION, SLOTS), 2, dtype=torch.long))
        self.register_buffer("source_b", torch.full((POPULATION, SLOTS), 2, dtype=torch.long))
        self.register_buffer("length", torch.zeros(POPULATION, dtype=torch.long))
        output = torch.full((POPULATION,), 2, dtype=torch.long)
        output[:4] = torch.arange(4)
        self.register_buffer("output_register", output)
        self.register_buffer("active_count", torch.tensor(4, dtype=torch.long))
        self.register_buffer("growth_generation", torch.zeros((), dtype=torch.long))
        self.register_buffer("generated_programs", torch.zeros((), dtype=torch.long))
        self.register_buffer("pruned_programs", torch.zeros((), dtype=torch.long))
        self.register_buffer("fitness_ema", torch.zeros(POPULATION))
        self.register_buffer("score_passes", torch.zeros((), dtype=torch.long))
        self.register_buffer("perfect_hits", torch.zeros(POPULATION, dtype=torch.long))
        self.register_buffer("solution_found", torch.zeros((), dtype=torch.long))
        self.program_logits = nn.Parameter(torch.zeros(POPULATION))
        # The custom loss places a zero/nonzero observational-duplicate bit in
        # this parameter's ordinary gradient.  The optimizer uses that gradient
        # only to avoid retaining multiple programs with identical executions.
        self.novelty_probe = nn.Parameter(torch.zeros(POPULATION))
        self.fitness_probe = nn.Parameter(torch.zeros(POPULATION))

    @staticmethod
    def _parse_number(input_ids: Tensor, mask: Tensor) -> Tensor:
        value = torch.zeros(input_ids.shape[0], device=input_ids.device, dtype=torch.long)
        for position in range(input_ids.shape[1]):
            digit = (input_ids[:, position] - DIGIT_OFFSET).clamp(0, 9)
            value = torch.where(mask[:, position], value * 10 + digit, value)
        return value

    def _parse(self, input_ids: Tensor, valid: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        digits = valid & (input_ids >= DIGIT_OFFSET) & (input_ids < DIGIT_OFFSET + 10)
        after_n = (input_ids == N_MARK).cumsum(dim=1) > 0
        after_x = (input_ids == X_MARK).cumsum(dim=1) > 0
        after_t = (input_ids == T_MARK).cumsum(dim=1) > 0
        modulus = self._parse_number(input_ids, digits & after_n & ~after_x)
        value = self._parse_number(input_ids, digits & after_x & ~after_t)
        steps = self._parse_number(input_ids, digits & after_t).clamp(1, MAX_OUTER_STEPS)
        return modulus, value, steps

    @staticmethod
    def _gather(registers: Tensor, source: Tensor) -> Tensor:
        return registers.gather(
            2, source[:, None, None].expand(-1, registers.shape[1], 1)
        ).squeeze(2)

    def _execute_once(
        self,
        value: Tensor,
        modulus: Tensor,
        count: int,
        *,
        return_registers: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        if value.ndim == 1:
            value = value[None].expand(count, -1)
        if modulus.ndim == 1:
            modulus = modulus[None].expand(count, -1)
        registers = torch.stack(
            (value, modulus, torch.zeros_like(value), torch.ones_like(value)), dim=2
        )
        for slot in range(SLOTS):
            left = self._gather(registers, self.source_a[:count, slot])
            right = self._gather(registers, self.source_b[:count, slot])
            denominator = right.clamp_min(1)
            choices = torch.stack(
                (
                    left,
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
            selected = choices.gather(
                2,
                self.opcode[:count, slot, None, None].expand(-1, value.shape[1], 1),
            ).squeeze(2)
            registers = torch.cat((registers, selected[:, :, None]), dim=2)
        output = self._gather(registers, self.output_register[:count])
        if return_registers:
            return output, registers
        return output

    def _execute_selected(self, value: Tensor, modulus: Tensor, selected: int) -> Tensor:
        registers = [value, modulus, torch.zeros_like(value), torch.ones_like(value)]
        for slot in range(SLOTS):
            left = registers[int(self.source_a[selected, slot])]
            right = registers[int(self.source_b[selected, slot])]
            denominator = right.clamp_min(1)
            choices = (
                left,
                (left + right).clamp(max=CAP),
                (left - right).clamp(min=0),
                safe_multiply(left, right),
                torch.div(left, denominator, rounding_mode="floor"),
                left.remainder(denominator),
                torch.bitwise_and(left, right),
                torch.minimum(left, right),
            )
            registers.append(choices[int(self.opcode[selected, slot])])
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
            F.one_hot(digits, NUM_DIGITS).float(),
        )
        return F.pad(
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
        log_weights = self.program_logits[:count].log_softmax(dim=0)
        if not self.training:
            selected = int(log_weights.argmax())
            state = value
            terminal = torch.zeros_like(value)
            for step in range(int(time_steps.max())):
                candidate = self._execute_selected(state, modulus, selected)
                terminal = torch.where(time_steps == step + 1, candidate, terminal)
                state = torch.where(time_steps > step, candidate, state)
            return self._place_logits(terminal, valid.sum(dim=1), input_ids.shape[1]), {
                "selected_program": selected,
                "selected_length": int(self.length[selected]),
            }

        depth = int(time_steps.min())
        supervision = time_steps == depth
        state, registers = self._execute_once(
            value, modulus, count, return_registers=True
        )
        for _ in range(1, depth):
            state = self._execute_once(state, modulus, count)
        # Programs with equal returned values can expose different intermediate
        # registers to future appended instructions.  Hash the complete register
        # environment, not only the output, to preserve continuation semantics.
        semantic = registers.reshape(count, -1)
        position = torch.arange(1, semantic.shape[1] + 1, device=semantic.device)
        prime_a = 1_000_003
        prime_b = 1_000_033
        weight_a = (position * 104_729).remainder(prime_a)
        weight_b = (position * position * 130_363 + 17).remainder(prime_b)
        hash_a = (
            semantic.remainder(prime_a) * weight_a[None]
        ).sum(dim=1).remainder(prime_a)
        hash_b = (
            semantic.remainder(prime_b) * weight_b[None]
        ).sum(dim=1).remainder(prime_b)
        semantic_signature = torch.stack(
            (self.length[:count], self.output_register[:count], hash_a, hash_b),
            dim=1,
        )
        logits = self.program_logits.new_zeros(
            input_ids.shape[0], input_ids.shape[1], self.config.vocab_size
        )
        return logits, {
            "program_outputs": state,
            "program_log_weights": log_weights,
            "modulus": modulus,
            "supervision_mask": supervision,
            "bit_width": self.bit_width,
            "novelty_probe": self.novelty_probe[:count],
            "fitness_probe": self.fitness_probe[:count],
            "semantic_signature": semantic_signature,
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
    modulus = batch.auxiliary["modulus"][None].clamp_min(2)
    exact_error = (output != target[None]).float()
    reduced = output.remainder(modulus)
    modular_error = (reduced != target[None]).float()
    distance = (reduced - target[None]).abs()
    distance = torch.minimum(distance, modulus - distance)
    circular_error = distance.float() / (modulus // 2).clamp_min(1)
    positions = torch.arange(batch.auxiliary["bit_width"], device=output.device)
    bit_error = (
        ((output[:, :, None] >> positions) & 1)
        != ((target[None, :, None] >> positions) & 1)
    ).float().mean(dim=2)
    penalty = 8.0 * exact_error + 4.0 * modular_error + circular_error + bit_error
    per_program = (penalty * supervision[None]).sum(dim=1) / supervision.sum().clamp_min(1)
    endpoint_loss = (
        batch.auxiliary["program_log_weights"].exp() * per_program
    ).sum()
    # Exact observational-equivalence pruning on the current endpoint batch.
    # Only the first row for each executed behavior receives a zero novelty
    # gradient.  The probe itself is held at zero, so this term changes no loss
    # value and supplies no hidden optimization step.
    _, inverse = torch.unique(
        batch.auxiliary["semantic_signature"], dim=0, return_inverse=True
    )
    rows = torch.arange(output.shape[0], device=output.device)
    first = torch.full(
        (output.shape[0],), output.shape[0], device=output.device, dtype=torch.long
    )
    first.scatter_reduce_(0, inverse, rows, reduce="amin", include_self=True)
    duplicate = rows != first[inverse]
    novelty_transport = (
        batch.auxiliary["novelty_probe"] * duplicate.to(output.dtype)
    ).sum()
    fitness_transport = (
        batch.auxiliary["fitness_probe"] * per_program.detach()
    ).sum()
    return endpoint_loss + novelty_transport + fitness_transport


class GrowthOptimizer:
    def __init__(
        self,
        model: Model,
        *,
        guided: bool = True,
        fused: bool = False,
    ) -> None:
        self.model = model
        self.guided = guided
        self.inner = torch.optim.AdamW(
            [model.program_logits, model.novelty_probe, model.fitness_probe],
            lr=0.5,
            weight_decay=0.0,
            fused=fused,
        )
        self.param_groups = self.inner.param_groups
        self.state = self.inner.state
        self.defaults = self.inner.defaults
        self.optimizers = [self.inner]
        self.generator: torch.Generator | None = None

    def _rng(self, device: torch.device) -> torch.Generator:
        if self.generator is None:
            self.generator = torch.Generator(device=device).manual_seed(20260808)
        return self.generator

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.inner.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return self.inner.state_dict()

    @torch.no_grad()
    def _grow(self) -> None:
        model = self.model
        count = int(model.active_count)
        gradient = model.fitness_probe.grad
        if gradient is None:
            return
        if bool(model.solution_found):
            model.novelty_probe.zero_()
            model.fitness_probe.zero_()
            return
        score = -gradient[:count].float()
        order = score.argsort()
        rank = torch.empty_like(score)
        rank[order] = torch.linspace(0.0, 1.0, count, device=score.device)
        model.fitness_ema[:count].mul_(0.75).add_(0.25 * rank)
        perfect = gradient[:count] == 0
        model.perfect_hits[:count] = torch.where(
            perfect,
            model.perfect_hits[:count] + 1,
            torch.zeros_like(model.perfect_hits[:count]),
        )
        if bool((model.perfect_hits[:count] >= SCORE_PASSES_PER_GROWTH).any()):
            model.solution_found.fill_(1)
            model.novelty_probe.zero_()
            model.fitness_probe.zero_()
            return
        model.score_passes.add_(1)
        if int(model.score_passes) % SCORE_PASSES_PER_GROWTH:
            model.novelty_probe.zero_()
            model.fitness_probe.zero_()
            return

        eligible = (model.length[:count] < SLOTS).nonzero().flatten()
        novelty_gradient = model.novelty_probe.grad
        if novelty_gradient is not None:
            unique = novelty_gradient[:count] == 0
            eligible = eligible[unique[eligible]]
        if eligible.numel() == 0:
            return
        parent_count = min(MAX_PARENTS, int(eligible.numel()))
        if self.guided and eligible.numel() > parent_count:
            top_count = 3 * parent_count // 4
            scores = model.fitness_ema[:count][eligible]
            order = scores.argsort(descending=True)
            top = eligible[order[:top_count]]
            remainder = eligible[order[top_count:]]
            random_count = parent_count - top_count
            random_rows = remainder[
                torch.randperm(
                    remainder.numel(),
                    device=remainder.device,
                    generator=self._rng(remainder.device),
                )[:random_count]
            ]
            parents = torch.cat((top, random_rows))
        else:
            parents = eligible[:parent_count]

        # Snapshot before overwriting the pruned population.
        parent_opcode = model.opcode[parents].clone()
        parent_a = model.source_a[parents].clone()
        parent_b = model.source_b[parents].clone()
        parent_length = model.length[parents].clone()
        parent_output = model.output_register[parents].clone()
        parent_fitness = model.fitness_ema[parents].clone()
        parent_hits = model.perfect_hits[parents].clone()
        child_count = min(
            POPULATION - parent_count,
            parent_count * CHILDREN_PER_PARENT,
        )
        child_rows = torch.arange(parent_count, parent_count + child_count, device=parents.device)
        parent_local = torch.div(
            torch.arange(child_count, device=parents.device),
            CHILDREN_PER_PARENT,
            rounding_mode="floor",
        ).clamp_max(parent_count - 1)
        variant = torch.arange(child_count, device=parents.device).remainder(
            CHILDREN_PER_PARENT
        )

        model.opcode[:parent_count] = parent_opcode
        model.source_a[:parent_count] = parent_a
        model.source_b[:parent_count] = parent_b
        model.length[:parent_count] = parent_length
        model.output_register[:parent_count] = parent_output
        model.fitness_ema[:parent_count] = parent_fitness
        model.perfect_hits[:parent_count] = parent_hits
        model.opcode[child_rows] = parent_opcode[parent_local]
        model.source_a[child_rows] = parent_a[parent_local]
        model.source_b[child_rows] = parent_b[parent_local]
        child_length = parent_length[parent_local]
        child_output = parent_output[parent_local]
        available = 4 + child_length
        # Three complementary mutations are needed for a genuinely useful
        # straight-line language:
        #
        #  * chain: combine the current return value with an existing register;
        #  * arbitrary-output: combine any two registers and return the result;
        #  * branch: combine any two registers but keep the old return value.
        #
        # The last form is essential.  It lets a promising recurrence retain
        # its endpoint fitness while constructing a constant or side branch in
        # a new register for a later instruction.  Without it, every useful
        # detour is judged as though the detour itself were the recurrence.
        generation = model.growth_generation.to(device=parents.device)
        parent_key = parent_local + 1
        chain = variant < CHAIN_CHILDREN
        arbitrary_output = (
            variant >= CHAIN_CHILDREN
        ) & (variant < CHAIN_CHILDREN + ARBITRARY_OUTPUT_CHILDREN)
        branch = ~chain & ~arbitrary_output

        chain_code = variant
        chain_operation = chain_code.remainder(len(OPERATIONS))
        chain_source = (
            torch.div(chain_code, len(OPERATIONS), rounding_mode="floor")
            + 3 * parent_key
            + 5 * generation
        ).remainder(available)
        chain_orientation = (parent_key + generation + chain_code).remainder(2)

        free_code = torch.where(
            arbitrary_output,
            variant - CHAIN_CHILDREN,
            variant - CHAIN_CHILDREN - ARBITRARY_OUTPUT_CHILDREN,
        ).clamp_min(0)
        # Cheap integer mixing makes successive retained parents explore new
        # operand pairs instead of regenerating the same fixed 126 children.
        free_operation = (
            free_code + 3 * parent_key + 5 * generation
        ).remainder(len(OPERATIONS))
        free_a = (
            5 * free_code + 7 * parent_key + 11 * generation
        ).remainder(available)
        free_b = (
            11 * free_code + 13 * parent_key + 17 * generation + 1
        ).remainder(available)

        operation = torch.where(chain, chain_operation, free_operation)
        left_source = torch.where(
            chain,
            torch.where(chain_orientation == 0, child_output, chain_source),
            free_a,
        )
        right_source = torch.where(
            chain,
            torch.where(chain_orientation == 0, chain_source, child_output),
            free_b,
        )
        model.opcode[child_rows, child_length] = operation
        model.source_a[child_rows, child_length] = left_source
        model.source_b[child_rows, child_length] = right_source
        model.length[child_rows] = child_length + 1
        model.output_register[child_rows] = torch.where(
            branch, child_output, 4 + child_length
        )
        model.fitness_ema[child_rows] = parent_fitness[parent_local] * 0.95
        model.perfect_hits[child_rows].zero_()

        new_count = parent_count + child_count
        model.pruned_programs.add_(max(0, count - parent_count))
        model.generated_programs.add_(child_count)
        model.active_count.fill_(new_count)
        model.growth_generation.add_(1)
        model.program_logits.zero_()
        model.novelty_probe.zero_()
        model.fitness_probe.zero_()
        for parameter in (
            model.program_logits,
            model.novelty_probe,
            model.fitness_probe,
        ):
            for value in self.inner.state.get(parameter, {}).values():
                if torch.is_tensor(value) and value.shape == parameter.shape:
                    value.zero_()

    @torch.no_grad()
    def step(self, closure=None):
        result = self.inner.step(closure)
        self._grow()
        return result


def build_model(spec: ModelSpec) -> Model:
    model = Model(spec)
    assert_model_state(model, spec)
    return model


def build_optimizer(model: Model, spec: OptimizerSpec) -> OptimizerBundle:
    return OptimizerBundle(
        optimizer=GrowthOptimizer(model, fused=spec.device_type == "cuda")
    )


SUBMISSION = Submission(
    build_model=build_model,
    build_optimizer=build_optimizer,
    token_training_loss=token_training_loss,
    batch_size=32,
    eval_batch_size=512,
)
