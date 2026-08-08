"""Endpoint-guided generation of a recurrent binary program.

A small population begins as random complete programs in a generic binary
transducer language.  The endpoint loss supplies dense per-program fitness;
the optimizer repeatedly selects, clones, mutates, and restarts programs before
ordinary posterior learning chooses the best generated program.
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
POPULATION = 96
MAX_OUTER_STEPS = 64
POSTERIOR_LR = 2e-1
RELATION_LR = 1e-4
SURROGATE_SCALE = 1e-3
ELITE_COUNT = 24
RESTART_COUNT = 19
MUTATION_LOGIT = 4.0
BETA_VALUES = (-2.0, -1.0, 0.0, 1.0, 2.0)
GAMMA_VALUES = (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0)
FIELD_NAMES = (
    "beta_logits",
    "gamma_logits",
    "branch_logits",
    "initial_logits",
    "direction_logits",
    "addend_logits",
    "bit_branch_logits",
    "second_pass_logits",
)


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
        self.bit_width = (3322 * self.decimal_width + 999) // 1000 + 2
        self.particles = POPULATION
        self.beta_logits = nn.Parameter(torch.randn(POPULATION, 5) * 0.05)
        self.gamma_logits = nn.Parameter(torch.randn(POPULATION, 7) * 0.05)
        self.branch_logits = nn.Parameter(torch.randn(POPULATION, 2) * 0.05)
        self.initial_logits = nn.Parameter(torch.randn(POPULATION, 2) * 0.05)
        self.direction_logits = nn.Parameter(torch.randn(POPULATION, 2) * 0.05)
        self.addend_logits = nn.Parameter(torch.randn(POPULATION, 2) * 0.05)
        self.bit_branch_logits = nn.Parameter(torch.randn(POPULATION, 2) * 0.05)
        self.second_pass_logits = nn.Parameter(torch.randn(POPULATION, 2) * 0.05)
        self.program_logits = nn.Parameter(torch.randn(POPULATION) * 0.02)
        self.register_buffer("evolution_fitness", torch.zeros(POPULATION))
        self.register_buffer("evolution_generation", torch.zeros((), dtype=torch.long))
        self.evolution_enabled = True
        self.transition_sharpness = 2.0

    def _choice(self, logits: Tensor, values: tuple[float, ...]) -> Tensor:
        probability = F.softmax(logits, dim=-1)
        hard = F.one_hot(probability.argmax(dim=-1), probability.shape[-1]).to(
            probability.dtype
        )
        if self.training:
            hard = hard + SURROGATE_SCALE * (probability - probability.detach())
        return hard @ logits.new_tensor(values)

    def _programs(self, indices: Tensor | None = None) -> Tensor:
        if indices is None:
            indices = torch.arange(self.particles, device=self.program_logits.device)
        return torch.stack(
            (
                self._choice(self.beta_logits[indices], BETA_VALUES),
                self._choice(self.gamma_logits[indices], GAMMA_VALUES),
                self._choice(self.branch_logits[indices], (0.0, 1.0)),
            ),
            dim=-1,
        )

    def _recurrence_programs(self, indices: Tensor | None = None) -> Tensor:
        if indices is None:
            indices = torch.arange(self.particles, device=self.program_logits.device)
        return torch.stack(
            (
                self._choice(self.initial_logits[indices], (0.0, 1.0)),
                self._choice(self.direction_logits[indices], (0.0, 1.0)),
                self._choice(self.addend_logits[indices], (0.0, 1.0)),
                self._choice(self.bit_branch_logits[indices], (0.0, 1.0)),
                self._choice(self.second_pass_logits[indices], (0.0, 1.0)),
            ),
            dim=-1,
        )

    def _transition_tables(self, indices: Tensor) -> Tensor:
        programs = self._programs(indices)
        beta, gamma = programs[:, 0], programs[:, 1]
        a, b, carry, output, next_carry = torch.meshgrid(
            *(
                torch.arange(2, device=beta.device, dtype=beta.dtype)
                for _ in range(5)
            ),
            indexing="ij",
        )
        residual = (
            a
            + b
            + carry
            + beta[:, None, None, None, None, None] * output
            + gamma[:, None, None, None, None, None] * next_carry
        )
        probability = (-self.transition_sharpness * residual.square()).flatten(
            -2
        ).softmax(dim=-1)
        hard = F.one_hot(probability.argmax(dim=-1), 4).to(probability.dtype)
        if self.training:
            probability = hard + SURROGATE_SCALE * (
                probability - probability.detach()
            )
        else:
            probability = hard
        return probability.reshape(-1, 2, 2, 2, 2, 2)

    def _scan_add(
        self, left: Tensor, right: Tensor, initial_carry: int, table: Tensor
    ) -> tuple[Tensor, Tensor]:
        carry = left.new_zeros(left.shape[0], left.shape[1], 2)
        carry[:, :, initial_carry] = 1.0
        outputs = []
        for position in range(left.shape[2]):
            if self.training and position > 0:
                carry = carry.detach()
            action = torch.einsum(
                "pxa,pxb,pxc,pabcon->pxon",
                left[:, :, position],
                right[:, :, position],
                carry,
                table,
            )
            action = action.clamp_min(0.0)
            action = action / action.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
            if not self.training:
                flat = action.reshape(*action.shape[:2], 4)
                action = F.one_hot(flat.argmax(dim=-1), 4).to(flat.dtype).reshape_as(
                    action
                )
            outputs.append(action.sum(dim=3))
            carry = action.sum(dim=2)
        return torch.stack(outputs, dim=2), carry

    def _modadd(
        self,
        left: Tensor,
        right: Tensor,
        modulus: Tensor,
        table: Tensor,
        branch: Tensor,
    ) -> Tensor:
        raw, _ = self._scan_add(left, right, 0, table)
        difference, carry = self._scan_add(raw, modulus.flip(-1), 1, table)
        orientation = branch[:, 1, None]
        carry_one = carry[:, :, 1]
        subtract = orientation * carry_one + (1.0 - orientation) * (1.0 - carry_one)
        result = (
            subtract[:, :, None, None] * difference
            + (1.0 - subtract[:, :, None, None]) * raw
        ).clamp_min(0.0)
        return result / result.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    def _recur(
        self,
        value: Tensor,
        modulus: Tensor,
        table: Tensor,
        branch: Tensor,
        recurrence: Tensor,
    ) -> Tensor:
        zero = value.new_zeros(value.shape)
        zero[:, :, :, 0] = 1.0
        initial_value = recurrence[:, 0, None, None, None]
        lsb_direction = recurrence[:, 1, None]
        use_value_addend = recurrence[:, 2, None, None, None]
        invert_bit_branch = recurrence[:, 3, None]
        two_pass = recurrence[:, 4, None, None, None]

        def horner(addend: Tensor) -> Tensor:
            accumulator = initial_value * value + (1.0 - initial_value) * zero
            for scan_step in range(value.shape[2]):
                if self.training and scan_step > 0:
                    accumulator = accumulator.detach()
                doubled = self._modadd(
                    accumulator, accumulator, modulus, table, branch
                )
                added = self._modadd(doubled, addend, modulus, table, branch)
                bit_lsb = value[:, :, scan_step, 1]
                bit_msb = value[:, :, value.shape[2] - 1 - scan_step, 1]
                bit = lsb_direction * bit_lsb + (1.0 - lsb_direction) * bit_msb
                choose_added = (
                    invert_bit_branch * (1.0 - bit)
                    + (1.0 - invert_bit_branch) * bit
                )
                accumulator = (
                    choose_added[:, :, None, None] * added
                    + (1.0 - choose_added[:, :, None, None]) * doubled
                ).clamp_min(0.0)
                accumulator = accumulator / accumulator.sum(
                    dim=-1, keepdim=True
                ).clamp_min(1e-12)
            return accumulator

        first_addend = use_value_addend * value + (1.0 - use_value_addend) * zero
        first = horner(first_addend)
        second = horner(first)
        return two_pass * second + (1.0 - two_pass) * first

    @staticmethod
    def _parse_number(input_ids: Tensor, mask: Tensor) -> Tensor:
        value = torch.zeros(
            input_ids.shape[0], device=input_ids.device, dtype=torch.long
        )
        for position in range(input_ids.shape[1]):
            digit = (input_ids[:, position] - DIGIT_OFFSET).clamp(0, 9)
            value = torch.where(mask[:, position], value * 10 + digit, value)
        return value

    @staticmethod
    def _time_steps(input_ids: Tensor, valid: Tensor) -> Tensor:
        digits = (
            valid
            & (input_ids >= DIGIT_OFFSET)
            & (input_ids < DIGIT_OFFSET + NUM_DIGITS)
        )
        after_t = (input_ids == T_MARK).cumsum(dim=1) > 0
        return Model._parse_number(input_ids, digits & after_t).clamp(
            min=1, max=MAX_OUTER_STEPS
        )

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
        value = self._parse_number(input_ids, digits & after_x & ~after_t)
        return modulus, value, self._time_steps(input_ids, valid)

    def _bits(self, value: Tensor) -> Tensor:
        position = torch.arange(self.bit_width, device=value.device)
        bits = ((value[:, None] >> position[None]) & 1).long()
        return F.one_hot(bits, 2).float()

    def _hard_scan_add_integer(
        self, left: Tensor, right: Tensor, initial_carry: int, table: Tensor
    ) -> tuple[Tensor, Tensor]:
        action = table.reshape(8, 4).argmax(dim=-1)
        carry = torch.full_like(left, initial_carry)
        result = torch.zeros_like(left)
        for position in range(self.bit_width):
            a = (left >> position) & 1
            b = (right >> position) & 1
            selected = action[a * 4 + b * 2 + carry]
            result = result | ((selected // 2) << position)
            carry = selected & 1
        return result, carry

    def _hard_modadd_integer(
        self,
        left: Tensor,
        right: Tensor,
        modulus: Tensor,
        table: Tensor,
        orientation: Tensor,
    ) -> Tensor:
        mask = (1 << self.bit_width) - 1
        raw, _ = self._hard_scan_add_integer(left, right, 0, table)
        difference, carry = self._hard_scan_add_integer(
            raw, (~modulus) & mask, 1, table
        )
        subtract = torch.where(orientation, carry, 1 - carry).bool()
        return torch.where(subtract, difference, raw)

    def _hard_recur_integer(
        self,
        value: Tensor,
        modulus: Tensor,
        table: Tensor,
        orientation: Tensor,
        recurrence: Tensor,
    ) -> Tensor:
        initial_value, lsb_direction, use_value_addend, invert_bit_branch, two_pass = (
            recurrence.bool().unbind()
        )

        def horner(addend: Tensor) -> Tensor:
            accumulator = value if initial_value else torch.zeros_like(value)
            for scan_step in range(self.bit_width):
                doubled = self._hard_modadd_integer(
                    accumulator, accumulator, modulus, table, orientation
                )
                added = self._hard_modadd_integer(
                    doubled, addend, modulus, table, orientation
                )
                position = scan_step if lsb_direction else self.bit_width - 1 - scan_step
                bit = ((value >> position) & 1).bool()
                if invert_bit_branch:
                    bit = ~bit
                accumulator = torch.where(bit, added, doubled)
            return accumulator

        first = horner(value if use_value_addend else torch.zeros_like(value))
        return horner(first) if two_pass else first

    def _hard_execute(
        self, value: Tensor, modulus: Tensor, t_values: Tensor, selected: int
    ) -> Tensor:
        index = torch.tensor([selected], device=value.device)
        table = self._transition_tables(index)[0]
        orientation = self._programs(index)[0, 2].bool()
        recurrence = self._recurrence_programs(index)[0]
        terminal = torch.zeros_like(value)
        state = value
        for outer_step in range(int(t_values.max().item())):
            candidate = self._hard_recur_integer(
                state, modulus, table, orientation, recurrence
            )
            terminal = torch.where(t_values == outer_step + 1, candidate, terminal)
            state = torch.where(t_values > outer_step, candidate, state)
        return terminal

    def _place_integer_logits(
        self, value: Tensor, lengths: Tensor, prompt: int
    ) -> Tensor:
        decimal_powers = 10 ** torch.arange(self.decimal_width, device=value.device)
        digits_lsd = (value[:, None] // decimal_powers[None]).remainder(10)
        digits = digits_lsd.flip(1)
        destination = lengths[:, None] - self.decimal_width + torch.arange(
            self.decimal_width, device=value.device
        )[None]
        destination = destination.clamp(0, prompt - 1)
        canvas = torch.zeros(
            value.shape[0], prompt, NUM_DIGITS, device=value.device
        )
        canvas.scatter_(
            1,
            destination.unsqueeze(-1).expand(-1, -1, NUM_DIGITS),
            F.one_hot(digits, NUM_DIGITS).float(),
        )
        digit_logits = canvas.clamp_min(1e-7).log()
        return F.pad(
            digit_logits,
            (DIGIT_OFFSET, self.config.vocab_size - DIGIT_OFFSET - NUM_DIGITS),
            value=-16.0,
        )

    def forward(
        self, input_ids: Tensor, attention_mask: Tensor | None = None
    ) -> tuple[Tensor, dict[str, object]]:
        if attention_mask is None:
            attention_mask = input_ids != PAD
        valid = attention_mask.bool()
        modulus_integer, value_integer, t_values = self._parse(input_ids, valid)
        log_weights = F.log_softmax(self.program_logits, dim=0)
        if not self.training:
            selected = int(log_weights.argmax().item())
            terminal_integer = self._hard_execute(
                value_integer, modulus_integer, t_values, selected
            )
            logits = self._place_integer_logits(
                terminal_integer, valid.sum(dim=1), input_ids.shape[1]
            )
            return logits, {
                "program_bits": self._bits(terminal_integer)[None],
                "program_log_weights": log_weights,
                "bit_width": self.bit_width,
                "t_values": t_values,
            }

        initial = self._bits(value_integer)
        modulus_bits = self._bits(modulus_integer)
        indices = torch.arange(self.particles, device=input_ids.device)
        table = self._transition_tables(indices)
        orientation = self._programs(indices)[:, 2]
        recurrence = self._recurrence_programs(indices)
        branch = torch.stack((1.0 - orientation, orientation), dim=-1).to(table.dtype)
        supervision_depth = int(t_values.min().item())
        supervision_mask = t_values == supervision_depth
        supervision_indices = supervision_mask.nonzero(as_tuple=False).flatten()
        terminal = initial.new_zeros(
            self.particles, initial.shape[0], self.bit_width, 2
        )
        if supervision_indices.numel() > 0:
            largest_modulus = int(modulus_integer[supervision_indices].max().item())
            arithmetic_width = max(2, largest_modulus.bit_length() + 1)
            value = initial[supervision_indices, :arithmetic_width][None].expand(
                self.particles, -1, -1, -1
            )
            modulus = modulus_bits[
                supervision_indices, :arithmetic_width
            ][None].expand_as(value)
            candidate = value
            for outer_step in range(supervision_depth):
                if outer_step > 0:
                    candidate = candidate.detach()
                candidate = self._recur(candidate, modulus, table, branch, recurrence)
            if arithmetic_width < self.bit_width:
                high_zero = candidate.new_zeros(
                    self.particles,
                    candidate.shape[1],
                    self.bit_width - arithmetic_width,
                    2,
                )
                high_zero[..., 0] = 1.0
                candidate = torch.cat((candidate, high_zero), dim=2)
            terminal = terminal.index_copy(1, supervision_indices, candidate)

        # The official loss is supplied entirely through the endpoint-program
        # auxiliary tensors.  Training logits need only satisfy the token API;
        # evaluation uses the selected program's exact integer execution above.
        logits = terminal.new_zeros(
            input_ids.shape[0], input_ids.shape[1], self.config.vocab_size
        )
        return logits, {
            "program_bits": terminal,
            "program_log_weights": log_weights,
            "bit_width": self.bit_width,
            "t_values": t_values,
            "supervision_mask": supervision_mask,
        }


def _target_integer(labels: Tensor, valid: Tensor) -> Tensor:
    value = torch.zeros(labels.shape[0], device=labels.device, dtype=torch.long)
    for position in range(labels.shape[1]):
        digit = (labels[:, position] - DIGIT_OFFSET).clamp(0, 9)
        value = torch.where(valid[:, position], value * 10 + digit, value)
    return value


def token_training_loss(batch: TokenLossBatch) -> Tensor:
    """Exact endpoint likelihood plus a dense endpoint-bit fitness slope."""
    supervision = batch.auxiliary["supervision_mask"].to(batch.logits.dtype)
    target = _target_integer(batch.labels, batch.valid_mask)
    bit_width = int(batch.auxiliary["bit_width"])
    position = torch.arange(bit_width, device=target.device)
    target_bits = ((target[:, None] >> position[None]) & 1).long()
    probability = batch.auxiliary["program_bits"].clamp_min(1e-12)
    observed_log = probability.gather(
        3,
        target_bits[None, :, :, None].expand(
            probability.shape[0], -1, -1, 1
        ),
    ).squeeze(3).log().sum(dim=2)
    log_weights = batch.auxiliary["program_log_weights"].float()
    exact_nll = -torch.logsumexp(log_weights[:, None] + observed_log, dim=0)
    dense_score = observed_log / bit_width
    dense_nll = -(log_weights.exp()[:, None] * dense_score).sum(dim=0)
    denominator = supervision.sum().clamp_min(1)
    exact_loss = (exact_nll * supervision).sum() / denominator
    dense_loss = (dense_nll * supervision).sum() / denominator
    return exact_loss + dense_loss


def structured_parameters(model: Model) -> list[nn.Parameter]:
    return [getattr(model, name) for name in FIELD_NAMES]


class EvolutionaryOptimizer:
    def __init__(self, model: Model, groups: list[dict], fused: bool) -> None:
        self.model = model
        self.inner = torch.optim.AdamW(
            groups, betas=(0.9, 0.95), eps=1e-8, fused=fused
        )
        self.param_groups = self.inner.param_groups
        self.state = self.inner.state
        self.defaults = self.inner.defaults
        self.optimizers = [self.inner]
        self.steps = 0
        self.generator: torch.Generator | None = None

    def _rng(self, device: torch.device) -> torch.Generator:
        if self.generator is None:
            self.generator = torch.Generator(device=device).manual_seed(20260808)
        return self.generator

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.inner.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict:
        payload = self.inner.state_dict()
        payload["evolution_steps"] = self.steps
        return payload

    def load_state_dict(self, state_dict: dict) -> None:
        payload = dict(state_dict)
        self.steps = int(payload.pop("evolution_steps", 0))
        self.inner.load_state_dict(payload)
        self.param_groups = self.inner.param_groups
        self.state = self.inner.state

    @torch.no_grad()
    def _zero_rows(self, parameter: nn.Parameter, rows: Tensor) -> None:
        for value in self.inner.state.get(parameter, {}).values():
            if torch.is_tensor(value) and value.shape == parameter.shape:
                value[rows] = 0

    @torch.no_grad()
    def _record_fitness(self) -> None:
        gradient = self.model.program_logits.grad
        if gradient is None:
            return
        score = -gradient.float()
        order = score.argsort()
        rank = torch.empty_like(score)
        rank[order] = torch.linspace(0.0, 1.0, score.numel(), device=score.device)
        self.model.evolution_fitness.copy_(rank)

    @torch.no_grad()
    def _evolve(self) -> None:
        model = self.model
        device = model.program_logits.device
        generator = self._rng(device)
        order = model.evolution_fitness.argsort(descending=True)
        elites = order[:ELITE_COUNT]
        children = order[ELITE_COUNT:]
        parents = elites[
            torch.randint(
                ELITE_COUNT,
                (children.numel(),),
                device=device,
                generator=generator,
            )
        ]
        fields = structured_parameters(model)
        for parameter in fields:
            parameter[children] = parameter[parents]

        restart_rows = children[:RESTART_COUNT]
        for parameter in fields:
            parameter[restart_rows].normal_(
                mean=0.0, std=0.05, generator=generator
            )

        mutate_rows = children[RESTART_COUNT:]
        field_choice = torch.randint(
            len(fields),
            (mutate_rows.numel(),),
            device=device,
            generator=generator,
        )
        self._mutate(fields, mutate_rows, field_choice, generator)
        second_mask = (
            torch.rand(mutate_rows.numel(), device=device, generator=generator) < 0.15
        )
        second_rows = mutate_rows[second_mask]
        if second_rows.numel():
            second_fields = torch.randint(
                len(fields),
                (second_rows.numel(),),
                device=device,
                generator=generator,
            )
            self._mutate(fields, second_rows, second_fields, generator)

        for parameter in fields:
            self._zero_rows(parameter, children)
        all_rows = torch.arange(model.particles, device=device)
        self._zero_rows(model.program_logits, all_rows)
        model.program_logits.zero_()
        model.evolution_fitness.zero_()
        model.evolution_generation.add_(1)

    @staticmethod
    @torch.no_grad()
    def _mutate(
        fields: list[nn.Parameter],
        rows: Tensor,
        choices: Tensor,
        generator: torch.Generator,
    ) -> None:
        for field_index, parameter in enumerate(fields):
            selected = rows[choices == field_index]
            if selected.numel() == 0:
                continue
            old = parameter[selected].argmax(dim=-1)
            offset = torch.randint(
                1,
                parameter.shape[1],
                (selected.numel(),),
                device=parameter.device,
                generator=generator,
            )
            new = (old + offset).remainder(parameter.shape[1])
            parameter[selected].fill_(-MUTATION_LOGIT)
            parameter[selected, new] = MUTATION_LOGIT

    @torch.no_grad()
    def step(self, closure=None):
        self._record_fitness()
        result = self.inner.step(closure)
        self.steps += 1
        if self.model.evolution_enabled:
            self._evolve()
        return result


class Schedule:
    def __init__(
        self, model: Model, optimizer: EvolutionaryOptimizer, budget: float
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.started = time.monotonic()
        self.budget = max(float(budget) * 0.98, 1.0)

    def step(self) -> None:
        fraction = min((time.monotonic() - self.started) / self.budget, 1.0)
        self.model.evolution_enabled = fraction < 0.65
        multiplier = 1.0
        if fraction > 0.90:
            progress = (fraction - 0.90) / 0.10
            multiplier = 0.2 + 0.8 * 0.5 * (1.0 + math.cos(math.pi * progress))
        for group in self.optimizer.param_groups:
            group["lr"] = group["base_lr"] * multiplier


def build_model(spec: ModelSpec) -> Model:
    model = Model(spec)
    assert_model_state(model, spec)
    return model


def build_optimizer(model: Model, spec: OptimizerSpec) -> OptimizerBundle:
    posterior = [model.program_logits]
    relation = structured_parameters(model)
    optimizer = EvolutionaryOptimizer(
        model,
        [
            {
                "params": posterior,
                "lr": POSTERIOR_LR,
                "base_lr": POSTERIOR_LR,
                "weight_decay": 0.0,
            },
            {
                "params": relation,
                "lr": RELATION_LR,
                "base_lr": RELATION_LR,
                "weight_decay": 0.0,
            },
        ],
        fused=spec.device_type == "cuda",
    )
    return OptimizerBundle(
        optimizer=optimizer,
        scheduler=Schedule(model, optimizer, spec.training_time_seconds),
    )


SUBMISSION = Submission(
    build_model=build_model,
    build_optimizer=build_optimizer,
    token_training_loss=token_training_loss,
    batch_size=128,
    eval_batch_size=512,
)
