"""Hard-tier factorized posterior over one recurrent discrete program.

Seven randomly initialized categorical vectors define one global program:
two small affine-relation coefficients, branch polarity, modulus transform,
reduction carry, scan direction, and bit-gate polarity.  Training exactly
marginalizes their small Cartesian hypothesis class using evaluator endpoints.
Evaluation executes only the learned joint MAP program, shared across every
bit, example, modulus, squaring step, and recurrence depth.
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
PROGRAMS = 5 * 7 * 2 * 2 * 2 * 2 * 2
MAX_OUTER_STEPS = 64
POSTERIOR_LR = 2e-1
DECODER_LR = 2e-3
BIT_LOSS_WEIGHT = 1.0
DIRECT_LOSS_WEIGHT = 0.01
BETA_VALUES = (-2.0, -1.0, 0.0, 1.0, 2.0)
GAMMA_VALUES = (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0)
FIELD_SIZES = (5, 7, 2, 2, 2, 2, 2)


class Config:
    def __init__(self, vocab_size: int, max_seq_len: int) -> None:
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len


class Model(nn.Module):
    num_loops = 1

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        # Three markers plus at least one T digit leave room for equally wide
        # N and x fields: 2 * decimal_width + 4 <= max_seq_len.
        self.decimal_width = max(2, (spec.max_seq_len - 4) // 2)
        # Decimal payload capacity plus one explicit addition-overflow bit.
        # (The ceiling term itself can be tight, so this requires +2.)
        self.bit_width = (3322 * self.decimal_width + 999) // 1000 + 2
        self.particles = PROGRAMS
        self.beta_logits = nn.Parameter(
            torch.randn(len(BETA_VALUES)) * 0.02
        )
        self.gamma_logits = nn.Parameter(
            torch.randn(len(GAMMA_VALUES)) * 0.02
        )
        self.branch_logits = nn.Parameter(
            torch.randn(2) * 0.02
        )
        self.invert_modulus_logits = nn.Parameter(
            torch.randn(2) * 0.02
        )
        self.reduction_carry_logits = nn.Parameter(
            torch.randn(2) * 0.02
        )
        self.scan_direction_logits = nn.Parameter(
            torch.randn(2) * 0.02
        )
        self.bit_gate_logits = nn.Parameter(
            torch.randn(2) * 0.02
        )
        self.register_buffer(
            "program_choices",
            torch.cartesian_prod(
                *(torch.arange(size) for size in FIELD_SIZES)
            ),
            persistent=True,
        )
        hidden = 64
        self.decimal_decoder = nn.Sequential(
            nn.Linear(2 * self.bit_width, hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.decimal_width * NUM_DIGITS),
        )
        self.transition_sharpness = 2.0

    def _programs(self, indices: Tensor | None = None) -> Tensor:
        if indices is None:
            indices = torch.arange(
                self.particles, device=self.program_choices.device
            )
        choices = self.program_choices[indices]
        beta_values = self.beta_logits.new_tensor(BETA_VALUES)
        gamma_values = self.gamma_logits.new_tensor(GAMMA_VALUES)
        return torch.stack(
            (
                beta_values[choices[:, 0]],
                gamma_values[choices[:, 1]],
                choices[:, 2].to(self.beta_logits.dtype),
            ),
            dim=-1,
        )

    def _controls(self, indices: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        choices = self.program_choices[indices]
        return tuple(
            choices[:, field].to(self.beta_logits.dtype)
            for field in range(3, 7)
        )

    def _transition_tables(self, indices: Tensor) -> Tensor:
        programs = self._programs(indices)
        beta = programs[:, 0]
        gamma = programs[:, 1]
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
        score = -self.transition_sharpness * residual.square()
        probability = score.flatten(-2).softmax(dim=-1)
        hard = F.one_hot(
            probability.argmax(dim=-1), 4
        ).to(probability.dtype)
        probability = hard
        return probability.reshape(-1, 2, 2, 2, 2, 2)

    def _scan_add(
        self,
        left: Tensor,
        right: Tensor,
        initial_carry: int | Tensor,
        table: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if isinstance(initial_carry, int):
            carry = left.new_zeros(left.shape[0], left.shape[1], 2)
            carry[:, :, initial_carry] = 1.0
        else:
            carry = torch.stack(
                (1.0 - initial_carry, initial_carry), dim=-1
            )[:, None].expand(-1, left.shape[1], -1)
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
            # Each scan step represents one joint distribution over the output
            # bit and next carry.  Long differentiable arithmetic chains amplify
            # even tiny mass drift, so restore the probability invariant here.
            action = action.clamp_min(0.0)
            action = action / action.sum(
                dim=(-2, -1), keepdim=True
            ).clamp_min(1e-12)
            if not self.training:
                flat = action.reshape(*action.shape[:2], 4)
                flat = F.one_hot(flat.argmax(dim=-1), 4).to(flat.dtype)
                action = flat.reshape_as(action)
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
        invert_modulus: Tensor,
        reduction_carry: Tensor,
    ) -> Tensor:
        raw, _ = self._scan_add(left, right, 0, table)
        reduction_operand = (
            invert_modulus[:, None, None, None] * modulus.flip(-1)
            + (1.0 - invert_modulus[:, None, None, None]) * modulus
        )
        difference, carry = self._scan_add(
            raw, reduction_operand, reduction_carry, table
        )
        orientation = branch[:, 1, None]
        carry_one = carry[:, :, 1]
        subtract = orientation * carry_one + (1.0 - orientation) * (
            1.0 - carry_one
        )
        result = (
            subtract[:, :, None, None] * difference
            + (1.0 - subtract[:, :, None, None]) * raw
        )
        result = result.clamp_min(0.0)
        return result / result.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    def _square(
        self,
        value: Tensor,
        modulus: Tensor,
        table: Tensor,
        branch: Tensor,
        invert_modulus: Tensor,
        reduction_carry: Tensor,
        scan_direction: Tensor,
        bit_gate: Tensor,
    ) -> Tensor:
        accumulator = value.new_zeros(value.shape)
        accumulator[:, :, :, 0] = 1.0
        for scan_step, position in enumerate(
            reversed(range(value.shape[2]))
        ):
            if self.training and scan_step > 0:
                accumulator = accumulator.detach()
            doubled = self._modadd(
                accumulator,
                accumulator,
                modulus,
                table,
                branch,
                invert_modulus,
                reduction_carry,
            )
            added = self._modadd(
                doubled,
                value,
                modulus,
                table,
                branch,
                invert_modulus,
                reduction_carry,
            )
            reverse_position = value.shape[2] - 1 - position
            bit = (
                scan_direction[:, None]
                * value[:, :, position, 1]
                + (1.0 - scan_direction[:, None])
                * value[:, :, reverse_position, 1]
            )
            bit = (
                bit_gate[:, None] * bit
                + (1.0 - bit_gate[:, None]) * (1.0 - bit)
            )
            accumulator = (
                bit[:, :, None, None] * added
                + (1.0 - bit[:, :, None, None]) * doubled
            )
            accumulator = accumulator.clamp_min(0.0)
            accumulator = accumulator / accumulator.sum(
                dim=-1, keepdim=True
            ).clamp_min(1e-12)
        return accumulator

    @staticmethod
    def _parse_number(input_ids: Tensor, mask: Tensor) -> Tensor:
        value = torch.zeros(
            input_ids.shape[0], device=input_ids.device, dtype=torch.long
        )
        for position in range(input_ids.shape[1]):
            digit = (input_ids[:, position] - DIGIT_OFFSET).clamp(0, 9)
            value = torch.where(
                mask[:, position], value * 10 + digit, value
            )
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
        modulus = self._parse_number(
            input_ids, digits & after_n & ~after_x
        )
        value = self._parse_number(
            input_ids, digits & after_x & ~after_t
        )
        return modulus, value, self._time_steps(input_ids, valid)

    def _bits(self, value: Tensor) -> Tensor:
        position = torch.arange(self.bit_width, device=value.device)
        bits = ((value[:, None] >> position[None]) & 1).long()
        return F.one_hot(bits, 2).float()

    def _program_log_weights(self) -> Tensor:
        choices = self.program_choices
        field_logits = (
            self.beta_logits,
            self.gamma_logits,
            self.branch_logits,
            self.invert_modulus_logits,
            self.reduction_carry_logits,
            self.scan_direction_logits,
            self.bit_gate_logits,
        )
        return sum(
            F.log_softmax(logits, dim=0)[choices[:, field]]
            for field, logits in enumerate(field_logits)
        )

    def _place_logits(
        self, digit_logits_lsd: Tensor, lengths: Tensor, prompt: int
    ) -> Tensor:
        digits = digit_logits_lsd.flip(1)
        destination = lengths[:, None] - self.decimal_width + torch.arange(
            self.decimal_width, device=digits.device
        )[None]
        destination = destination.clamp(0, prompt - 1)
        canvas = torch.zeros(
            digits.shape[0],
            prompt,
            NUM_DIGITS,
            device=digits.device,
            dtype=digits.dtype,
        )
        canvas = canvas.scatter(
            1,
            destination.unsqueeze(-1).expand(-1, -1, NUM_DIGITS),
            digits,
        )
        return F.pad(
            canvas,
            (
                DIGIT_OFFSET,
                self.config.vocab_size - DIGIT_OFFSET - NUM_DIGITS,
            ),
            value=-16.0,
        )

    def _decode_soft(self, bits: Tensor) -> Tensor:
        return self.decimal_decoder(bits.flatten(2)).reshape(
            bits.shape[0],
            bits.shape[1],
            self.decimal_width,
            NUM_DIGITS,
        ).flip(2)

    def _decode_hard(self, bits: Tensor) -> Tensor:
        binary = bits.argmax(dim=-1)
        powers = 1 << torch.arange(self.bit_width, device=bits.device)
        value = (binary * powers[None, :]).sum(dim=1)
        decimal_powers = 10 ** torch.arange(
            self.decimal_width, device=bits.device
        )
        digits = (value[:, None] // decimal_powers[None]).remainder(10)
        logits = bits.new_full(
            (bits.shape[0], self.decimal_width, NUM_DIGITS), -16.0
        )
        logits.scatter_(2, digits.unsqueeze(-1), 0.0)
        return logits

    def _hard_scan_add_integer(
        self, left: Tensor, right: Tensor, initial_carry: int, action: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Execute the selected learned finite-state relation on integers."""
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
        self, left: Tensor, right: Tensor, modulus: Tensor, action: Tensor,
        orientation: Tensor, invert_modulus: Tensor,
        reduction_carry: Tensor,
    ) -> Tensor:
        mask = (1 << self.bit_width) - 1
        raw, _ = self._hard_scan_add_integer(left, right, 0, action)
        reduction_operand = torch.where(
            invert_modulus,
            (~modulus) & mask,
            modulus,
        )
        difference, carry = self._hard_scan_add_integer(
            raw,
            reduction_operand,
            int(reduction_carry.item()),
            action,
        )
        subtract = torch.where(orientation, carry, 1 - carry).bool()
        return torch.where(subtract, difference, raw)

    def _hard_square_integer(
        self, value: Tensor, modulus: Tensor, action: Tensor,
        orientation: Tensor, invert_modulus: Tensor,
        reduction_carry: Tensor, scan_direction: Tensor,
        bit_gate: Tensor,
    ) -> Tensor:
        accumulator = torch.zeros_like(value)
        forward_scan = bool(scan_direction.item())
        normal_gate = bool(bit_gate.item())
        for scan_step in range(self.bit_width):
            position = (
                self.bit_width - 1 - scan_step
                if forward_scan
                else scan_step
            )
            doubled = self._hard_modadd_integer(
                accumulator,
                accumulator,
                modulus,
                action,
                orientation,
                invert_modulus,
                reduction_carry,
            )
            added = self._hard_modadd_integer(
                doubled,
                value,
                modulus,
                action,
                orientation,
                invert_modulus,
                reduction_carry,
            )
            bit = ((value >> position) & 1).bool()
            if not normal_gate:
                bit = ~bit
            accumulator = torch.where(bit, added, doubled)
        return accumulator

    def _hard_execute(
        self, value: Tensor, modulus: Tensor, t_values: Tensor, selected: int
    ) -> Tensor:
        index = torch.tensor([selected], device=value.device)
        table = self._transition_tables(index)[0]
        action = table.reshape(8, 4).argmax(dim=-1)
        orientation = self._programs(index)[0, 2].bool()
        controls = self._controls(index)
        invert_modulus, reduction_carry, scan_direction, bit_gate = (
            control[0].bool() for control in controls
        )
        terminal = torch.zeros_like(value)
        state = value
        for outer_step in range(int(t_values.max().item())):
            candidate = self._hard_square_integer(
                state,
                modulus,
                action,
                orientation,
                invert_modulus,
                reduction_carry,
                scan_direction,
                bit_gate,
            )
            terminal = torch.where(t_values == outer_step + 1, candidate, terminal)
            state = torch.where(t_values > outer_step, candidate, state)
        return terminal

    def _decode_hard_integer(self, value: Tensor) -> Tensor:
        decimal_powers = 10 ** torch.arange(
            self.decimal_width, device=value.device
        )
        digits = (value[:, None] // decimal_powers[None]).remainder(10)
        logits = self.beta_logits.new_full(
            (value.shape[0], self.decimal_width, NUM_DIGITS), -16.0
        )
        logits.scatter_(2, digits.unsqueeze(-1), 0.0)
        return logits

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, object]]:
        if attention_mask is None:
            attention_mask = input_ids != PAD
        valid = attention_mask.bool()
        modulus_integer, value_integer, t_values = self._parse(input_ids, valid)
        if not self.training:
            log_weights = self._program_log_weights()
            selected = int(log_weights.argmax().item())
            terminal_integer = self._hard_execute(
                value_integer, modulus_integer, t_values, selected
            )
            terminal = self._bits(terminal_integer)[None]
            digit_logits_lsd = self._decode_hard_integer(terminal_integer)
            logits = self._place_logits(
                digit_logits_lsd, valid.sum(dim=1), input_ids.shape[1]
            )
            return logits, {
                "program_bits": terminal,
                "program_log_weights": log_weights,
                "program_decimal_logits_lsd": terminal.new_zeros(
                    1, terminal.shape[1], self.decimal_width, NUM_DIGITS
                ),
                "bit_width": self.bit_width,
                "t_values": t_values,
            }
        initial = self._bits(value_integer)
        modulus_bits = self._bits(modulus_integer)
        log_weights = self._program_log_weights()
        indices = torch.arange(self.particles, device=input_ids.device)
        table = self._transition_tables(indices)
        orientation = self._programs(indices)[:, 2]
        branch = torch.stack((1.0 - orientation, orientation), dim=-1).to(
            table.dtype
        )
        controls = self._controls(indices)
        invert_modulus, reduction_carry, scan_direction, bit_gate = controls

        # Use the shallowest endpoint represented in this evaluator-owned batch.
        # Easy supplies T=1, while public variable-N Medium starts at T=2 and
        # private Hard is deliberately unspecified.  One globally selected rule
        # is still composed unchanged at every larger evaluation depth.
        supervision_depth = int(t_values.min().item())
        supervision_mask = t_values == supervision_depth
        supervision_indices = supervision_mask.nonzero(
            as_tuple=False
        ).flatten()
        terminal = initial.new_zeros(
            self.particles,
            initial.shape[0],
            self.bit_width,
            2,
        )
        if supervision_indices.numel() > 0:
            largest_modulus = int(
                modulus_integer[supervision_indices].max().item()
            )
            arithmetic_width = max(2, largest_modulus.bit_length() + 1)
            value = initial[
                supervision_indices, :arithmetic_width
            ][None].expand(self.particles, -1, -1, -1)
            modulus = modulus_bits[
                supervision_indices, :arithmetic_width
            ][None].expand_as(value)
            candidate = value
            for outer_step in range(supervision_depth):
                if outer_step > 0:
                    candidate = candidate.detach()
                candidate = self._square(
                    candidate,
                    modulus,
                    table,
                    branch,
                    invert_modulus,
                    reduction_carry,
                    scan_direction,
                    bit_gate,
                )
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

        if self.training:
            program_decimal_logits = self._decode_soft(terminal)
            decimal_probability = F.softmax(
                program_decimal_logits, dim=-1
            )
            mixture = (
                log_weights.exp()[:, None, None, None]
                * decimal_probability
            ).sum(dim=0)
            digit_logits_lsd = mixture.clamp_min(1e-8).log()
        logits = self._place_logits(
            digit_logits_lsd, valid.sum(dim=1), input_ids.shape[1]
        )
        return logits, {
            "program_bits": terminal,
            "program_log_weights": log_weights,
            "program_decimal_logits_lsd": program_decimal_logits,
            "bit_width": self.bit_width,
            "t_values": t_values,
            "supervision_mask": supervision_mask,
        }


def _target_integer(labels: Tensor, valid: Tensor) -> Tensor:
    value = torch.zeros(
        labels.shape[0], device=labels.device, dtype=torch.long
    )
    for position in range(labels.shape[1]):
        digit = (labels[:, position] - DIGIT_OFFSET).clamp(0, 9)
        value = torch.where(
            valid[:, position], value * 10 + digit, value
        )
    return value


def token_training_loss(batch: TokenLossBatch) -> Tensor:
    valid = batch.valid_mask
    endpoint = F.cross_entropy(
        batch.logits.transpose(1, 2),
        batch.labels,
        ignore_index=-100,
        reduction="none",
    )
    supervision = batch.auxiliary["supervision_mask"].to(endpoint.dtype)
    per_example = (endpoint * valid).sum(dim=1) / valid.sum(
        dim=1
    ).clamp_min(1)
    direct = (per_example * supervision).sum() / supervision.sum().clamp_min(1)
    target = _target_integer(batch.labels, valid)
    bit_width = int(batch.auxiliary["bit_width"])
    position = torch.arange(bit_width, device=target.device)
    target_bits = ((target[:, None] >> position[None]) & 1).long()
    probability = batch.auxiliary["program_bits"].clamp_min(1e-12)
    observed = probability.gather(
        3,
        target_bits[None, :, :, None].expand(
            probability.shape[0], -1, -1, 1
        ),
    ).squeeze(3).log().sum(dim=2)
    log_weights = batch.auxiliary["program_log_weights"].float()
    endpoint_nll = -torch.logsumexp(
        log_weights[:, None] + observed, dim=0
    )
    program_endpoint = (endpoint_nll * supervision).sum() / supervision.sum().clamp_min(1)
    return DIRECT_LOSS_WEIGHT * direct + BIT_LOSS_WEIGHT * program_endpoint


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
        elapsed_fraction = min(
            (time.monotonic() - self.started_at) / self.budget_seconds, 1.0
        )
        if elapsed_fraction < 0.92:
            multiplier = 1.0
        else:
            progress = (elapsed_fraction - 0.92) / 0.08
            multiplier = 0.2 + 0.8 * 0.5 * (
                1.0 + math.cos(math.pi * progress)
            )
        for group in self.optimizer.param_groups:
            group["lr"] = group["base_lr"] * multiplier


def build_model(spec: ModelSpec) -> Model:
    model = Model(spec)
    assert_model_state(model, spec)
    return model


def build_optimizer(model: Model, spec: OptimizerSpec) -> OptimizerBundle:
    posterior = [
        model.beta_logits,
        model.gamma_logits,
        model.branch_logits,
        model.invert_modulus_logits,
        model.reduction_carry_logits,
        model.scan_direction_logits,
        model.bit_gate_logits,
    ]
    structured_ids = {id(parameter) for parameter in posterior}
    decoder = [
        parameter
        for parameter in model.parameters()
        if id(parameter) not in structured_ids
    ]
    optimizer = torch.optim.AdamW(
        [
            {
                "params": posterior,
                "weight_decay": 0.0,
                "lr": POSTERIOR_LR,
                "base_lr": POSTERIOR_LR,
            },
            {
                "params": decoder,
                "weight_decay": 0.01,
                "lr": DECODER_LR,
                "base_lr": DECODER_LR,
            },
        ],
        lr=POSTERIOR_LR,
        betas=(0.9, 0.95),
        eps=1e-8,
        fused=spec.device_type == "cuda",
    )
    return OptimizerBundle(
        optimizer=optimizer,
        scheduler=WallClockSchedule(
            optimizer, spec.training_time_seconds
        ),
    )


SUBMISSION = Submission(
    build_model=build_model,
    build_optimizer=build_optimizer,
    token_training_loss=token_training_loss,
    batch_size=1,
    eval_batch_size=512,
)
