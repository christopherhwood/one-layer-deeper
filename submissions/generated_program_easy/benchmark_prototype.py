"""Local benchmark bridge for endpoint-driven program generation.

This file deliberately imports the proven interpreter while the evolutionary
optimizer is being calibrated.  The final submission will vendor the code into
one self-contained file after the generation gate passes.
"""

from __future__ import annotations

import math
import time

import torch
import torch.nn.functional as F
from torch import Tensor

from benchmark import ModelSpec, OptimizerBundle, OptimizerSpec, Submission
from submissions.binary_relation_hard import submission as BASE


POPULATION = 96
ELITE_COUNT = 24
RESTART_COUNT = 19
EVOLVE_EVERY = 1
FITNESS_DECAY = 0.5
MUTATION_LOGIT = 4.0
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


def token_training_loss(batch) -> Tensor:
    """Endpoint loss with a dense program-ranking surrogate.

    The ordinary marginal likelihood has essentially no slope until one whole
    program gets every output bit right.  The extra term still uses only the
    endpoint, but rewards a candidate independently for every correct endpoint
    bit.  Its gradient through the posterior is therefore a useful fitness
    ranking even before a perfect program exists.
    """
    base_loss = BASE.token_training_loss(batch)
    valid = batch.valid_mask
    target = BASE._target_integer(batch.labels, valid)
    bit_width = int(batch.auxiliary["bit_width"])
    position = torch.arange(bit_width, device=target.device)
    target_bits = ((target[:, None] >> position[None]) & 1).long()
    probability = batch.auxiliary["program_bits"].clamp_min(1e-12)
    observed = probability.gather(
        3,
        target_bits[None, :, :, None].expand(
            probability.shape[0], -1, -1, 1
        ),
    ).squeeze(3)
    per_program_bit_score = observed.log().mean(dim=2)
    weights = batch.auxiliary["program_log_weights"].float().exp()
    dense_per_example = -(
        weights[:, None] * per_program_bit_score
    ).sum(dim=0)
    supervision = batch.auxiliary["supervision_mask"].to(
        dense_per_example.dtype
    )
    dense_loss = (dense_per_example * supervision).sum() / supervision.sum().clamp_min(1)
    return base_loss + dense_loss


def structured_parameters(model) -> list[torch.nn.Parameter]:
    return [getattr(model, name) for name in FIELD_NAMES]


def build_model(spec: ModelSpec):
    previous = BASE.PROGRAMS
    BASE.PROGRAMS = POPULATION
    try:
        model = BASE.build_model(spec)
    finally:
        BASE.PROGRAMS = previous
    model.register_buffer(
        "evolution_fitness", torch.zeros(POPULATION), persistent=True
    )
    model.register_buffer(
        "evolution_generation", torch.zeros((), dtype=torch.long), persistent=True
    )
    model.evolution_enabled = True
    return model


class EvolutionaryOptimizer(torch.optim.Optimizer):
    def __init__(self, model, groups: list[dict]) -> None:
        self.model = model
        self.inner = torch.optim.AdamW(
            groups, betas=(0.9, 0.95), eps=1e-8
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
    def _record_fitness(self) -> None:
        gradient = self.model.program_logits.grad
        if gradient is None:
            return
        score = -gradient.float()
        order = score.argsort()
        rank = torch.empty_like(score)
        rank[order] = torch.linspace(
            0.0, 1.0, score.numel(), device=score.device
        )
        self.model.evolution_fitness.mul_(FITNESS_DECAY).add_(
            rank * (1.0 - FITNESS_DECAY)
        )

    @torch.no_grad()
    def _zero_rows(self, parameter: torch.nn.Parameter, rows: Tensor) -> None:
        state = self.inner.state.get(parameter, {})
        for value in state.values():
            if torch.is_tensor(value) and value.shape == parameter.shape:
                value[rows] = 0

    @torch.no_grad()
    def _evolve(self) -> None:
        model = self.model
        device = model.program_logits.device
        generator = self._rng(device)
        order = model.evolution_fitness.argsort(descending=True)
        elites = order[:ELITE_COUNT]
        children = order[ELITE_COUNT:]
        parent_slot = torch.randint(
            ELITE_COUNT, (children.numel(),), device=device, generator=generator
        )
        parents = elites[parent_slot]
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
            len(fields), (mutate_rows.numel(),), device=device, generator=generator
        )
        for field_index, parameter in enumerate(fields):
            selected = mutate_rows[field_choice == field_index]
            if selected.numel() == 0:
                continue
            old = parameter[selected].argmax(dim=-1)
            offset = torch.randint(
                1,
                parameter.shape[1],
                (selected.numel(),),
                device=device,
                generator=generator,
            )
            new = (old + offset).remainder(parameter.shape[1])
            parameter[selected].fill_(-MUTATION_LOGIT)
            parameter[selected, new] = MUTATION_LOGIT

        # Occasional second-field mutations escape one-coordinate local basins.
        second_mask = torch.rand(
            mutate_rows.numel(), device=device, generator=generator
        ) < 0.15
        second_rows = mutate_rows[second_mask]
        if second_rows.numel():
            second_fields = torch.randint(
                len(fields),
                (second_rows.numel(),),
                device=device,
                generator=generator,
            )
            for field_index, parameter in enumerate(fields):
                selected = second_rows[second_fields == field_index]
                if selected.numel() == 0:
                    continue
                old = parameter[selected].argmax(dim=-1)
                offset = torch.randint(
                    1,
                    parameter.shape[1],
                    (selected.numel(),),
                    device=device,
                    generator=generator,
                )
                new = (old + offset).remainder(parameter.shape[1])
                parameter[selected].fill_(-MUTATION_LOGIT)
                parameter[selected, new] = MUTATION_LOGIT

        for parameter in fields:
            self._zero_rows(parameter, children)
        # The posterior is reset to uniform after every generation, so its
        # momentum must also be reset for every row.  Otherwise Adam's stale
        # moments silently carry the previous generation's ranking forward.
        all_rows = torch.arange(model.particles, device=device)
        self._zero_rows(model.program_logits, all_rows)
        model.program_logits.zero_()
        model.evolution_fitness.zero_()
        model.evolution_generation.add_(1)

    @torch.no_grad()
    def step(self, closure=None):
        self._record_fitness()
        result = self.inner.step(closure)
        self.steps += 1
        if self.model.evolution_enabled and self.steps % EVOLVE_EVERY == 0:
            self._evolve()
        return result


class Schedule:
    def __init__(self, model, optimizer: EvolutionaryOptimizer, budget: float) -> None:
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
            multiplier = 0.2 + 0.8 * 0.5 * (
                1.0 + math.cos(math.pi * progress)
            )
        for group in self.optimizer.param_groups:
            group["lr"] = group["base_lr"] * multiplier


def build_optimizer(model, spec: OptimizerSpec) -> OptimizerBundle:
    posterior = [model.program_logits]
    relation = structured_parameters(model)
    structured_ids = {id(parameter) for parameter in (*posterior, *relation)}
    decoder = [
        parameter
        for parameter in model.parameters()
        if id(parameter) not in structured_ids
    ]
    groups = [
        {
            "params": posterior,
            "lr": BASE.POSTERIOR_LR,
            "base_lr": BASE.POSTERIOR_LR,
            "weight_decay": 0.0,
        },
        {
            "params": relation,
            "lr": BASE.RELATION_LR,
            "base_lr": BASE.RELATION_LR,
            "weight_decay": 0.0,
        },
        {
            "params": decoder,
            "lr": BASE.DECODER_LR,
            "base_lr": BASE.DECODER_LR,
            "weight_decay": 0.01,
        },
    ]
    optimizer = EvolutionaryOptimizer(model, groups)
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
