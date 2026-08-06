"""Exhaustive width-two checks for the observable Markov belief state."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
import torch.nn.functional as F


def _load_module():
    path = Path(__file__).with_name("submission.py")
    spec = importlib.util.spec_from_file_location("observable_belief_probe", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load submission")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    torch.manual_seed(74)
    module = _load_module()
    square = module.ObservableBeliefSquare(width=2)
    register = F.one_hot(torch.randint(0, 10, (3, 2)), 10).float()
    modulus = F.one_hot(torch.randint(0, 10, (3, 2)), 10).float()
    marginals_lsd, beliefs, chain = square(register, modulus)
    start, transition = chain

    start_error = float((start.exp().sum(-1) - 1.0).abs().max().detach())
    transition_error = float(
        (transition.exp().sum(-1) - 1.0).abs().max().detach()
    )
    marginal_error = float(
        (marginals_lsd.sum(-1) - 1.0).abs().max().detach()
    )

    sequences = torch.cartesian_prod(torch.arange(10), torch.arange(10))
    brute_scores = (
        start[:, sequences[:, 0]]
        + transition[:, 0, sequences[:, 0], sequences[:, 1]]
    )
    partition_error = float(torch.logsumexp(brute_scores, dim=1).abs().max().detach())
    brute_path = sequences[brute_scores.argmax(dim=1)]
    dynamic_path = module.Model._viterbi(start, transition)
    viterbi_equal = bool(torch.equal(brute_path, dynamic_path))

    target = torch.tensor([1, 4, 7])
    joint = brute_scores.exp().reshape(3, 10, 10)
    observed_probability = joint[torch.arange(3), :, target].sum(dim=1)
    loss = -observed_probability.clamp_min(1e-30).log().mean()
    loss.backward()
    gradients = [
        parameter.grad
        for parameter in square.parameters()
        if parameter.requires_grad
    ]
    finite_gradients = all(
        gradient is not None and torch.isfinite(gradient).all().item()
        for gradient in gradients
    )
    nonzero_gradients = sum(
        gradient is not None and bool(gradient.abs().max() > 0)
        for gradient in gradients
    )

    print(f"belief_updates={len(beliefs)}")
    print(f"start_normalization_error={start_error:.9e}")
    print(f"transition_normalization_error={transition_error:.9e}")
    print(f"marginal_normalization_error={marginal_error:.9e}")
    print(f"partition_error={partition_error:.9e}")
    print(f"viterbi_matches_exhaustive={viterbi_equal}")
    print(f"finite_gradients={finite_gradients}")
    print(f"nonzero_gradient_tensors={nonzero_gradients}/{len(gradients)}")
    if (
        start_error > 1e-6
        or transition_error > 1e-6
        or marginal_error > 1e-6
        or partition_error > 1e-6
        or not viterbi_equal
        or not finite_gradients
    ):
        raise AssertionError("observable belief-state diagnostic failed")


if __name__ == "__main__":
    main()
