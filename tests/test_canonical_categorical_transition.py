import importlib.util
import unittest
from pathlib import Path

import torch
from torch import nn

from benchmark import ModelSpec


PATH = (
    Path(__file__).parents[1]
    / "submissions"
    / "canonical_categorical_transition"
    / "submission.py"
)
SPEC = importlib.util.spec_from_file_location(
    "canonical_categorical_transition_submission", PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class CountingProgram(nn.Module):
    """A differentiable transition used to audit endpoint connectivity."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = width
        self.weight = nn.Parameter(torch.randn(1, 1, MODULE.NUM_DIGITS))
        self.calls = 0

    def forward(self, register, modulus, temperature, hardness):
        del modulus, temperature, hardness
        self.calls += 1
        if self.calls == 1:
            logits = self.weight.expand(register.shape[0], self.width, -1)
        else:
            # Only the first transition reads the parameter. The terminal
            # loss can reach it only through every intervening outer state.
            logits = register.clamp_min(1e-6).log()
        candidate = logits.softmax(dim=-1)
        phases = tuple(logits for _ in range(MODULE.COMPUTE_SWEEPS))
        controller = torch.full(
            (
                register.shape[0],
                1,
                self.width,
                MODULE.CONTROL_STATES,
            ),
            1.0 / MODULE.CONTROL_STATES,
            device=register.device,
        )
        digit_soft = candidate[:, None]
        return candidate, logits, phases, controller, digit_soft


class CanonicalCategoricalTransitionTest(unittest.TestCase):
    def test_even_prompt_length_preserves_every_decimal_digit(self) -> None:
        model = MODULE.build_model(ModelSpec(17, 10, 500_000_000))
        self.assertEqual(model.width, 3)
        inputs = torch.tensor(
            [[MODULE.N_MARK, 10, 9, 10, MODULE.X_MARK, 9, 7, 10,
              MODULE.T_MARK, 8]]
        )
        modulus, value, time_steps = model._parse(
            inputs, torch.ones_like(inputs).bool()
        )
        self.assertEqual(modulus.argmax(dim=-1).tolist(), [[3, 2, 3]])
        self.assertEqual(value.argmax(dim=-1).tolist(), [[3, 0, 2]])
        self.assertEqual(time_steps.tolist(), [1])

    def test_training_reaches_endpoint_beyond_eight_recurrences(self) -> None:
        torch.manual_seed(0)
        model = MODULE.build_model(ModelSpec(17, 12, 500_000_000))
        program = CountingProgram(model.width)
        model.program = program
        model.train()

        # N=5, x=2, T=16. The former eight-step training cap left this row's
        # terminal logits equal to a constant zero tensor.
        inputs = torch.tensor(
            [[MODULE.N_MARK, 12, MODULE.X_MARK, 9, MODULE.T_MARK, 8, 13]]
        )
        logits, auxiliary = model(inputs, torch.ones_like(inputs).bool())

        self.assertEqual(int(auxiliary["t_values"].item()), 16)
        self.assertEqual(program.calls, 16)
        loss = logits[..., MODULE.DIGIT_OFFSET].sum()
        loss.backward()
        self.assertIsNotNone(program.weight.grad)
        self.assertGreater(float(program.weight.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
