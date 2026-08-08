import importlib.util
import unittest
from pathlib import Path

import torch
from torch import nn

from benchmark import ModelSpec


PATH = (
    Path(__file__).parents[1]
    / "submissions"
    / "neural_gpu_recurrent"
    / "submission.py"
)
SPEC = importlib.util.spec_from_file_location(
    "neural_gpu_recurrent_submission", PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class CountingTransition(nn.Module):
    def __init__(self, digits: int) -> None:
        super().__init__()
        self.digits = digits
        self.weight = nn.Parameter(torch.randn(1, 1, MODULE.NUM_DIGITS))
        self.calls = 0

    def forward(self, source, modulus, temperature):
        del modulus, temperature
        self.calls += 1
        if self.calls == 1:
            logits = self.weight.expand(source.shape[0], self.digits, -1)
            candidate = logits.softmax(dim=-1)
        else:
            candidate = source
            logits = source.clamp_min(1e-6).log()
        return candidate, tuple(logits for _ in range(MODULE.SWEEPS))


class NeuralGpuRecurrentTest(unittest.TestCase):
    def test_even_prompt_capacity_preserves_three_digit_fields(self) -> None:
        model = MODULE.build_model(ModelSpec(17, 10, 500_000_000))
        self.assertEqual(model.digits, 3)
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

    def test_training_reaches_sixteen_step_endpoint(self) -> None:
        torch.manual_seed(0)
        model = MODULE.build_model(ModelSpec(17, 15, 500_000_000))
        transition = CountingTransition(model.digits)
        model.transition = transition
        model.train()
        inputs = torch.tensor(
            [[MODULE.N_MARK, 12, MODULE.X_MARK, 9, MODULE.T_MARK, 8, 13]]
        )
        logits, _ = model(inputs, torch.ones_like(inputs).bool())
        self.assertEqual(transition.calls, 16)
        loss = logits[..., MODULE.DIGIT_OFFSET].sum()
        loss.backward()
        self.assertIsNotNone(transition.weight.grad)
        self.assertGreater(float(transition.weight.grad.abs().sum()), 0.0)

    def test_endpoint_heads_preserve_only_endpoint_information(self) -> None:
        torch.manual_seed(0)
        model = MODULE.build_model(ModelSpec(17, 10, 500_000_000))
        model.train()
        inputs = torch.tensor(
            [[MODULE.N_MARK, 10, 9, 10, MODULE.X_MARK, 9, 7, 10,
              MODULE.T_MARK, 8]]
        )
        _, auxiliary = model(inputs, torch.ones_like(inputs).bool())
        self.assertEqual(auxiliary["modulus_integer"].tolist(), [323])
        self.assertEqual(len(auxiliary["bit_logits"]), MODULE.SWEEPS)
        self.assertEqual(
            tuple(auxiliary["bit_logits"][0].shape),
            (1, model.bit_width, 2),
        )
        self.assertEqual(len(auxiliary["ordinal_logits"]), MODULE.SWEEPS)
        self.assertEqual(
            tuple(auxiliary["ordinal_logits"][0].shape),
            (1, MODULE.ORDINAL_BINS),
        )


if __name__ == "__main__":
    unittest.main()
