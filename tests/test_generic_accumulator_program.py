import importlib.util
import unittest
from pathlib import Path

import torch

from benchmark import ModelSpec


PATH = (
    Path(__file__).parents[1]
    / "submissions"
    / "generic_accumulator_program"
    / "submission.py"
)
SPEC = importlib.util.spec_from_file_location(
    "generic_accumulator_program_submission", PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class GenericAccumulatorProgramTest(unittest.TestCase):
    def _model_with_program(self, choices: list[int]):
        model = MODULE.build_model(ModelSpec(17, 13, 500_000_000))
        with torch.no_grad():
            model.instruction_logits.zero_()
            for slot, choice in enumerate(choices):
                model.instruction_logits[slot, choice] = 10.0
        return model

    def test_program_space_is_complete_and_initial_map_is_not_planted(self):
        torch.manual_seed(74)
        model = MODULE.build_model(ModelSpec(17, 13, 500_000_000))
        self.assertEqual(tuple(model.programs.shape), (65_536, 4))
        initial = model.programs[model._program_log_weights().argmax()].tolist()
        self.assertEqual(initial, [14, 2, 9, 13])
        self.assertNotEqual(initial, [12, 0, 0, 0])

    def test_same_isa_represents_changed_recurrences(self):
        modulus = torch.tensor([323, 437])
        value = torch.tensor([17, 29])
        time_steps = torch.tensor([1, 3])
        # choice encoding: opcode * 4 + rhs; ADD-x=4, ADD-1=5,
        # MUL-x=12, and choice 0 is a no-op.
        programs = {
            "square": [12, 0, 0, 0],
            "cube": [12, 12, 0, 0],
            "affine": [4, 4, 5, 0],
            "quadratic_plus_two": [12, 4, 5, 5],
        }

        def iterate(kind: str, x: int, n: int, depth: int) -> int:
            for _ in range(depth):
                if kind == "square":
                    x = x * x % n
                elif kind == "cube":
                    x = x * x * x % n
                elif kind == "affine":
                    x = (3 * x + 1) % n
                else:
                    x = (x * x + x + 2) % n
            return x

        for kind, choices in programs.items():
            model = self._model_with_program(choices)
            actual = model._execute_map(value, modulus, time_steps)
            expected = torch.tensor(
                [
                    iterate(kind, int(x), int(n), int(t))
                    for x, n, t in zip(value, modulus, time_steps)
                ]
            )
            self.assertTrue(torch.equal(actual, expected), kind)

    def test_endpoint_likelihood_reaches_every_instruction_logit(self):
        torch.manual_seed(74)
        model = MODULE.build_model(ModelSpec(17, 13, 500_000_000))
        modulus = torch.tensor([323, 437, 667])
        source = torch.tensor([17, 29, 41])
        outputs = model._execute_programs(source, modulus, 1)
        targets = source.square().remainder(modulus)
        exact = (outputs == targets[None]).float()
        log_likelihood = (exact * -1e-4 + (1.0 - exact) * -12.0).sum(1)
        loss = -torch.logsumexp(model._program_log_weights() + log_likelihood, 0)
        loss.backward()
        gradient = model.instruction_logits.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
