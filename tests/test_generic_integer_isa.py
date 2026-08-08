import importlib.util
import unittest
from pathlib import Path

import torch

from benchmark import ModelSpec


PATH = (
    Path(__file__).parents[1]
    / "submissions"
    / "generic_integer_isa"
    / "submission.py"
)
SPEC = importlib.util.spec_from_file_location("generic_integer_isa", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class GenericIntegerIsaTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(74)
        self.model = MODULE.build_model(ModelSpec(17, 17, 500_000_000))

    def _select(self, choices: list[int]) -> None:
        with torch.no_grad():
            self.model.instruction_logits.zero_()
            for slot, choice in enumerate(choices):
                self.model.instruction_logits[slot, choice] = 10.0

    def test_complete_space_and_wrong_initial_map(self) -> None:
        self.assertEqual(tuple(self.model.programs.shape), (1_048_576, 5))
        initial = self.model.programs[
            self.model._program_log_weights().argmax()
        ].tolist()
        self.assertEqual(initial, [14, 2, 9, 13, 0])
        self.assertNotEqual(initial, [3, 9, 0, 0, 0])

    def test_explicit_modulo_programs_cover_changed_recurrences(self) -> None:
        modulus = torch.tensor([323, 437])
        source = torch.tensor([17, 29])
        time_steps = torch.tensor([1, 3])
        programs = {
            "square": [3, 9, 0, 0, 0],
            "cube": [3, 9, 3, 9, 0],
            "affine": [1, 1, 4, 9, 0],
            "quadratic_plus_two": [3, 1, 4, 4, 9],
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
            self._select(choices)
            actual = self.model._execute_map(source, modulus, time_steps)
            expected = torch.tensor(
                [
                    iterate(kind, int(x), int(n), int(t))
                    for x, n, t in zip(source, modulus, time_steps)
                ]
            )
            self.assertTrue(torch.equal(actual, expected), kind)

    def test_endpoint_gradient_reaches_all_instruction_logits(self) -> None:
        source = torch.tensor([17, 29, 41])
        modulus = torch.tensor([323, 437, 667])
        outputs = self.model._execute_programs(source, modulus, 1)
        target = source.square().remainder(modulus)
        exact = (outputs == target[None]).float()
        likelihood = (exact * -1e-4 + (1.0 - exact) * -12.0).sum(1)
        loss = -torch.logsumexp(
            self.model._program_log_weights() + likelihood, dim=0
        )
        loss.backward()
        gradient = self.model.instruction_logits.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertEqual(int((gradient != 0).sum()), MODULE.SLOTS * MODULE.CHOICES)


if __name__ == "__main__":
    unittest.main()
