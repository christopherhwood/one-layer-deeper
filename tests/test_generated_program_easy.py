import importlib.util
import unittest
from pathlib import Path

import torch

from benchmark import ModelSpec, OptimizerSpec, TokenLossBatch


PATH = (
    Path(__file__).parents[1]
    / "submissions"
    / "generated_program_easy"
    / "submission.py"
)
SPEC = importlib.util.spec_from_file_location("generated_program_easy", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class GeneratedProgramEasyTest(unittest.TestCase):
    @staticmethod
    def _canonical(model) -> torch.Tensor:
        relation = model._programs()
        recurrence = model._recurrence_programs()
        return (
            (relation[:, 0] == -1)
            & (relation[:, 1] == -2)
            & (relation[:, 2] == 1)
            & (recurrence[:, 0] == 0)
            & (recurrence[:, 1] == 0)
            & (recurrence[:, 2] == 1)
            & (recurrence[:, 3] == 0)
            & (recurrence[:, 4] == 0)
        ).nonzero(as_tuple=False).flatten()

    @staticmethod
    def _teaching_batch():
        pairs = [(modulus, value) for modulus in range(2, 13) for value in range(modulus)]

        def prompt(modulus: int, value: int) -> list[int]:
            tokens = [
                MODULE.N_MARK,
                *[MODULE.DIGIT_OFFSET + int(d) for d in str(modulus)],
                MODULE.X_MARK,
                *[MODULE.DIGIT_OFFSET + int(d) for d in str(value)],
                MODULE.T_MARK,
                MODULE.DIGIT_OFFSET + 1,
            ]
            return tokens + [MODULE.PAD] * (13 - len(tokens))

        inputs = torch.tensor([prompt(n, x) for n, x in pairs])
        labels = torch.full((len(pairs), 2), -100, dtype=torch.long)
        valid = torch.zeros_like(labels, dtype=torch.bool)
        for row, (modulus, value) in enumerate(pairs):
            answer = str(value * value % modulus)
            labels[row, -len(answer) :] = torch.tensor(
                [MODULE.DIGIT_OFFSET + int(d) for d in answer]
            )
            valid[row, -len(answer) :] = True
        return inputs, labels, valid

    def test_endpoint_feedback_creates_absent_program_and_selects_it(self) -> None:
        torch.manual_seed(74)
        torch.set_num_threads(1)
        model = MODULE.build_model(ModelSpec(17, 13, 500_000_000))
        model.train()
        self.assertEqual(self._canonical(model).numel(), 0)
        inputs, labels, valid = self._teaching_batch()
        bundle = MODULE.build_optimizer(model, OptimizerSpec(1000.0, "cpu"))
        optimizer = bundle.optimizer

        for _ in range(10):
            optimizer.zero_grad()
            logits, auxiliary = model(inputs, inputs != MODULE.PAD)
            loss = MODULE.token_training_loss(
                TokenLossBatch(logits[:, :2], labels, valid, None, auxiliary)
            )
            loss.backward()
            optimizer.step()
            if self._canonical(model).numel():
                break
        created = self._canonical(model)
        self.assertGreater(created.numel(), 0)

        model.evolution_enabled = False
        for _ in range(20):
            optimizer.zero_grad()
            logits, auxiliary = model(inputs, inputs != MODULE.PAD)
            loss = MODULE.token_training_loss(
                TokenLossBatch(logits[:, :2], labels, valid, None, auxiliary)
            )
            loss.backward()
            optimizer.step()
        self.assertIn(int(model.program_logits.argmax()), created.tolist())

        model.eval()
        moduli = torch.arange(13, 24)
        values = moduli - 1
        expected = values.square().remainder(moduli)
        with torch.no_grad():
            actual = model._hard_execute(
                values,
                moduli,
                torch.ones_like(values),
                int(model.program_logits.argmax()),
            )
        self.assertTrue(torch.equal(actual, expected))


if __name__ == "__main__":
    unittest.main()
