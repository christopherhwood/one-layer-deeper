import importlib.util
import unittest
from pathlib import Path

import torch

from benchmark import ModelSpec, TokenLossBatch


PATH = (
    Path(__file__).parents[1]
    / "submissions"
    / "binary_relation_hard_control"
    / "submission.py"
)
SPEC = importlib.util.spec_from_file_location(
    "binary_relation_hard_control_submission", PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class BinaryRelationHardControlSubmissionTest(unittest.TestCase):
    def _loss(self, model, time_token: int) -> torch.Tensor:
        inputs = torch.tensor([[2, 8, 11, 10, 3, 8, 9, 4, time_token, 0]])
        logits, auxiliary = model(inputs, inputs != 0)
        positions = torch.tensor([[6, 7, 8]])
        labels = torch.tensor([[8, 11, 11]])
        return MODULE.token_training_loss(
            TokenLossBatch(
                logits[:, positions[0]],
                labels,
                torch.ones_like(labels, dtype=torch.bool),
                positions,
                auxiliary,
            )
        )

    def test_t2_only_endpoint_reaches_all_trainable_state(self) -> None:
        torch.manual_seed(0)
        model = MODULE.build_model(ModelSpec(17, 10, 500_000_000))
        loss = self._loss(model, time_token=9)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(
            all(
                parameter.grad is not None
                and torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            )
        )

    def test_seed74_has_canonical_particle_and_t64_capacity(self) -> None:
        torch.manual_seed(74)
        model = MODULE.build_model(ModelSpec(17, 21, 500_000_000))
        programs = model._programs()
        controls = model._controls(torch.arange(model.particles))
        canonical_mask = (
            (programs[:, 0] == -1)
            & (programs[:, 1] == -2)
            & (programs[:, 2] == 1)
        )
        for control in controls:
            canonical_mask &= control == 1
        canonical = canonical_mask.nonzero(as_tuple=False).flatten()
        self.assertGreaterEqual(canonical.numel(), 1)
        with torch.no_grad():
            model.program_logits.fill_(-10.0)
            model.program_logits[canonical[0]] = 10.0
        model.eval()

        modulus = 99_999_989
        value = 99_999_988
        time_steps = 64
        prompt = (
            [MODULE.N_MARK]
            + [MODULE.DIGIT_OFFSET + int(digit) for digit in str(modulus)]
            + [MODULE.X_MARK]
            + [MODULE.DIGIT_OFFSET + int(digit) for digit in str(value)]
            + [MODULE.T_MARK]
            + [
                MODULE.DIGIT_OFFSET + int(digit)
                for digit in str(time_steps)
            ]
        )
        self.assertEqual(len(prompt), 21)
        input_ids = torch.tensor([prompt])
        with torch.no_grad():
            logits, _ = model(input_ids, torch.ones_like(input_ids).bool())
        expected = str(pow(value, 1 << time_steps, modulus))
        predictions = logits.argmax(dim=-1)[0, -len(expected) :]
        target = torch.tensor(
            [MODULE.DIGIT_OFFSET + int(digit) for digit in expected]
        )
        self.assertTrue(torch.equal(predictions, target))


if __name__ == "__main__":
    unittest.main()
