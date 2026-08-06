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

    def test_seed74_starts_wrong_but_learned_map_has_t64_capacity(self) -> None:
        torch.manual_seed(74)
        model = MODULE.build_model(ModelSpec(17, 21, 500_000_000))
        initial = int(model._program_log_weights().argmax())
        initial_program = model._programs(torch.tensor([initial]))[0]
        initial_controls = model._controls(torch.tensor([initial]))
        initial_signature = (
            float(initial_program[0]),
            float(initial_program[1]),
            int(initial_program[2]),
            *(int(control[0]) for control in initial_controls),
            *(int(choice) for choice in model.microcode_choices[initial]),
        )
        canonical_signature = (-1.0, -2.0, 1, 1, 1, 1, 1, 0, 1, 1)
        self.assertNotEqual(initial_signature, canonical_signature)

        learned_choices = (
            (model.beta_logits, MODULE.BETA_VALUES.index(-1.0)),
            (model.gamma_logits, MODULE.GAMMA_VALUES.index(-2.0)),
            (model.branch_logits, 1),
            (model.invert_modulus_logits, 1),
            (model.reduction_carry_logits, 1),
            (model.scan_direction_logits, 1),
            (model.slot1_rhs_logits, 0),
            (model.slot2_rhs_logits, 1),
            (model.commit_logits, 1),
        )
        with torch.no_grad():
            for logits, choice in learned_choices:
                logits.fill_(-10.0)
                logits[choice] = 10.0
        selected = int(model._program_log_weights().argmax())
        learned_program = model._programs(torch.tensor([selected]))[0]
        learned_controls = model._controls(torch.tensor([selected]))
        learned_signature = (
            float(learned_program[0]),
            float(learned_program[1]),
            int(learned_program[2]),
            *(int(control[0]) for control in learned_controls),
            *(int(choice) for choice in model.microcode_choices[selected]),
        )
        self.assertEqual(learned_signature, canonical_signature)
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

    def test_learned_max_width_program_covers_every_hard_ladder_rung(self) -> None:
        torch.manual_seed(74)
        model = MODULE.build_model(ModelSpec(17, 21, 500_000_000))
        learned_choices = (
            (model.beta_logits, MODULE.BETA_VALUES.index(-1.0)),
            (model.gamma_logits, MODULE.GAMMA_VALUES.index(-2.0)),
            (model.branch_logits, 1),
            (model.invert_modulus_logits, 1),
            (model.reduction_carry_logits, 1),
            (model.scan_direction_logits, 1),
            (model.slot1_rhs_logits, 0),
            (model.slot2_rhs_logits, 1),
            (model.commit_logits, 1),
        )
        with torch.no_grad():
            for logits, choice in learned_choices:
                logits.fill_(-10.0)
                logits[choice] = 10.0
        model.eval()

        ladder = (1, 2, 4, 8, 16, 32, 64)
        cases = [
            (99_999_989 - 2 * row, 91_357_913 - 97 * row, time_steps)
            for row, time_steps in enumerate(ladder)
        ]
        prompts = []
        for modulus, value, time_steps in cases:
            prompts.append(
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
        input_ids = torch.zeros(len(prompts), 21, dtype=torch.long)
        attention = torch.zeros_like(input_ids, dtype=torch.bool)
        for row, prompt in enumerate(prompts):
            input_ids[row, : len(prompt)] = torch.tensor(prompt)
            attention[row, : len(prompt)] = True

        with torch.no_grad():
            logits, _ = model(input_ids, attention)
        predictions = logits.argmax(dim=-1)
        for row, ((modulus, value, time_steps), prompt) in enumerate(
            zip(cases, prompts, strict=True)
        ):
            expected = str(pow(value, 1 << time_steps, modulus))
            target = torch.tensor(
                [MODULE.DIGIT_OFFSET + int(digit) for digit in expected]
            )
            start = len(prompt) - len(expected)
            self.assertTrue(
                torch.equal(predictions[row, start : len(prompt)], target),
                msg=f"failed max-width T={time_steps}",
            )


if __name__ == "__main__":
    unittest.main()
