from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import torch

from benchmark import ModelSpec, OptimizerSpec, TokenLossBatch


PATH = (
    Path(__file__).parents[1]
    / "submissions"
    / "autoregressive_program_writer"
    / "submission.py"
)
SPEC = importlib.util.spec_from_file_location(
    "autoregressive_program_writer_submission", PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _teaching_batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pairs = [
        (modulus, value)
        for modulus in range(5, 37)
        for value in range(1, min(modulus, 8))
    ]

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


class AutoregressiveProgramWriterTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(74)
        torch.set_num_threads(1)

    def test_random_writer_has_no_program_table_or_winning_ticket(self) -> None:
        model = MODULE.build_model(ModelSpec(17, 13, 500_000_000))
        self.assertEqual(sum(p.numel() for p in model.parameters()), 1040)
        self.assertEqual(list(model.buffers()), [])
        self.assertLess(MODULE.SAMPLES, MODULE.CHOICES**MODULE.SLOTS)
        self.assertNotEqual(model._map_program().tolist(), [3, 9, 0, 0, 0])
        source = PATH.read_text()
        self.assertNotIn("PROGRAMS =", source)
        self.assertNotIn("torch.arange(PROGRAMS", source)
        self.assertNotIn(".backward(", source)
        self.assertNotIn("autograd.grad", source)

    def test_writer_samples_conditionally_and_viterbi_decodes(self) -> None:
        model = MODULE.build_model(ModelSpec(17, 13, 500_000_000))
        expected = [3, 9, 0, 0, 0]
        with torch.no_grad():
            model.start_logits.fill_(-20.0)
            model.start_logits[expected[0]] = 20.0
            model.transition_logits.fill_(-20.0)
            for slot in range(1, MODULE.SLOTS):
                model.transition_logits[
                    slot - 1, expected[slot - 1], expected[slot]
                ] = 20.0
        programs, log_probability = model._sample_programs()
        self.assertEqual(programs.shape, (MODULE.SAMPLES, MODULE.SLOTS))
        self.assertEqual(log_probability.shape, (MODULE.SAMPLES,))
        self.assertTrue(
            torch.equal(
                programs,
                torch.tensor(expected)[None].expand(MODULE.SAMPLES, -1),
            )
        )
        self.assertEqual(model._map_program().tolist(), expected)

    def test_endpoint_loss_reaches_both_writer_parameter_tensors(self) -> None:
        model = MODULE.build_model(ModelSpec(17, 13, 500_000_000))
        model.train()
        inputs, labels, valid = _teaching_batch()
        logits, auxiliary = model(inputs[:32], inputs[:32] != MODULE.PAD)
        loss = MODULE.token_training_loss(
            TokenLossBatch(logits[:, :2], labels[:32], valid[:32], None, auxiliary)
        )
        loss.backward()
        for parameter in (model.start_logits, model.transition_logits):
            self.assertIsNotNone(parameter.grad)
            assert parameter.grad is not None
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(int((parameter.grad != 0).sum()), 0)

    def test_sampled_endpoint_training_discovers_ood_square_program(self) -> None:
        model = MODULE.build_model(ModelSpec(17, 13, 500_000_000))
        model.train()
        inputs, labels, valid = _teaching_batch()
        bundle = MODULE.build_optimizer(model, OptimizerSpec(1000.0, "cpu"))
        optimizer = bundle.optimizer
        for _ in range(12):
            optimizer.zero_grad()
            logits, auxiliary = model(inputs, inputs != MODULE.PAD)
            loss = MODULE.token_training_loss(
                TokenLossBatch(logits[:, :2], labels, valid, None, auxiliary)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        moduli = torch.tensor([101, 103, 107, 109, 113, 127])
        values = torch.tensor([17, 29, 43, 61, 79, 97])
        depths = torch.tensor([1, 2, 4, 8, 16, 32])
        expected = values.clone()
        for step in range(int(depths.max())):
            updated = expected.square().remainder(moduli)
            expected = torch.where(depths > step, updated, expected)
        with torch.no_grad():
            actual = model._execute_map(values, moduli, depths)
        self.assertTrue(
            torch.equal(actual, expected),
            msg=f"learned program was {model._map_program().tolist()}",
        )

    def test_optimizer_owns_every_writer_parameter_once(self) -> None:
        model = MODULE.build_model(ModelSpec(17, 13, 500_000_000))
        bundle = MODULE.build_optimizer(model, OptimizerSpec(60.0, "cpu"))
        expected = [id(parameter) for parameter in model.parameters()]
        actual = [
            id(parameter)
            for group in bundle.optimizer.param_groups
            for parameter in group["params"]
        ]
        self.assertCountEqual(actual, expected)
        self.assertEqual(len(actual), len(set(actual)))


if __name__ == "__main__":
    unittest.main()
