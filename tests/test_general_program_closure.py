import unittest

import torch

from benchmark import ModelSpec, OptimizerSpec
from submissions.generated_program_easy import closure_archive_benchmark as BENCHMARK
from submissions.generated_program_easy.semantic_closure_synthesis import (
    build_archive,
    synthesize,
)


class GeneralProgramClosureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)
        _, _, _, archive = build_archive(4)
        cls.expressions = list(archive.values())

    def _index(self, source: str) -> int:
        return next(
            index
            for index, expression in enumerate(self.expressions)
            if str(expression) == source
        )

    def test_closure_generates_and_gradient_selects_every_target(self) -> None:
        result = synthesize()
        self.assertTrue(result["all_targets_discovered"])
        self.assertEqual(result["semantic_archive_size"], 49_437)
        for name, target in result["targets"].items():
            with self.subTest(target=name):
                self.assertEqual(target["train_exact"], 1.0)
                self.assertEqual(target["ood_exact"], 1.0)
                self.assertTrue(target["strict_gradient_winner"])
                self.assertEqual(target["gradient_selected_ood_exact"], 1.0)

    def test_archive_grows_from_terminals_by_operation_count(self) -> None:
        model = BENCHMARK.build_model(ModelSpec(17, 10, 500_000_000))
        self.assertEqual(int(model.active_count), 4)
        optimizer = BENCHMARK.build_optimizer(
            model, OptimizerSpec(60.0, "cpu")
        ).optimizer
        for expected in (21, 201, 2_907, 49_437):
            model.program_logits.grad = torch.zeros_like(model.program_logits)
            optimizer.step()
            self.assertEqual(int(model.active_count), expected)

    def test_same_forward_executes_affine_recurrence(self) -> None:
        model = BENCHMARK.build_model(ModelSpec(17, 10, 500_000_000))
        selected = self._index("MOD(ADD(1,ADD(ADD(x,x),x)),N)")
        model.active_count.copy_(model.level_boundaries[-1])
        with torch.no_grad():
            model.program_logits.fill_(-10.0)
            model.program_logits[selected] = 10.0
        model.eval()
        modulus, value, time_steps = 323, 140, 3
        prompt = [
            BENCHMARK.N_MARK,
            *[BENCHMARK.DIGIT_OFFSET + int(digit) for digit in str(modulus)],
            BENCHMARK.X_MARK,
            *[BENCHMARK.DIGIT_OFFSET + int(digit) for digit in str(value)],
            BENCHMARK.T_MARK,
            BENCHMARK.DIGIT_OFFSET + time_steps,
        ]
        input_ids = torch.tensor([prompt])
        with torch.no_grad():
            logits, _ = model(input_ids, torch.ones_like(input_ids).bool())
        expected = value
        for _ in range(time_steps):
            expected = (3 * expected + 1) % modulus
        target = torch.tensor(
            [BENCHMARK.DIGIT_OFFSET + int(digit) for digit in str(expected)]
        )
        self.assertTrue(torch.equal(logits.argmax(dim=-1)[0, -len(target) :], target))

    def test_extreme_width_square_is_exact_through_t64(self) -> None:
        model = BENCHMARK.build_model(ModelSpec(17, 23, 500_000_000))
        selected = self._index("MOD(MUL(x,x),N)")
        moduli = torch.tensor([66_994_189, 134_217_689, 536_870_909, 1_073_741_789])
        values = moduli - torch.tensor([12_345, 23_456, 34_567, 45_678])
        actual = values
        with torch.no_grad():
            for _ in range(64):
                actual = model._execute_selected(actual, moduli, selected)
        expected = torch.tensor(
            [pow(int(value), 1 << 64, int(modulus)) for value, modulus in zip(values, moduli)]
        )
        self.assertTrue(torch.equal(actual, expected))


if __name__ == "__main__":
    unittest.main()
