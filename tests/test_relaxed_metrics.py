import unittest

import torch

from benchmark.runner import (
    _accumulate_relaxed_metrics,
    _finalize_relaxed_metrics,
)


class RelaxedMetricTests(unittest.TestCase):
    def test_alignment_metrics_distinguish_partial_answers(self) -> None:
        predictions = torch.tensor(
            [[1, 2, 9, 8], [0, 5, 8, 8], [9, 8, 8, 8]]
        )
        targets = torch.tensor(
            [[1, 2, 3, -100], [4, 5, -100, -100], [6, -100, -100, -100]]
        )
        valid = targets != -100
        totals: dict[str, float] = {}

        _accumulate_relaxed_metrics(
            totals, predictions, targets, valid
        )
        metrics = _finalize_relaxed_metrics(totals)

        self.assertAlmostEqual(metrics["token_accuracy"], 0.5)
        self.assertAlmostEqual(metrics["mean_row_token_accuracy"], 7 / 18)
        self.assertAlmostEqual(metrics["mean_wrong_tokens"], 1.0)
        self.assertAlmostEqual(metrics["within_one_token_accuracy"], 1.0)
        self.assertAlmostEqual(metrics["within_two_token_accuracy"], 1.0)
        self.assertAlmostEqual(metrics["first_token_accuracy"], 1 / 3)
        self.assertAlmostEqual(metrics["last_token_accuracy"], 1 / 3)
        self.assertAlmostEqual(metrics["mean_correct_prefix_fraction"], 2 / 9)
        self.assertAlmostEqual(metrics["mean_correct_suffix_fraction"], 1 / 6)

    def test_accumulation_is_batch_partition_invariant(self) -> None:
        predictions = torch.tensor([[1, 0], [2, 3]])
        targets = torch.tensor([[1, 4], [2, 3]])
        valid = torch.ones_like(targets, dtype=torch.bool)
        together: dict[str, float] = {}
        split: dict[str, float] = {}

        _accumulate_relaxed_metrics(together, predictions, targets, valid)
        for row in range(2):
            _accumulate_relaxed_metrics(
                split,
                predictions[row : row + 1],
                targets[row : row + 1],
                valid[row : row + 1],
            )

        self.assertEqual(
            _finalize_relaxed_metrics(together),
            _finalize_relaxed_metrics(split),
        )


if __name__ == "__main__":
    unittest.main()
