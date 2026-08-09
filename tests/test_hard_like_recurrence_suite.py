from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from scripts.generate_hard_like_recurrence_suite import (
    DEPTH_LADDER,
    generate_suite,
)
from scripts.generate_hidden_recurrence_probe import TRANSITIONS


class HardLikeRecurrenceSuiteTest(unittest.TestCase):
    def test_transition_reference_values_are_stable(self) -> None:
        expected = {
            "affine": 5,
            "cube": 47,
            "cubic_polynomial": 30,
            "quadratic_plus_two": 19,
            "quotient_mask_mix": 5,
            "alternating_square_affine": 93,
            "conditional_threshold": 5,
            "source_residual": 18,
            "second_order_add": 32,
            "time_indexed": 85,
            "modulus_keyed": 84,
            "bitwise_shift_mix": 67,
        }
        self.assertEqual(set(TRANSITIONS), set(expected))
        self.assertEqual(
            {
                name: transition(5, 3, 11, 13)
                for name, transition in TRANSITIONS.items()
            },
            expected,
        )

    def test_stateful_families_are_not_one_state_semigroups(self) -> None:
        # Reapplying a one-step endpoint function resets hidden source/phase
        # state.  These inequalities certify that a unary tied accumulator is
        # structurally insufficient, independently of optimization quality.
        for family in (
            "alternating_square_affine",
            "source_residual",
            "second_order_add",
            "time_indexed",
        ):
            transition = TRANSITIONS[family]
            direct = transition(7, 2, 11, 13)
            once = transition(7, 1, 11, 13)
            reset_composition = transition(once, 1, 11, 13)
            self.assertNotEqual(
                direct,
                reset_composition,
                msg=f"{family} unexpectedly collapsed to a unary semigroup",
            )

    def test_suite_has_deep_endpoints_and_seen_and_unseen_n_profiles(self) -> None:
        families = [
            "alternating_square_affine",
            "second_order_add",
            "modulus_keyed",
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            suite = generate_suite(
                root,
                families,
                examples_per_setting=32,
                profile_examples=4,
            )
            self.assertEqual(suite["training_time_steps"], [2, 4, 8])
            self.assertEqual(suite["ood_time_steps"], [16])
            self.assertEqual(suite["depth_ladder"], list(DEPTH_LADDER))
            self.assertEqual(
                json.loads((root / "suite_config.json").read_text()), suite
            )

            for family in families:
                family_root = root / family
                config = json.loads(
                    (family_root / "dataset_config.json").read_text()
                )
                self.assertEqual(
                    config["generator_config"]["generator_family"],
                    f"hard_like_{family}",
                )
                self.assertGreater(config["split_counts"]["train"], 0)
                self.assertGreater(config["split_counts"]["test"], 0)
                self.assertGreater(config["split_counts"]["ood"], 0)
                for depth in DEPTH_LADDER:
                    seen = family_root / f"depth_t_{depth}.jsonl"
                    unseen = family_root / f"depth_ood_n_t_{depth}.jsonl"
                    self.assertTrue(seen.is_file())
                    self.assertTrue(unseen.is_file())
                    self.assertEqual(len(seen.read_text().splitlines()), 8)
                    self.assertEqual(len(unseen.read_text().splitlines()), 8)


if __name__ == "__main__":
    unittest.main()
