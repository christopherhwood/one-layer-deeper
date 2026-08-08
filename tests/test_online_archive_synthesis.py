import unittest

import torch

from benchmark import ModelSpec, OptimizerSpec
from submissions.generated_program_easy import online_growth_benchmark as BENCHMARK
from submissions.generated_program_easy.online_archive_synthesis import (
    EXTENDED_TARGETS,
    OnlineClosureArchive,
    synthesize_online,
)
from submissions.generated_program_easy.register_machine_synthesis import TRAIN_MODULI
from submissions.generated_program_easy.semantic_closure_synthesis import (
    build_archive,
    exhaustive_pairs,
)


class OnlineArchiveSynthesisTest(unittest.TestCase):
    def test_archive_really_starts_with_only_four_terminals(self) -> None:
        archive = OnlineClosureArchive(
            exhaustive_pairs(TRAIN_MODULI),
            EXTENDED_TARGETS["square_mod"],
            guided=True,
            parent_budget=32,
        )
        self.assertEqual(len(archive.entries), 4)
        self.assertEqual(
            [entry.node.operation for entry in archive.entries],
            ["x", "N", "0", "1"],
        )
        self.assertTrue(all(entry.node.left == -1 for entry in archive.entries))
        archive.grow_exhaustive()
        self.assertGreater(len(archive.entries), 4)
        self.assertGreater(archive.generated_candidates, 0)

    def test_cubic_polynomial_is_absent_through_four_operations(self) -> None:
        pairs, _, _, archive = build_archive(4)
        target = tuple(
            EXTENDED_TARGETS["cubic_polynomial_mod"](*pair) for pair in pairs
        )
        self.assertNotIn(target, archive)

    def test_quadratic_plus_two_is_absent_through_four_operations(self) -> None:
        pairs, _, _, archive = build_archive(4)
        target = tuple(
            EXTENDED_TARGETS["quadratic_plus_two_mod"](*pair) for pair in pairs
        )
        self.assertNotIn(target, archive)

    def test_guided_archive_finds_beyond_four_operation_target(self) -> None:
        result = synthesize_online(
            "cubic_polynomial_mod",
            guided=True,
            beam_width=192,
            partner_width=24,
            max_rounds=8,
        )
        self.assertTrue(result["found_exact"], result)
        self.assertGreater(result["selected_instructions"], 4, result)
        self.assertEqual(result["ood_exact"], 1.0, result)
        self.assertGreater(result["capacity_pruned"], 0)

    def test_matched_enumeration_budget_misses_the_bridge(self) -> None:
        guided = synthesize_online(
            "cubic_polynomial_mod", guided=True, beam_width=192, max_rounds=5
        )
        enumeration = synthesize_online(
            "cubic_polynomial_mod", guided=False, beam_width=192, max_rounds=5
        )
        self.assertEqual(
            guided["generated_candidates"], enumeration["generated_candidates"]
        )
        self.assertTrue(guided["found_exact"])
        self.assertFalse(enumeration["found_exact"])
        self.assertEqual(guided["ood_exact"], 1.0)

    def test_benchmark_population_starts_empty_and_grows_in_optimizer(self) -> None:
        model = BENCHMARK.build_model(ModelSpec(17, 13, 500_000_000))
        self.assertEqual(int(model.active_count), 4)
        self.assertTrue(torch.equal(model.length[:4], torch.zeros(4, dtype=torch.long)))
        self.assertTrue(torch.equal(model.output_register[:4], torch.arange(4)))
        optimizer = BENCHMARK.build_optimizer(
            model, OptimizerSpec(60.0, "cpu")
        ).optimizer
        # Three growth rounds fill the bounded population and force the third
        # round to overwrite/prune rows rather than merely append forever.
        for _ in range(3 * BENCHMARK.SCORE_PASSES_PER_GROWTH):
            model.program_logits.grad = torch.zeros_like(model.program_logits)
            model.novelty_probe.grad = torch.zeros_like(model.novelty_probe)
            model.fitness_probe.grad = torch.ones_like(model.fitness_probe)
            optimizer.step()
        self.assertGreater(int(model.active_count), 4)
        self.assertEqual(int(model.growth_generation), 3)
        self.assertGreater(int(model.generated_programs), 0)
        self.assertGreater(int(model.pruned_programs), 0)
        self.assertLessEqual(int(model.active_count), BENCHMARK.POPULATION)
        children = slice(4, int(model.active_count))
        # Some mutations append a side computation without replacing the
        # parent's return register.  This is what permits useful multi-branch
        # programs to survive endpoint-only selection.
        self.assertTrue(
            bool((model.output_register[children] < 4).any()),
            "growth produced no return-preserving branch mutations",
        )


if __name__ == "__main__":
    unittest.main()
