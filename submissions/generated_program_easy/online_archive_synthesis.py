"""On-the-fly, endpoint-guided growth of a bounded semantic program archive.

This is the research implementation for the step after exhaustive semantic
closure.  Unlike ``semantic_closure_synthesis``, it never constructs a complete
archive up front.  It starts with the four terminals, proposes compositions of
live nodes, executes them on endpoint examples, removes observational
duplicates, and retains a bounded mixture of promising and diverse nodes.

The archive is a DAG: a newly retained node may refer only to older retained
nodes.  Pruning removes nodes from the future parent pool, while referenced
ancestors remain frozen so every retained program stays executable.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from typing import Callable

from submissions.generated_program_easy.generality_probe import (
    _audit_pairs,
)
from submissions.generated_program_easy.register_machine_synthesis import (
    CAP,
    OOD_MODULI,
    TARGETS,
    TRAIN_MODULI,
)
from submissions.generated_program_easy.semantic_closure_synthesis import (
    COMMUTATIVE,
    OPERATIONS,
    apply,
    exhaustive_pairs,
)


Target = Callable[[int, int], int]


EXTENDED_TARGETS: dict[str, Target] = {
    name: function for name, (function, _) in TARGETS.items()
}
EXTENDED_TARGETS.update(
    {
        # A six-instruction DAG in this grammar:
        # x2=x*x; x3=x2*x; a=x3+x2; b=a+x; c=b+1; return c%N.
        "cubic_polynomial_mod": (
            lambda modulus, value: (value**3 + value**2 + value + 1) % modulus
        ),
        "quadratic_plus_two_mod": (
            lambda modulus, value: (value**2 + value + 2) % modulus
        ),
        # Another multi-branch expression rather than a power ladder.
        # q=(x*x)//N; m=x&(N-1); a=q+m; b=a+1; return b%N.
        "quotient_mask_mix_mod": (
            lambda modulus, value: (
                value**2 // modulus + (value & (modulus - 1)) + 1
            )
            % modulus
        ),
    }
)


@dataclass(frozen=True)
class Node:
    operation: str
    left: int = -1
    right: int = -1
    instructions: int = 0


@dataclass
class ArchiveEntry:
    node: Node
    behavior: tuple[int, ...]
    score: float = 0.0
    exact: float = 0.0
    modular_exact: float = 0.0


def _terminal_behavior(name: str, pairs: list[tuple[int, int]]) -> tuple[int, ...]:
    if name == "x":
        return tuple(value for _, value in pairs)
    if name == "N":
        return tuple(modulus for modulus, _ in pairs)
    return (int(name),) * len(pairs)


def _compose_behavior(
    operation: str,
    left: tuple[int, ...],
    right: tuple[int, ...],
) -> tuple[int, ...]:
    return tuple(apply(operation, a, b) for a, b in zip(left, right))


def _endpoint_score(
    behavior: tuple[int, ...],
    target: tuple[int, ...],
    pairs: list[tuple[int, int]],
) -> tuple[float, float, float]:
    count = max(1, len(target))
    exact = sum(a == b for a, b in zip(behavior, target)) / count
    modular_exact = sum(
        a % modulus == b for a, b, (modulus, _) in zip(behavior, target, pairs)
    ) / count
    circular = 0.0
    prefix = 0.0
    bit_match = 0.0
    for actual, expected, (modulus, _) in zip(behavior, target, pairs):
        reduced = actual % modulus
        distance = abs(reduced - expected)
        distance = min(distance, modulus - distance)
        circular += 1.0 - distance / max(1, modulus // 2)
        actual_text = str(actual)
        expected_text = str(expected)
        matched = 0
        if len(actual_text) == len(expected_text):
            for a, b in zip(actual_text, expected_text):
                if a != b:
                    break
                matched += 1
        prefix += matched / max(1, len(expected_text))
        width = max(1, actual.bit_length(), expected.bit_length())
        bit_match += 1.0 - (actual ^ expected).bit_count() / width
    # Exact endpoints decide completion.  Modular equivalence is deliberately
    # the strongest partial signal: it preserves an unreduced correct formula
    # long enough for a later MOD node to make it exact.
    score = (
        16.0 * exact
        + 6.0 * modular_exact
        + 2.0 * circular / count
        + prefix / count
        + bit_match / count
    )
    return score, exact, modular_exact


class OnlineArchive:
    """A bounded append-only DAG with a pruned future-parent frontier."""

    def __init__(
        self,
        pairs: list[tuple[int, int]],
        target: Target,
        *,
        beam_width: int = 192,
        partner_width: int = 24,
        seed: int = 0,
        guided: bool = True,
    ) -> None:
        self.pairs = pairs
        self.target_behavior = tuple(target(*pair) for pair in pairs)
        self.beam_width = beam_width
        self.partner_width = partner_width
        self.guided = guided
        self.rng = random.Random(91_000 + seed)
        self.entries: list[ArchiveEntry] = []
        self.behavior_to_index: dict[tuple[int, ...], int] = {}
        for name in ("x", "N", "0", "1"):
            self._append(
                Node(name), _terminal_behavior(name, pairs), allow_duplicate=False
            )
        self.frontier = list(range(4))
        self.live_parent_pool = list(range(4))
        self.generated_candidates = 0
        self.semantic_duplicates_pruned = 0
        self.capacity_pruned = 0
        self.rounds = 0

    def _append(
        self,
        node: Node,
        behavior: tuple[int, ...],
        *,
        allow_duplicate: bool,
    ) -> int | None:
        if not allow_duplicate and behavior in self.behavior_to_index:
            return None
        score, exact, modular_exact = _endpoint_score(
            behavior, self.target_behavior, self.pairs
        )
        index = len(self.entries)
        self.entries.append(ArchiveEntry(node, behavior, score, exact, modular_exact))
        self.behavior_to_index.setdefault(behavior, index)
        return index

    def _partner_pool(self) -> list[int]:
        terminals = list(range(4))
        candidates = [index for index in self.live_parent_pool if index >= 4]
        if not candidates:
            return terminals
        if self.guided:
            ordered = sorted(
                candidates,
                key=lambda index: self.entries[index].score,
                reverse=True,
            )
            strong_count = self.partner_width // 2
            strong = ordered[:strong_count]
            # Reserve slots for the best ADD/MUL/etc. intermediates.  A single
            # scalar endpoint score otherwise fills the pool with many variants
            # of the same operation and destroys compositional reachability.
            structural: list[int] = []
            per_operation = max(1, strong_count // len(OPERATIONS))
            for operation in OPERATIONS:
                structural.extend(
                    [index
                    for index in ordered
                    if self.entries[index].node.operation == operation
                    ][:per_operation]
                )
            structural = structural[:strong_count]
            chosen = list(dict.fromkeys(strong + structural))
            remainder = [index for index in ordered if index not in set(chosen)]
            diverse_count = max(0, self.partner_width - len(chosen))
            diverse = self.rng.sample(
                remainder, min(diverse_count, len(remainder))
            )
            return list(dict.fromkeys(terminals + chosen + diverse))
        # Matched-compute enumeration ablation: parent choice is independent of
        # endpoint scores and follows archive insertion order.
        return terminals + candidates[: self.partner_width]

    def grow(self) -> int | None:
        """Generate one operation layer, deduplicate, and prune its frontier."""
        self.rounds += 1
        partners = self._partner_pool()
        proposed: dict[tuple[int, ...], Node] = {}
        for left_index in self.frontier:
            left = self.entries[left_index]
            for right_index in partners:
                right = self.entries[right_index]
                for operation in OPERATIONS:
                    orientations = ((left_index, right_index),)
                    if operation not in COMMUTATIVE and left_index != right_index:
                        orientations += ((right_index, left_index),)
                    for a_index, b_index in orientations:
                        a = self.entries[a_index]
                        b = self.entries[b_index]
                        self.generated_candidates += 1
                        behavior = _compose_behavior(
                            operation, a.behavior, b.behavior
                        )
                        if behavior in self.behavior_to_index or behavior in proposed:
                            self.semantic_duplicates_pruned += 1
                            continue
                        proposed[behavior] = Node(
                            operation,
                            a_index,
                            b_index,
                            max(a.node.instructions, b.node.instructions) + 1,
                        )

        ranked: list[tuple[float, float, float, tuple[int, ...], Node]] = []
        perfect: tuple[tuple[int, ...], Node] | None = None
        for behavior, node in proposed.items():
            score, exact, modular_exact = _endpoint_score(
                behavior, self.target_behavior, self.pairs
            )
            ranked.append((score, exact, modular_exact, behavior, node))
            if exact == 1.0 and perfect is None:
                perfect = behavior, node

        if self.guided:
            ranked.sort(key=lambda row: (row[0], row[2], row[1]), reverse=True)
            exploitation = max(1, self.beam_width // 2)
            retained = ranked[:exploitation]
            # A structural shelf keeps the strongest candidates produced by
            # every primitive operation.  These are still endpoint-ranked; the
            # shelf only prevents one operation family from monopolizing the
            # finite archive.
            shelf_quota = max(1, self.beam_width // (4 * len(OPERATIONS)))
            retained_behaviors = {row[3] for row in retained}
            for operation in OPERATIONS:
                shelf = [
                    row
                    for row in ranked
                    if row[4].operation == operation
                    and row[3] not in retained_behaviors
                ][:shelf_quota]
                retained.extend(shelf)
                retained_behaviors.update(row[3] for row in shelf)
            retained = retained[: min(len(retained), 3 * self.beam_width // 4)]
            selected_behaviors = {row[3] for row in retained}
            tail = [row for row in ranked if row[3] not in selected_behaviors]
            explore_count = min(self.beam_width - len(retained), len(tail))
            if explore_count:
                retained.extend(self.rng.sample(tail, explore_count))
        else:
            retained = ranked[: self.beam_width]
        self.capacity_pruned += max(0, len(ranked) - len(retained))

        new_frontier: list[int] = []
        for _, _, _, behavior, node in retained:
            index = self._append(node, behavior, allow_duplicate=False)
            if index is not None:
                new_frontier.append(index)

        if perfect is not None:
            behavior, node = perfect
            existing = self.behavior_to_index.get(behavior)
            if existing is not None:
                return existing
            return self._append(node, behavior, allow_duplicate=False)

        self.frontier = new_frontier
        # Every retained node remains eligible for a bounded number of future
        # compositions.  The pool itself is pruned to keep growth subquadratic.
        combined = list(dict.fromkeys(self.live_parent_pool + new_frontier))
        if self.guided:
            combined.sort(
                key=lambda index: self.entries[index].score,
                reverse=True,
            )
            fixed = list(range(4))
            nonterminals = [index for index in combined if index >= 4]
            self.live_parent_pool = fixed + nonterminals[
                : max(self.beam_width, self.partner_width)
            ]
        else:
            self.live_parent_pool = combined[: 4 + self.beam_width]
        return None

    def describe(self, index: int) -> str:
        cache: dict[int, str] = {0: "x", 1: "N", 2: "0", 3: "1"}

        def visit(current: int) -> str:
            if current in cache:
                return cache[current]
            node = self.entries[current].node
            source = f"{node.operation}({visit(node.left)},{visit(node.right)})"
            cache[current] = source
            return source

        return visit(index)

    def instruction_count(self, index: int) -> int:
        reachable: set[int] = set()

        def visit(current: int) -> None:
            if current < 4 or current in reachable:
                return
            reachable.add(current)
            node = self.entries[current].node
            visit(node.left)
            visit(node.right)

        visit(index)
        return len(reachable)


class OnlineClosureArchive:
    """Grow exact semantic levels online, then prune at a capacity boundary.

    Levels through ``exhaustive_operations`` are genuinely constructed by
    successive calls to :meth:`grow_exhaustive`; no later syntax exists in the
    initial state.  Once exhaustive closure becomes too large, :meth:`grow_bounded`
    chooses a fixed number of parent nodes and gives both the guided and
    enumeration policies the identical one-operation proposal mechanism.
    """

    def __init__(
        self,
        pairs: list[tuple[int, int]],
        target: Target,
        *,
        guided: bool,
        parent_budget: int,
    ) -> None:
        self.pairs = pairs
        self.target_behavior = tuple(target(*pair) for pair in pairs)
        self.guided = guided
        self.parent_budget = parent_budget
        self.entries: list[ArchiveEntry] = []
        self.behavior_to_index: dict[tuple[int, ...], int] = {}
        self.levels: list[list[int]] = [[]]
        self.generated_candidates = 0
        self.semantic_duplicates_pruned = 0
        self.capacity_pruned = 0
        self.rounds = 0
        for name in ("x", "N", "0", "1"):
            behavior = _terminal_behavior(name, pairs)
            index = self._append(Node(name), behavior)
            self.levels[0].append(index)

    def _append(self, node: Node, behavior: tuple[int, ...]) -> int:
        score, exact, modular_exact = _endpoint_score(
            behavior, self.target_behavior, self.pairs
        )
        index = len(self.entries)
        self.entries.append(ArchiveEntry(node, behavior, score, exact, modular_exact))
        self.behavior_to_index[behavior] = index
        return index

    def grow_exhaustive(self) -> None:
        """Construct the next exact expression-tree-cost level."""
        size = len(self.levels)
        proposed: dict[tuple[int, ...], Node] = {}
        for left_size in range(size):
            right_size = size - 1 - left_size
            for left_index in self.levels[left_size]:
                left = self.entries[left_index]
                for right_index in self.levels[right_size]:
                    right = self.entries[right_index]
                    for operation in OPERATIONS:
                        if operation in COMMUTATIVE and left_index > right_index:
                            continue
                        self.generated_candidates += 1
                        behavior = _compose_behavior(
                            operation, left.behavior, right.behavior
                        )
                        if behavior in self.behavior_to_index or behavior in proposed:
                            self.semantic_duplicates_pruned += 1
                            continue
                        proposed[behavior] = Node(
                            operation,
                            left_index,
                            right_index,
                            left.node.instructions + right.node.instructions + 1,
                        )
        level: list[int] = []
        for behavior, node in proposed.items():
            level.append(self._append(node, behavior))
        self.levels.append(level)
        self.rounds += 1

    def grow_bounded(self) -> int | None:
        """Prune parents, then give each survivor the same terminal wraps."""
        highest = self.levels[-1]
        if self.guided:
            ordered = sorted(
                highest,
                key=lambda index: (
                    self.entries[index].score,
                    self.entries[index].modular_exact,
                    self.entries[index].exact,
                ),
                reverse=True,
            )
        else:
            ordered = list(highest)
        parents = ordered[: self.parent_budget]
        self.capacity_pruned += max(0, len(highest) - len(parents))
        proposed: dict[tuple[int, ...], Node] = {}
        perfect: tuple[tuple[int, ...], Node] | None = None
        for parent_index in parents:
            parent = self.entries[parent_index]
            for terminal_index in range(4):
                terminal = self.entries[terminal_index]
                for operation in OPERATIONS:
                    orientations = ((parent_index, terminal_index),)
                    if operation not in COMMUTATIVE and parent_index != terminal_index:
                        orientations += ((terminal_index, parent_index),)
                    for left_index, right_index in orientations:
                        left = self.entries[left_index]
                        right = self.entries[right_index]
                        self.generated_candidates += 1
                        behavior = _compose_behavior(
                            operation, left.behavior, right.behavior
                        )
                        if behavior in self.behavior_to_index or behavior in proposed:
                            self.semantic_duplicates_pruned += 1
                            continue
                        node = Node(
                            operation,
                            left_index,
                            right_index,
                            left.node.instructions + right.node.instructions + 1,
                        )
                        proposed[behavior] = node
                        _, exact, _ = _endpoint_score(
                            behavior, self.target_behavior, self.pairs
                        )
                        if exact == 1.0 and perfect is None:
                            perfect = behavior, node
        level: list[int] = []
        for behavior, node in proposed.items():
            level.append(self._append(node, behavior))
        self.levels.append(level)
        self.rounds += 1
        if perfect is None:
            return None
        return self.behavior_to_index[perfect[0]]

    def describe(self, index: int) -> str:
        cache = {0: "x", 1: "N", 2: "0", 3: "1"}

        def visit(current: int) -> str:
            if current in cache:
                return cache[current]
            node = self.entries[current].node
            result = f"{node.operation}({visit(node.left)},{visit(node.right)})"
            cache[current] = result
            return result

        return visit(index)


def _fixed_probe_pairs() -> list[tuple[int, int]]:
    pairs = _audit_pairs(TRAIN_MODULI)
    # Enough behavioral coordinates to avoid trivial small-N aliases while
    # keeping millions of candidate executions practical on CPU.
    count = min(64, len(pairs))
    return list(dict.fromkeys(
        pairs[index * (len(pairs) - 1) // max(1, count - 1)]
        for index in range(count)
    ))


def synthesize_online(
    target_name: str,
    *,
    guided: bool = True,
    seed: int = 0,
    beam_width: int = 192,
    partner_width: int = 24,
    max_rounds: int = 8,
) -> dict[str, object]:
    function = EXTENDED_TARGETS[target_name]
    del partner_width
    pairs = exhaustive_pairs(TRAIN_MODULI)
    archive = OnlineClosureArchive(
        pairs,
        function,
        guided=guided,
        parent_budget=beam_width,
    )
    initial_entries = len(archive.entries)
    selected = None
    exhaustive_operations = min(4, max_rounds)
    for _ in range(exhaustive_operations):
        archive.grow_exhaustive()
    for _ in range(exhaustive_operations, max_rounds):
        selected = archive.grow_bounded()
        if selected is not None:
            break

    ood_pairs = _audit_pairs(OOD_MODULI)
    if selected is None:
        best = max(
            range(len(archive.entries)),
            key=lambda index: archive.entries[index].score,
        )
    else:
        best = selected
    source = archive.describe(best)

    # Re-evaluate the discovered DAG directly on a disjoint, wider set.
    values: list[tuple[int, ...]] = []
    for index, entry in enumerate(archive.entries):
        node = entry.node
        if index < 4:
            values.append(_terminal_behavior(node.operation, ood_pairs))
        else:
            values.append(
                _compose_behavior(node.operation, values[node.left], values[node.right])
            )
    expected = tuple(function(*pair) for pair in ood_pairs)
    actual = values[best]
    return {
        "target": target_name,
        "policy": "guided_diverse" if guided else "pure_enumeration",
        "seed": seed,
        "initial_entries": initial_entries,
        "rounds": archive.rounds,
        "found_exact": selected is not None,
        "selected_instructions": archive.entries[best].node.instructions,
        "train_exact": sum(a == b for a, b in zip(archive.entries[best].behavior, archive.target_behavior)) / len(pairs),
        "ood_exact": sum(a == b for a, b in zip(actual, expected)) / len(expected),
        "archive_entries": len(archive.entries),
        "generated_candidates": archive.generated_candidates,
        "semantic_duplicates_pruned": archive.semantic_duplicates_pruned,
        "capacity_pruned": archive.capacity_pruned,
        "program": source,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=EXTENDED_TARGETS, default="cubic_polynomial_mod")
    parser.add_argument("--policy", choices=("guided", "enumeration", "both"), default="both")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--beam-width", type=int, default=192)
    parser.add_argument("--partner-width", type=int, default=24)
    parser.add_argument("--max-rounds", type=int, default=8)
    args = parser.parse_args()
    policies = (True, False) if args.policy == "both" else (args.policy == "guided",)
    results = [
        synthesize_online(
            args.target,
            guided=guided,
            seed=args.seed,
            beam_width=args.beam_width,
            partner_width=args.partner_width,
            max_rounds=args.max_rounds,
        )
        for guided in policies
    ]
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
