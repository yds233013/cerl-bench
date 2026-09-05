#!/usr/bin/env python
"""Re-derive the cost table in docs/live-pilot-proposal.md.

Offline and free. Every figure is an **estimate** produced from serialised
request payloads driven by the oracle -- not a measured live cost. No paid call
has ever been made from this repository.
"""

from __future__ import annotations

from pathlib import Path

from cerl.eval import pilot as pilot_module
from cerl.eval import splits
from cerl.scenario import freeze as freeze_module

FROZEN = Path("scenarios/frozen")


def main() -> None:
    scenarios = [freeze_module.load(p) for p in sorted(FROZEN.glob("*.json"))]

    print(f"fixed prefix (system + tool schemas): {pilot_module.fixed_prefix_tokens():,} tokens")
    print("token counter: heuristic (the anthropic SDK is an optional extra)")
    print("ALL FIGURES ARE ESTIMATES. No live call has been made.\n")

    header = (
        f"{'configuration':52s} {'episodes':>8s} {'estimated':>11s} "
        f"{'worst case':>12s} {'cap':>7s}"
    )
    print(header)
    print("-" * len(header))

    inventory = splits.inventory(scenarios)
    print(
        f"split {inventory.version}: {inventory.eligible} of {inventory.total} "
        f"scenarios are training-eligible "
        f"({sum(inventory.held_out_in_train.values())} train scenarios are "
        f"registered holdouts)\n",
    )

    for per_branch in (1, 2):
        projection = pilot_module.project(scenarios, splits.Partition.TRAIN, per_branch)
        for label, cached in (("no caching", False), ("prompt caching", True)):
            expected = projection.total_cents(worst_case=False, cached=cached)
            worst = projection.total_cents(worst_case=True, cached=cached)
            cap = int((worst // 100 + 1) * 100)
            name = f"{per_branch}/branch, {projection.max_tokens} tok, {label}"
            print(
                f"{name:52s} {len(projection.episodes):8d} "
                f"{f'${expected / 100:.2f}':>11s} {f'${worst / 100:.2f}':>12s} "
                f"{f'${cap / 100:.0f}':>7s}",
            )
        audit = projection.audit
        print(
            f"    -> {len(audit.branch_coverage)}/10 branches, partition "
            f"{audit.partition}, leakage-free={audit.clean}",
        )
        if audit.uncoverable_branches:
            print(
                f"       not reachable from the eligible pool: "
                f"{audit.uncoverable_branches}",
            )

    print("\nNothing was sent and nothing was spent.")


if __name__ == "__main__":
    main()
