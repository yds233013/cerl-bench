#!/usr/bin/env python
"""Re-derive the cost table in docs/live-pilot-proposal.md.

Offline and free. Every figure comes from serialising the exact request payloads
the harness builds, over the actual pilot selection, driven by the oracle.
"""

from __future__ import annotations

from pathlib import Path

from cerl.eval import pilot as pilot_module
from cerl.scenario import freeze as freeze_module

FROZEN = Path("scenarios/frozen")


def main() -> None:
    scenarios = [freeze_module.load(p) for p in sorted(FROZEN.glob("*.json"))]
    projection = pilot_module.project(scenarios)

    print(f"fixed prefix (system + tool schemas): {pilot_module._fixed_prefix_tokens():,} tokens")  # noqa: SLF001
    print(f"episodes: {len(projection.episodes)} across {len(projection.branches)} branches\n")

    steps = sorted(e.oracle_steps for e in projection.episodes)
    print(f"oracle steps/episode: min {steps[0]} median {steps[len(steps) // 2]} max {steps[-1]}")

    print(f"\n{'configuration':46s} {'expected':>10s} {'worst case':>12s}")
    print("-" * 70)
    for label, cached in (("no caching", False), ("prompt caching", True)):
        expected = projection.total_cents(worst_case=False, cached=cached)
        worst = projection.total_cents(worst_case=True, cached=cached)
        name = f"{len(projection.episodes)} episodes, {projection.max_tokens} tok, {label}"
        print(f"{name:46s} {'$%.2f' % (expected / 100):>10s} {'$%.2f' % (worst / 100):>12s}")

    cap = projection.recommended_cap_cents()
    print(f"\nrecommended cap: ${cap / 100:.2f} ({cap} cents)")
    print("Nothing was sent and nothing was spent.")


if __name__ == "__main__":
    main()
