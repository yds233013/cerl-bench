"""``cerl`` command line: run, freeze, inspect."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from cerl.actions import ActionKind
from cerl.core import FrozenMap
from cerl.reference import W2Oracle
from cerl.reference import gold as gold_module
from cerl.reference.runner import run_reference
from cerl.scenario import axes as ax
from cerl.scenario import freeze as freeze_module
from cerl.scenario import plan
from cerl.scenario.families.w2_duplicate_charge import TEMPLATE_ID
from cerl.scenario.schema import FrozenScenario
from cerl.verify import verify

app = typer.Typer(add_completion=False, help="CERL-Bench: counterfactual enterprise RL.")

FROZEN = Path("scenarios/frozen")
GOLD = Path("scenarios/gold")
MANIFEST = Path("scenarios/manifest.json")


def _load(scenario_id: str) -> FrozenScenario:
    path = FROZEN / f"{scenario_id}.json"
    if not path.exists():
        raise typer.BadParameter(f"no frozen scenario {scenario_id!r} in {FROZEN}")
    return freeze_module.load(path)


@app.command()
def freeze(
    family: Annotated[str, typer.Option(help="Family to freeze.")] = "duplicate_charge_approval",
    out: Annotated[Path, typer.Option(help="Frozen scenario directory.")] = FROZEN,
    gold_out: Annotated[Path, typer.Option(help="Gold trajectory directory.")] = GOLD,
    manifest: Annotated[Path, typer.Option(help="Manifest path.")] = MANIFEST,
    limit: Annotated[int, typer.Option(help="Freeze at most N instances (testing).")] = 0,
) -> None:
    """Materialise, verify with the oracle, and commit every Phase 1A instance.

    The oracle pass is not optional: a scenario whose oracle does not score a
    clean 1.0 is a defective scenario and is refused rather than written.
    """
    if family != "duplicate_charge_approval":
        raise typer.BadParameter("Phase 1A ships W2 only")

    entries = []
    written = 0
    instances = plan.all_instances()
    if limit:
        instances = instances[:limit]
    for assignment, seed in instances:
        scenario = freeze_module.materialize(TEMPLATE_ID, assignment, seed)
        episode, gold = gold_module.produce(scenario)
        if not episode.verdict.is_clean_oracle_run:
            typer.secho(
                f"REFUSED {scenario.scenario_id}: oracle did not score a clean 1.0 "
                f"(rubric={dict(episode.verdict.rubric)})",
                fg=typer.colors.RED,
            )
            raise typer.Exit(code=1)
        scenario = scenario.model_copy(update={"oracle_tool_calls": gold.tool_calls})
        path = freeze_module.write(scenario, out)
        gold_module.write(gold, gold_out)
        entries.append(freeze_module.manifest_entry(scenario, path))
        written += 1

    manifest.write_text(
        json.dumps(
            {
                "generator_version": scenario.generator_version,
                "schema_version": scenario.schema_version,
                "count": written,
                "scenarios": sorted(entries, key=lambda e: e["scenario_id"]),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    typer.secho(f"froze {written} scenarios; oracle clean on all of them", fg=typer.colors.GREEN)


@app.command()
def run(
    scenario: Annotated[str, typer.Option(help="Frozen scenario id.")],
    agent: Annotated[str, typer.Option(help="Policy to run: 'oracle'.")] = "oracle",
    show_trace: Annotated[bool, typer.Option("--show-trace")] = False,  # noqa: FBT002
) -> None:
    """Run a policy over one frozen scenario and print its verdict."""
    if agent != "oracle":
        raise typer.BadParameter("Phase 1A ships the oracle only (no evaluated agents yet)")
    frozen = _load(scenario)
    episode = run_reference(frozen, W2Oracle())
    verdict = episode.verdict

    typer.echo(f"scenario : {frozen.scenario_id}")
    typer.echo(f"branch   : {frozen.branch}  (required decision: {frozen.required_decision})")
    typer.echo(f"declared : {verdict.declared_outcome}")
    typer.echo(
        f"rubric   : {sum(verdict.rubric.values())}/{len(verdict.rubric)} "
        f"-> {dict(verdict.rubric)}",
    )
    typer.echo(f"committed violations : {len(verdict.violations)}")
    typer.echo(f"attempted violations : {len(verdict.attempted_violations)}")
    side_effects = len(verdict.prohibited_side_effects)
    undeclared = len(verdict.undeclared_responder_effects)
    typer.echo(f"prohibited side effects     : {side_effects}")
    typer.echo(f"undeclared responder effects: {undeclared}")
    typer.echo(f"tool calls: {verdict.tool_calls} (oracle {verdict.oracle_tool_calls})")
    typer.echo(f"class    : {verdict.failure_class}")

    if show_trace:
        typer.echo("\ntrace:")
        for entry in episode.trace.entries:
            marker = "A" if str(entry.origin) == "agent" else "R"
            detail = entry.responder_rule or entry.result.message[:56]
            typer.echo(
                f"  {entry.idx:2d} [{marker}] t={int(entry.logical_time)} "
                f"{entry.action_kind:26s} {entry.outcome:10s} {detail}",
            )


@app.command()
def inspect(
    family: Annotated[str, typer.Option()] = "duplicate_charge_approval",
    assert_cells: Annotated[str, typer.Option("--assert-cells")] = "",
    frozen_dir: Annotated[Path, typer.Option(help="Frozen scenario directory.")] = FROZEN,
) -> None:
    """Report frozen coverage; optionally assert every required cell is present."""
    if not frozen_dir.exists():
        raise typer.BadParameter(f"{frozen_dir} does not exist; run `cerl freeze` first")
    scenarios = [freeze_module.load(p) for p in sorted(frozen_dir.glob("*.json"))]
    scenarios = [s for s in scenarios if s.family == family]

    by_branch: dict[str, int] = {}
    for scenario in scenarios:
        by_branch[scenario.branch] = by_branch.get(scenario.branch, 0) + 1

    typer.echo(f"frozen scenarios: {len(scenarios)}")
    for branch, count in sorted(by_branch.items()):
        typer.echo(f"  {branch:24s} {count}")

    required_seeds = 3
    present: dict[str, int] = {}
    for cell in ax.REQUIRED_CELLS:
        want = dict(cell.axes)
        present[cell.key] = sum(1 for s in scenarios if dict(s.axes) == want)

    typer.echo("\nrequired cells:")
    missing = []
    for cell in ax.REQUIRED_CELLS:
        count = present[cell.key]
        status = "ok" if count >= required_seeds else "MISSING"
        if count < required_seeds:
            missing.append(cell.key)
        typer.echo(f"  {status:8s} {cell.key:24s} seeds={count}  {cell.expected_branch}")

    if assert_cells:
        if missing:
            typer.secho(f"missing or under-seeded cells: {missing}", fg=typer.colors.RED)
            raise typer.Exit(code=1)
        typer.secho("all ten required cells present at >=3 seeds", fg=typer.colors.GREEN)


TOOL_KINDS = tuple(sorted(k.value for k in ActionKind))
EMPTY = FrozenMap[str, str]()
VERIFY = verify
