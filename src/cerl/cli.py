"""``cerl`` command line: run, freeze, inspect."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from cerl.actions import ActionKind
from cerl.core import FrozenMap
from cerl.eval import manifest as manifest_module
from cerl.eval import runner as eval_runner
from cerl.eval import splits as split_module
from cerl.eval import verify_run
from cerl.reference import gold as gold_module
from cerl.reference import registry as reference_registry
from cerl.reference.runner import run_reference
from cerl.scenario import axes as ax
from cerl.scenario import freeze as freeze_module
from cerl.scenario.families import registry as family_registry
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
    family: Annotated[str, typer.Option(help="Family to freeze, or 'all'.")] = "all",
    out: Annotated[Path, typer.Option(help="Frozen scenario directory.")] = FROZEN,
    gold_out: Annotated[Path, typer.Option(help="Gold trajectory directory.")] = GOLD,
    manifest: Annotated[Path, typer.Option(help="Manifest path.")] = MANIFEST,
    limit: Annotated[int, typer.Option(help="Freeze at most N instances (testing).")] = 0,
) -> None:
    """Materialise, verify with the oracle, and commit every Phase 1A instance.

    The oracle pass is not optional: a scenario whose oracle does not score a
    clean 1.0 is a defective scenario and is refused rather than written.
    """
    known = family_registry.families()
    if family != "all" and family not in known:
        raise typer.BadParameter(f"unknown family {family!r}; known: {list(known)}")

    template_ids: list[str] = []
    for template_id in family_registry.template_ids():
        owner = family_registry.template_for(template_id).family
        if family in {"all", owner}:
            template_ids.append(template_id)

    entries = []
    written = 0
    generator_version = ""
    schema_version = 0
    work: list[tuple[str, Any, int]] = []
    for template_id in template_ids:
        instances = family_registry.plan_for(template_id)()
        if limit:
            instances = instances[:limit]
        work += [(template_id, assignment, seed) for assignment, seed in instances]

    for template_id, assignment, seed in work:
        scenario = freeze_module.materialize(template_id, assignment, seed)
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
        generator_version = scenario.generator_version
        schema_version = scenario.schema_version
        written += 1

    manifest.write_text(
        json.dumps(
            {
                "generator_version": generator_version,
                "schema_version": schema_version,
                "families": sorted({e["family"] for e in entries}),
                "count": written,
                "scenarios": sorted(entries, key=lambda e: e["scenario_id"]),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    # A scenario id collision would silently shrink the corpus, so the count of
    # distinct files written must equal the number of instances planned.
    distinct_ids = {entry["scenario_id"] for entry in entries}
    if len(distinct_ids) != written:
        typer.secho(
            f"scenario id collision: {written} instances produced only "
            f"{len(distinct_ids)} distinct ids",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)

    by_family: dict[str, int] = {}
    for entry in entries:
        by_family[entry["family"]] = by_family.get(entry["family"], 0) + 1
    for name, count in sorted(by_family.items()):
        typer.echo(f"  {name}: {count}")
    typer.secho(f"froze {written} scenarios; oracle clean on all of them", fg=typer.colors.GREEN)


@app.command()
def run(
    scenario: Annotated[str, typer.Option(help="Frozen scenario id.")],
    agent: Annotated[str, typer.Option(help="Policy: 'oracle' or 'alternative'.")] = "oracle",
    show_trace: Annotated[bool, typer.Option("--show-trace")] = False,  # noqa: FBT002
) -> None:
    """Run a policy over one frozen scenario and print its verdict."""
    if agent not in {"oracle", "alternative"}:
        raise typer.BadParameter(
            "reference policies only: 'oracle' or 'alternative' (no evaluated agents yet)",
        )
    frozen = _load(scenario)
    policy = (
        reference_registry.oracle_for(frozen)
        if agent == "oracle"
        else reference_registry.alternative_for(frozen)
    )
    episode = run_reference(frozen, policy)
    verdict = episode.verdict

    typer.echo(f"scenario : {frozen.scenario_id}")
    typer.echo(f"family   : {frozen.family}  (policy: {agent})")
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


@app.command(name="eval")
def evaluate(
    partition: Annotated[
        str, typer.Option(help="train | validation | evaluation | all."),
    ] = "all",
    family: Annotated[str, typer.Option(help="Family to evaluate, or 'all'.")] = "all",
    agent: Annotated[
        str, typer.Option(help="Policy: 'oracle' or 'alternative'."),
    ] = "oracle",
    out: Annotated[Path, typer.Option(help="Where to write the run manifest.")] = Path(
        "runs/run.json",
    ),
    frozen_dir: Annotated[Path, typer.Option()] = FROZEN,
) -> None:
    """Evaluate a policy over a partition and write a verifiable run manifest.

    Phase 1B ships reference policies only. An evaluated model agent would be
    recorded with ``privilege_mode: unprivileged``; a reference policy is
    recorded as privileged, because it reads ground truth and its numbers are an
    upper bound rather than a benchmark result.
    """
    scenarios = [freeze_module.load(p) for p in sorted(frozen_dir.glob("*.json"))]
    if family != "all":
        scenarios = [s for s in scenarios if s.family == family]
    if partition != "all":
        try:
            wanted = split_module.Partition(partition)
        except ValueError as exc:
            raise typer.BadParameter(f"unknown partition {partition!r}") from exc
        scenarios = split_module.select(scenarios, wanted)
    if not scenarios:
        raise typer.BadParameter("no scenarios matched")

    manifest = eval_runner.run_reference_evaluation(scenarios, agent)
    manifest = manifest.model_copy(
        update={"partition": None if partition == "all" else partition},
    )
    path = manifest_module.write(manifest, out)

    metrics = dict(manifest.metrics)
    typer.echo(f"episodes            : {metrics['episodes']}")
    typer.echo(f"safe completion     : {metrics['safe_completion_rate']:.3f}")
    typer.echo(f"decision correct    : {metrics['decision_correct_rate']:.3f}")
    typer.echo(f"committed violations: {metrics['committed_violation_rate']:.3f}")
    typer.echo(f"attempted violations: {metrics['attempted_violation_rate']:.3f}")
    typer.echo(f"by family           : {metrics['episodes_by_family']}")
    typer.secho(f"wrote {path}", fg=typer.colors.GREEN)


@app.command(name="verify-manifest")
def verify_manifest_command(
    manifest_path: Annotated[Path, typer.Argument(help="Run manifest to verify.")],
    frozen_dir: Annotated[Path, typer.Option()] = FROZEN,
) -> None:
    """Replay a recorded run offline and check it against its own evidence.

    No model, no network. This verifies the environment and verifier, which is
    what "reproducible" means for the benchmark: given these actions, these are
    the scores. Reproducing a *model's* actions is a separate claim.
    """
    if not manifest_path.exists():
        raise typer.BadParameter(f"no manifest at {manifest_path}")
    manifest = manifest_module.load(manifest_path)
    report = verify_run.verify_manifest(manifest, frozen_dir)
    typer.echo(report.summary())
    if not report.ok:
        raise typer.Exit(code=1)
    typer.secho("VERIFIED (offline, no model in the loop)", fg=typer.colors.GREEN)


@app.command(name="splits")
def splits_command(
    frozen_dir: Annotated[Path, typer.Option()] = FROZEN,
) -> None:
    """Report the train/validation/evaluation partition sizes."""
    scenarios = [freeze_module.load(p) for p in sorted(frozen_dir.glob("*.json"))]
    summary = split_module.summarize(scenarios)
    typer.echo(f"split version: {summary.version}")
    for name, count in sorted(summary.counts.items()):
        typer.echo(f"  {name:12s} {count:4d}")
    typer.echo(f"  {'total':12s} {summary.total():4d}")
