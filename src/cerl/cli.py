"""``cerl`` command line: run, freeze, inspect."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from cerl.actions import ActionKind
from cerl.agents import budget as budget_module
from cerl.agents import model_client, synthetic_transport
from cerl.core import FrozenMap
from cerl.eval import demos as demo_module
from cerl.eval import manifest as manifest_module
from cerl.eval import pilot as pilot_module
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


@app.command(name="pilot")
def pilot_command(
    *,
    execute: Annotated[
        bool,
        typer.Option("--execute/--dry-run", help="Run the pilot. Default is dry run."),
    ] = False,
    synthetic: Annotated[
        bool,
        typer.Option(help="Use the synthetic transport: offline, free, labelled synthetic."),
    ] = False,
    cached: Annotated[
        bool, typer.Option(help="Assume prompt caching in the projection."),
    ] = True,
    partition: Annotated[
        str, typer.Option(help="Partition to draw from. Default: train."),
    ] = "train",
    per_branch: Annotated[
        int, typer.Option(help="Episodes per eligible outcome branch."),
    ] = pilot_module.EPISODES_PER_BRANCH,
    eligible_only: Annotated[
        bool,
        typer.Option(
            help="Restrict to training-eligible scenarios. Off is a leakage diagnostic.",
        ),
    ] = True,
    out: Annotated[Path, typer.Option(help="Run manifest path.")] = Path(
        "runs/pilot.json",
    ),
    transcripts_out: Annotated[Path, typer.Option(help="Transcript cache path.")] = Path(
        "runs/pilot_transcripts.json",
    ),
    ledger_out: Annotated[Path, typer.Option(help="Spend ledger path.")] = Path(
        "runs/pilot_ledger.json",
    ),
    frozen_dir: Annotated[Path, typer.Option()] = FROZEN,
) -> None:
    """Project or execute the live-evaluation pilot.

    ``--dry-run`` is the default and spends nothing: it selects the scenarios,
    audits them against the split manifest, and estimates the cost from the real
    request payloads. ``--execute`` runs the pilot, and requires either explicit
    spending authorisation or ``--synthetic``, which uses an offline transport
    whose results are labelled synthetic and can never be labelled live.
    """
    scenarios = [freeze_module.load(p) for p in sorted(frozen_dir.glob("*.json"))]
    if not scenarios:
        raise typer.BadParameter(f"no frozen scenarios in {frozen_dir}")
    try:
        wanted = split_module.Partition(partition)
    except ValueError as exc:
        raise typer.BadParameter(f"unknown partition {partition!r}") from exc

    projection = pilot_module.project(
        scenarios, wanted, per_branch, eligible_only=eligible_only,
    )
    report = projection.audit

    typer.echo(f"model            : {projection.model}")
    typer.echo(f"max_tokens       : {projection.max_tokens}")
    typer.echo(f"token counter    : {projection.counter_name} (estimate, not a live measurement)")
    typer.echo(f"partition        : {report.partition}")
    pool_label = "training-eligible only" if report.eligible_only else "ALL (diagnostic)"
    typer.echo(
        f"eligibility      : {pool_label}  "
        f"(canonical split {split_module.SPLIT_VERSION_CANONICAL})",
    )
    typer.echo(f"episodes         : {len(projection.episodes)}")
    typer.echo(f"branches covered : {len(report.branch_coverage)}")
    typer.echo(f"sibling groups   : {len(report.sibling_groups)}")

    if report.held_out_partitions_touched:
        typer.secho(
            f"SPLIT INTEGRITY: selection touches {report.held_out_partitions_touched}",
            fg=typer.colors.RED,
        )
    if report.held_out_selected:
        typer.secho(
            f"LEAKAGE: selection includes registered held-out scenarios "
            f"{report.held_out_selected}",
            fg=typer.colors.RED,
        )
    if report.uncoverable_branches:
        pool = (
            f"eligible {report.partition}" if report.eligible_only else report.partition
        )
        typer.secho(
            f"COVERAGE LIMIT: {pool} cannot supply {report.uncoverable_branches}. "
            f"Reported, not backfilled -- importing a held-out value to restore "
            f"coverage is what this check exists to prevent.",
            fg=typer.colors.YELLOW,
        )
    if report.clean:
        typer.secho(
            f"split audit: clean (no leakage; {len(report.branch_coverage)}/10 "
            f"branches eligible)",
            fg=typer.colors.GREEN,
        )

    estimated = projection.total_cents(worst_case=False, cached=cached)
    worst = projection.total_cents(worst_case=True, cached=cached)
    cap = projection.recommended_cap_cents()
    typer.echo(f"estimated cost   : ${estimated / 100:.2f}  (projection, no live call)")
    typer.echo(f"modelled worst   : ${worst / 100:.2f}  (every episode to budget_steps)")
    typer.echo(f"recommended cap  : ${cap / 100:.2f}  ({cap} cents)")

    if not execute:
        typer.secho("DRY RUN: nothing was sent and nothing was spent.", fg=typer.colors.GREEN)
        return

    if not report.clean:
        raise typer.BadParameter(
            "refusing to execute a selection with leakage; see the audit above",
        )
    chosen = list(
        pilot_module.select(scenarios, wanted, per_branch, eligible_only=eligible_only),
    )
    result = (
        _run_synthetic_pilot(chosen, cap, ledger_out)
        if synthetic
        else _run_live_pilot(chosen, ledger_out)
    )

    manifest_path = manifest_module.write(result.manifest, out)
    transcript_path = pilot_module.write_transcripts(result.transcripts, transcripts_out)
    typer.echo(f"completed        : {result.completed}/{len(chosen)} episodes")
    typer.echo(f"source           : {result.source}")
    typer.echo(f"spend            : {dict(result.spend)}")
    for entry in result.interrupted:
        typer.secho(f"INTERRUPTED {entry}", fg=typer.colors.YELLOW)
    for request_id in result.orphaned_requests:
        typer.secho(
            f"UNRESOLVED request {request_id}: recovered without an outcome, "
            f"charged against the cap and NOT replayed",
            fg=typer.colors.YELLOW,
        )
    typer.secho(f"wrote {manifest_path} and {transcript_path}", fg=typer.colors.GREEN)
    if result.source != "live":
        typer.secho(
            "These are SYNTHETIC results. They are not model output and must "
            "never be reported as such.",
            fg=typer.colors.YELLOW,
        )


def _open_ledger_or_explain(ledger_path: Path, cap_cents: float) -> Any:
    """Resume the ledger, turning a refused resume into a usable instruction.

    Refusing to resume under a changed cap is the guard working -- reusing a
    ledger across caps is how a run silently gets a bigger allowance. But the
    fix is a person's choice between two different intentions, so say which
    they are rather than raising a traceback.
    """
    try:
        return pilot_module.open_ledger(ledger_path, cap_cents)
    except ValueError as exc:
        raise typer.BadParameter(
            f"{exc}\n\nThis ledger belongs to a run with different settings. "
            f"Either resume that run with its original cap, or start a new run "
            f"by passing a fresh --ledger-out path. Deleting the ledger would "
            f"discard the record of what has already been spent.",
        ) from exc


def _run_synthetic_pilot(
    scenarios: list[FrozenScenario], cap_cents: int, ledger_path: Path,
) -> pilot_module.PilotResult:
    """Exercise the whole execution path offline, at zero cost.

    Deliberately symmetric with the live path, ledger persistence included: a
    rehearsal that skips a step is not a rehearsal of that step.
    """
    ledger = _open_ledger_or_explain(ledger_path, float(cap_cents))
    ledger.counter_name = budget_module.MockTokenCounter.name
    transport = synthetic_transport.SyntheticTransport(
        default=synthetic_transport.text_turn("no scripted turn for this step"),
    )
    client = pilot_module.build_client(
        ledger, transport=transport, counter=budget_module.MockTokenCounter(),
    )
    return pilot_module.execute(
        scenarios, client, ledger, source=transport.source, ledger_path=ledger_path,
    )


def _run_live_pilot(
    scenarios: list[FrozenScenario], ledger_path: Path,
) -> pilot_module.PilotResult:
    """Run against the provider. Requires authorisation and a configured budget."""
    authorized = model_client.live_evaluation_budget()
    if authorized <= 0:
        raise typer.BadParameter(
            "live execution requires CERL_LIVE_EVAL_AUTHORIZED=1 and a positive "
            "CERL_LIVE_EVAL_BUDGET_CENTS. Credentials alone are not a budget. "
            "Use --synthetic to exercise the path offline.",
        )
    # Resume rather than restart: a run that began from zero would grant itself
    # the whole cap again, turning "a $50 cap" into $50 per attempt. Recovery
    # reads the per-request journal, so spend inside an interrupted episode is
    # not lost, and an orphaned request is charged rather than replayed.
    ledger = _open_ledger_or_explain(ledger_path, float(authorized))
    client = pilot_module.build_client(ledger)
    return pilot_module.execute(
        scenarios, client, ledger, source="live", ledger_path=ledger_path,
    )


@app.command(name="regenerate")
def regenerate_command(
    manifest_path: Annotated[Path, typer.Argument(help="Run manifest to regenerate.")],
    transcripts: Annotated[
        Path, typer.Option(help="Transcript cache recorded with the run."),
    ] = Path("runs/pilot_transcripts.json"),
    frozen_dir: Annotated[Path, typer.Option()] = FROZEN,
) -> None:
    """Replay a recorded run from its transcript cache. Offline, no model.

    This is the model-reproducibility claim, and it is weaker than and separate
    from ``verify-manifest``: replaying actions proves the environment and
    verifier are deterministic, while replaying transcripts proves only that the
    same recorded turns produce the same actions. A cache miss fails rather than
    falling back to a live call.
    """
    if not manifest_path.exists():
        raise typer.BadParameter(f"no manifest at {manifest_path}")
    if not transcripts.exists():
        raise typer.BadParameter(f"no transcript cache at {transcripts}")
    scenarios = [freeze_module.load(p) for p in sorted(frozen_dir.glob("*.json"))]
    manifest = manifest_module.load(manifest_path)
    recorded = pilot_module.read_transcripts(transcripts)
    report = pilot_module.regenerate(scenarios, manifest, list(recorded))
    typer.echo(report.summary())
    if not report.ok:
        raise typer.Exit(code=1)
    typer.secho("REGENERATED (offline, from cache)", fg=typer.colors.GREEN)


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
    *,
    write: Annotated[
        bool, typer.Option(help="Rebuild and commit the canonical split manifest."),
    ] = False,
    show_moves: Annotated[
        bool, typer.Option(help="List the sibling groups 1.2.0 moved out of training."),
    ] = False,
) -> None:
    """Report the partition sizes, or rebuild the canonical split manifest."""
    if write or show_moves:
        scenarios = [freeze_module.load(p) for p in sorted(frozen_dir.glob("*.json"))]
        built = split_module.build_manifest(scenarios)
        problems = split_module.verify_manifest_totality(scenarios, built)
        for problem in problems:
            typer.secho(f"MANIFEST DEFECT {problem}", fg=typer.colors.RED)
        if problems:
            raise typer.Exit(code=1)
        if show_moves:
            typer.echo(
                f"{len(built.moved_groups)} of {len(built.groups)} sibling groups "
                f"moved from their 1.0.0 partition:",
            )
            for group in built.moved_groups:
                typer.echo(
                    f"  {group.pair_key[:8]}  {group.partition_v1_0_0.value} -> "
                    f"{group.partition.value}  ({len(group.members)} members)  "
                    f"{group.reason}",
                )
        if write:
            path = split_module.write_manifest(built)
            typer.secho(f"wrote {path}", fg=typer.colors.GREEN)
        return
    _report_splits(frozen_dir)


def _report_splits(frozen_dir: Path) -> None:
    """Report the train/validation/evaluation partition sizes."""
    scenarios = [freeze_module.load(p) for p in sorted(frozen_dir.glob("*.json"))]
    summary = split_module.summarize(scenarios)
    typer.echo(f"split version: {summary.version}")
    for name, count in sorted(summary.counts.items()):
        typer.echo(f"  {name:12s} {count:4d}")
    typer.echo(f"  {'total':12s} {summary.total():4d}")


@app.command()
def demo(
    which: Annotated[str, typer.Option(help="Demo key, or 'all'.")] = "all",
    frozen_dir: Annotated[Path, typer.Option()] = FROZEN,
) -> None:
    """Run small executable demonstrations of the five core behaviours."""
    scenarios = [freeze_module.load(p) for p in sorted(frozen_dir.glob("*.json"))]
    if not scenarios:
        raise typer.BadParameter(f"no frozen scenarios in {frozen_dir}")

    keys = [d.key for d in demo_module.DEMOS] if which == "all" else [which]
    for key in keys:
        info = demo_module.demo_for(key)
        scenario, episode = demo_module.run_demo(key, scenarios)
        verdict = episode.verdict

        typer.secho(f"\n=== {info.title} ===", fg=typer.colors.CYAN)
        typer.echo(f"look for : {info.what_to_look_for}")
        typer.echo(f"scenario : {scenario.scenario_id}")
        typer.echo(f"branch   : {scenario.branch} (required: {scenario.required_decision})")
        typer.echo(f"declared : {verdict.declared_outcome}")
        typer.echo(
            f"rubric   : {sum(verdict.rubric.values())}/{len(verdict.rubric)} passed",
        )
        typer.echo(f"committed violations : {len(verdict.violations)}"
                   + (f" {[str(v.cost_class) for v in verdict.violations]}"
                      if verdict.violations else ""))
        typer.echo(f"attempted violations : {len(verdict.attempted_violations)}"
                   + (f" {[str(v.cost_class) for v in verdict.attempted_violations]}"
                      if verdict.attempted_violations else ""))
        typer.echo(f"prohibited side effects: {len(verdict.prohibited_side_effects)}")
        typer.echo(f"class    : {verdict.failure_class}")

        denied = [e for e in episode.trace.entries if e.denied_interlock]
        for entry in denied:
            typer.echo(f"  refused by backend: {entry.action_kind} -> {entry.denied_interlock}")
        failed = [e for e in episode.trace.entries if str(e.outcome) == "failed"]
        for entry in failed:
            typer.echo(f"  tool failure: {entry.action_kind} -> {entry.result.message[:60]}")
