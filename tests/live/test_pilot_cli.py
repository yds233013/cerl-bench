"""The complete CLI path, exercised offline through the synthetic transport.

`cerl pilot --dry-run` -> `--execute --synthetic` -> `cerl regenerate` ->
`cerl verify-manifest`, end to end, with no network and nothing spent.
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from cerl.cli import app

runner = CliRunner()


@pytest.fixture(scope="module")
def executed(tmp_path_factory):
    out = tmp_path_factory.mktemp("pilot")
    result = runner.invoke(
        app,
        [
            "pilot", "--execute", "--synthetic",
            "--out", str(out / "run.json"),
            "--transcripts-out", str(out / "transcripts.json"),
            "--ledger-out", str(out / "ledger.json"),
        ],
    )
    assert result.exit_code == 0, result.output
    return out, result.output


def test_dry_run_is_the_default_and_spends_nothing():
    result = runner.invoke(app, ["pilot"])
    assert result.exit_code == 0, result.output
    assert "DRY RUN" in result.output
    assert "nothing was sent and nothing was spent" in result.output


def test_the_dry_run_reports_the_split_audit():
    result = runner.invoke(app, ["pilot"])
    assert "split audit: clean" in result.output
    assert "partition        : train" in result.output
    assert "eligibility      : training-eligible only" in result.output
    assert "canonical split 1.2.0" in result.output
    assert "branches covered : 5" in result.output
    # The five branches the training pool cannot supply are named, not hidden.
    assert "COVERAGE LIMIT" in result.output
    assert "escalate_fraud" in result.output


def test_the_dry_run_labels_its_figures_as_projections():
    result = runner.invoke(app, ["pilot"])
    assert "estimate, not a live measurement" in result.output
    assert "projection, no live call" in result.output


def test_live_execution_is_refused_without_authorisation(monkeypatch):
    monkeypatch.delenv("CERL_LIVE_EVAL_AUTHORIZED", raising=False)
    monkeypatch.delenv("CERL_LIVE_EVAL_BUDGET_CENTS", raising=False)
    result = runner.invoke(app, ["pilot", "--execute"])
    assert result.exit_code != 0
    assert "Credentials alone are not a budget" in result.output


def test_a_coverage_limit_is_reported_rather_than_worked_around():
    result = runner.invoke(app, ["pilot", "--partition", "evaluation"])
    assert result.exit_code == 0
    assert "COVERAGE LIMIT" in result.output
    assert "not backfilled" in result.output


def test_synthetic_execution_completes_every_episode(executed):
    _, output = executed
    assert "completed        : 9/9 episodes" in output
    assert "source           : synthetic" in output


def test_synthetic_results_are_labelled_loudly(executed):
    _, output = executed
    assert "These are SYNTHETIC results" in output
    assert "never be reported as such" in output


def test_the_manifest_records_synthetic_provenance(executed):
    out, _ = executed
    manifest = json.loads((out / "run.json").read_text())
    assert manifest["verification_mode"] == "synthetic"
    assert manifest["agent"]["provider"] == "synthetic-transport"
    assert manifest["agent"]["privilege_mode"] == "unprivileged"
    assert manifest["agent"]["kind"] == "model"


def test_the_manifest_carries_the_spend_account(executed):
    out, _ = executed
    manifest = json.loads((out / "run.json").read_text())
    spend = manifest["metrics"]["spend"]
    for key in ("confirmed_cents", "reserved_cents", "unresolved_cents", "cap_cents"):
        assert key in spend, key
    assert spend["reserved_cents"] == 0.0


def test_the_ledger_is_written_for_resumption(executed):
    out, _ = executed
    ledger = json.loads((out / "ledger.json").read_text())
    assert ledger["model"] == "claude-opus-5"
    assert "confirmed_cents" in ledger
    assert "reserved_cents" not in ledger


def test_regenerate_replays_the_run_from_its_cache(executed):
    out, _ = executed
    result = runner.invoke(
        app,
        ["regenerate", str(out / "run.json"),
         "--transcripts", str(out / "transcripts.json")],
    )
    assert result.exit_code == 0, result.output
    assert "REGENERATED (offline, from cache)" in result.output


def test_regenerate_fails_on_a_gutted_cache(executed, tmp_path):
    out, _ = executed
    payload = json.loads((out / "transcripts.json").read_text())
    for transcript in payload["transcripts"]:
        transcript["entries"] = []
    broken = tmp_path / "broken.json"
    broken.write_text(json.dumps(payload))
    result = runner.invoke(
        app, ["regenerate", str(out / "run.json"), "--transcripts", str(broken)],
    )
    assert result.exit_code == 1
    assert "CACHE MISS" in result.output


def test_verify_manifest_replays_the_actions_offline(executed):
    out, _ = executed
    result = runner.invoke(app, ["verify-manifest", str(out / "run.json")])
    assert result.exit_code == 0, result.output
    assert "VERIFIED (offline, no model in the loop)" in result.output


# --------------------------------------------------------------------------
# the 8-episode development configuration and the leakage gate
# --------------------------------------------------------------------------


def test_the_five_episode_configuration_is_selectable():
    result = runner.invoke(app, ["pilot", "--per-branch", "1"])
    assert result.exit_code == 0, result.output
    assert "episodes         : 5" in result.output
    assert "branches covered : 5" in result.output
    assert "recommended cap  : $13.00" in result.output


def test_execution_is_refused_when_a_selection_would_leak(monkeypatch):
    """The gate still works; it just has nothing to catch on a clean split.

    Forced by pointing the audit at a partition that does carry holdouts, which
    is the situation the gate exists for.
    """
    from cerl.eval import pilot as pilot_module

    result = runner.invoke(
        app,
        ["pilot", "--execute", "--synthetic", "--partition", "validation",
         "--no-eligible-only"],
    )
    assert result.exit_code != 0, result.output
    assert "LEAKAGE" in result.output
    assert "refusing to execute" in result.output
    assert pilot_module.PILOT_ELIGIBLE_ONLY is True


def test_the_eligibility_filter_is_now_a_no_op_over_a_clean_split():
    """With the canonical split fixed, turning the filter off changes nothing.

    Under 1.1.0 this flag exposed 85 counterfactuals. That it is now inert is
    the clearest single demonstration that the split itself was repaired rather
    than merely filtered.
    """
    filtered = runner.invoke(app, ["pilot"])
    unfiltered = runner.invoke(app, ["pilot", "--no-eligible-only"])
    assert filtered.exit_code == unfiltered.exit_code == 0
    assert "LEAKAGE" not in unfiltered.output
    assert "episodes         : 9" in unfiltered.output


def test_a_ledger_from_a_different_cap_is_refused_with_an_instruction(tmp_path):
    """The guard is right; the traceback was not.

    Reusing a ledger across caps is how a run silently gets a bigger allowance,
    so the refusal stays. What changed is that the CLI names the two things a
    person might have meant instead of raising.
    """
    ledger = tmp_path / "ledger.json"
    first = runner.invoke(
        app,
        ["pilot", "--execute", "--synthetic", "--per-branch", "1",
         "--out", str(tmp_path / "a.json"),
         "--transcripts-out", str(tmp_path / "at.json"),
         "--ledger-out", str(ledger)],
    )
    assert first.exit_code == 0, first.output

    second = runner.invoke(
        app,
        ["pilot", "--execute", "--synthetic", "--per-branch", "2",
         "--out", str(tmp_path / "b.json"),
         "--transcripts-out", str(tmp_path / "bt.json"),
         "--ledger-out", str(ledger)],
    )
    assert second.exit_code != 0
    assert "different settings" in second.output
    assert "fresh --ledger-out" in second.output
    # The record of prior spend is intact, not discarded to get past the error.
    assert ledger.exists()
