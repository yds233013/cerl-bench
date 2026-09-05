"""The three Phase 1A commands: run, freeze, inspect."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from cerl.cli import app
from tests.helpers import frozen_paths

runner = CliRunner()


def test_inspect_reports_coverage_and_asserts_cells():
    result = runner.invoke(app, ["inspect", "--assert-cells", "all"])
    assert result.exit_code == 0, result.output
    assert "all ten required cells present" in result.output
    assert "refund_now" in result.output
    assert "request_then_refund" in result.output


def test_inspect_fails_when_a_required_cell_is_missing(tmp_path):
    # A directory holding a single scenario cannot satisfy the coverage gate.
    single = tmp_path / "frozen"
    single.mkdir()
    (single / frozen_paths()[0].name).write_text(
        frozen_paths()[0].read_text(encoding="utf-8"), encoding="utf-8",
    )
    result = runner.invoke(
        app, ["inspect", "--assert-cells", "all", "--frozen-dir", str(single)],
    )
    assert result.exit_code == 1
    assert "missing or under-seeded" in result.output


def test_inspect_rejects_a_missing_directory(tmp_path):
    result = runner.invoke(app, ["inspect", "--frozen-dir", str(tmp_path / "nope")])
    assert result.exit_code != 0


def _w2_scenario_id() -> str:
    return next(p.stem for p in frozen_paths() if p.stem.startswith("dup_charge_threshold"))


def _w1_scenario_id() -> str:
    return next(p.stem for p in frozen_paths() if p.stem.startswith("dup_profile"))


def test_run_oracle_prints_a_clean_verdict():
    scenario_id = _w2_scenario_id()
    result = runner.invoke(app, ["run", "--scenario", scenario_id, "--agent", "oracle"])
    assert result.exit_code == 0, result.output
    assert "SUCCESS" in result.output
    assert "committed violations : 0" in result.output
    assert "attempted violations : 0" in result.output
    assert "undeclared responder effects: 0" in result.output


def test_run_show_trace_renders_agent_and_responder_entries():
    responder_scenario = next(
        p.stem for p in frozen_paths()
        if p.stem.startswith("dup_charge_threshold") and "appr-missing_obtainable" in p.stem
    )
    result = runner.invoke(
        app, ["run", "--scenario", responder_scenario, "--agent", "oracle", "--show-trace"],
    )
    assert result.exit_code == 0, result.output
    assert "[A]" in result.output
    assert "[R]" in result.output, "the responder transition must be visible in the trace"
    assert "r_manager_grants" in result.output


def test_run_rejects_an_unknown_scenario():
    result = runner.invoke(app, ["run", "--scenario", "does_not_exist"])
    assert result.exit_code != 0


def test_run_rejects_a_non_reference_agent():
    """No evaluated agent yet; asking for one must fail loudly."""
    scenario_id = _w2_scenario_id()
    result = runner.invoke(app, ["run", "--scenario", scenario_id, "--agent", "prompt_only"])
    assert result.exit_code != 0


def test_run_accepts_the_alternative_reference_policy():
    scenario_id = _w2_scenario_id()
    result = runner.invoke(
        app, ["run", "--scenario", scenario_id, "--agent", "alternative"],
    )
    assert result.exit_code == 0, result.output
    assert "policy: alternative" in result.output


def test_freeze_rejects_an_unknown_family(tmp_path):
    """Always with tmp output paths.

    A freeze test that writes to the real scenario directories can silently
    replace the committed corpus with a partial one -- which is exactly what
    happened once, and the manifest count caught it.
    """
    result = runner.invoke(
        app,
        [
            "freeze", "--family", "not_a_real_family",
            "--out", str(tmp_path / "f"), "--gold-out", str(tmp_path / "g"),
            "--manifest", str(tmp_path / "m.json"),
        ],
    )
    assert result.exit_code != 0
    assert not (tmp_path / "f").exists()


def test_freeze_one_family_only(tmp_path):
    out, gold, manifest = tmp_path / "f", tmp_path / "g", tmp_path / "m.json"
    result = runner.invoke(
        app,
        [
            "freeze", "--family", "duplicate_billing_profile",
            "--out", str(out), "--gold-out", str(gold), "--manifest", str(manifest),
            "--limit", "3",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["families"] == ["duplicate_billing_profile"]
    assert payload["count"] == 3


def test_freeze_writes_scenarios_gold_and_a_manifest(tmp_path):
    out = tmp_path / "frozen"
    gold = tmp_path / "gold"
    manifest = tmp_path / "manifest.json"
    result = runner.invoke(
        app,
        [
            "freeze", "--out", str(out), "--gold-out", str(gold),
            "--manifest", str(manifest), "--limit", "4",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "oracle clean on all of them" in result.output
    # --limit applies per family, so the total is a multiple of the family count.
    written = len(list(out.glob("*.json")))
    assert written == len(list(gold.glob("*.json")))
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["count"] == written
    assert len(payload["scenarios"]) == written
    assert written >= 4
    for entry in payload["scenarios"]:
        assert entry["sha256"]
        assert entry["generator_version"]
        assert (out / entry["file"]).exists()


def test_run_works_for_every_family():
    """Each family's vertical slice runs from the CLI."""
    for scenario_id in (_w2_scenario_id(), _w1_scenario_id()):
        result = runner.invoke(app, ["run", "--scenario", scenario_id, "--agent", "oracle"])
        assert result.exit_code == 0, result.output
        assert "SUCCESS" in result.output
        assert "committed violations : 0" in result.output
