"""Evaluation harness: splits, runner, manifest, metrics, offline verification."""

from cerl.eval.agent_harness import AgentRun, run_agent
from cerl.eval.manifest import AgentConfig, EpisodeRecord, RunManifest
from cerl.eval.metrics import aggregate, attempted_violation_rate, committed_violation_rate
from cerl.eval.runner import record_episode, run_recorded_actions, run_reference_evaluation
from cerl.eval.splits import Partition, partition_of, select, summarize
from cerl.eval.verify_run import VerificationReport, verify_manifest

__all__ = [
    "AgentConfig",
    "AgentRun",
    "EpisodeRecord",
    "Partition",
    "RunManifest",
    "VerificationReport",
    "aggregate",
    "attempted_violation_rate",
    "committed_violation_rate",
    "partition_of",
    "record_episode",
    "run_agent",
    "run_recorded_actions",
    "run_reference_evaluation",
    "select",
    "summarize",
    "verify_manifest",
]
