"""The frozen experiment protocol. Written before any inference was run.

Everything a result depends on is pinned here rather than passed on a command
line, because a pilot whose settings drifted between the before and the after
measurement produces a number that means nothing. The before/after evaluation
uses one function with one configuration, and the training loop is not allowed
to reach into it.
"""

from __future__ import annotations

import collections
import json
import pathlib
from typing import Any

from cerl.core import Frozen
from cerl.scenario import freeze
from cerl.scenario.schema import FrozenScenario

REPO = pathlib.Path(__file__).resolve().parents[2]
SPLIT_MANIFEST = REPO / "scenarios" / "v2" / "split_manifest.json"

#: Pinned exactly, including the revision. A model id alone is not a pin: the
#: same name can serve different weights over time, and a pilot that cannot say
#: which weights it trained is not reproducible.
MODEL_ID = "Qwen/Qwen3-0.6B"
MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"

FAMILY = "duplicate_charge_approval"

# -- decoding, identical for the before and the after measurement -----------
#: Greedy. A pilot this small cannot afford sampling noise on top of everything
#: else; with n=5 validation scenarios, temperature would make the comparison
#: unreadable.
EVAL_TEMPERATURE = 0.0
EVAL_SEED = 20260907
#: Rollouts must vary or GRPO has no signal, so sampling is on for training.
TRAIN_TEMPERATURE = 1.0

MAX_PROMPT_TOKENS = 3072
#: One action is a short JSON object. The earlier 4B run failed by spending its
#: budget on prose, so this is sized for the action and the model is told not to
#: think (see ``render_prompt``), rather than being given room to ramble.
MAX_NEW_TOKENS = 160
#: Environment steps per episode **as the pilot ran**. Kept at 12 so the
#: recorded results stay reproducible.
#:
#: .. warning::
#:
#:    **This value made the task partly infeasible and invalidates the pilot's
#:    training signal as a measure of learning.** It was chosen for compute
#:    budget, without checking that a known-correct trajectory fits. Replaying
#:    the canonical gold trajectories through this wrapper shows that only 9 of
#:    the 15 selected scenarios are solved *by the reference trajectory* within
#:    12 actions: 7/10 train and 2/5 validation.
#:
#:    What that establishes is that the **reference** solutions exceed this
#:    limit. It does **not** establish that no policy could solve the remaining
#:    six within 12 actions -- a shorter correct trajectory may exist, and the
#:    oracle is one correct policy rather than the shortest. No maximum
#:    achievable score is implied.
#:    See ``NEXT_MAX_ACTIONS`` and ``evidence/rl-pilot/feasibility.json``.
MAX_ACTIONS = 12

# -- proposed for the next run; nothing historical uses these ---------------
#: Gold trajectories for the 15 selected scenarios need 10-16 actions, and all
#: 15 succeed at a uniform limit of 16.
#:
#: **16 is not a proven minimum.** It is the length of the reference solution,
#: and a shorter correct trajectory may well exist -- the oracle is one correct
#: policy, not the shortest one. 24 is 16 plus 50% headroom, so a policy that
#: takes a couple of redundant reads, or recovers from a malformed turn, is
#: still able to finish.
#:
#: An earlier version of this comment said waste is discouraged by "the
#: efficiency term in the reward". That was wrong: ``default_scalar`` weights
#: outcome, task, decision and committed cost, and does **not** weight
#: ``r_efficiency`` at all. Nothing in this objective penalises a redundant
#: action. The comment is corrected rather than the reward changed -- altering
#: the objective to match a stale comment would be the wrong repair.
NEXT_MAX_ACTIONS = 24

#: The longest gold trajectory reaches a 3,639-token prompt, so the 3,072 cap
#: below would have ended 5 of the 15 episodes early even at a raised action
#: limit -- it bit at step 11-12, before the action limit did. 8,192 clears the
#: observed maximum with room for the longer episodes a 24-action limit allows,
#: and is far inside the model's 40,960-token context.
NEXT_MAX_PROMPT_TOKENS = 8192

#: The longest reference action encodes to 99 tokens, so 160 already fits. Kept
#: unchanged: raising it would mostly buy room for prose, which is what the
#: earlier 4B run drowned in.
NEXT_MAX_NEW_TOKENS = 160

#: The pilot's training objective. It is ``cerl``'s existing default scalar,
#: used unchanged. It is NOT the benchmark metric, and no result here is
#: reported as one.
REWARD_NAME = "cerl.env.reward.default_scalar"


class Selection(Frozen):
    """A recorded, frozen list of scenario ids and why they were chosen."""

    partition: str
    rule: str
    scenario_ids: tuple[str, ...]

    def load(self) -> list[FrozenScenario]:
        return [freeze.load(freeze.FROZEN_DIR / f"{i}.json") for i in self.scenario_ids]


def _partitions() -> dict[str, str]:
    data = json.loads(SPLIT_MANIFEST.read_text())
    return {m: g["partition"] for g in data["groups"] for m in g["members"]}


def _w2() -> dict[str, FrozenScenario]:
    out = {}
    for path in sorted(freeze.FROZEN_DIR.glob("*.json")):
        scenario = freeze.load(path)
        if scenario.family == FAMILY:
            out[scenario.scenario_id] = scenario
    return out


def training_selection() -> Selection:
    """Every W2 scenario the canonical split assigns to ``train``. All ten.

    Read from the committed manifest, never hard-coded: a hard-coded list would
    keep working after the split changed, which is how held-out data quietly
    enters a training set.
    """
    part, w2 = _partitions(), _w2()
    ids = tuple(sorted(k for k in w2 if part[k] == "train"))
    return Selection(
        partition="train",
        rule="every W2 scenario in the canonical 1.2.0 train partition",
        scenario_ids=ids,
    )


def validation_selection() -> Selection:
    """Five fixed scenarios for the before/after measurement.

    Deterministic and stated in advance so it cannot be tuned after seeing a
    result: the lexicographically first scenario in each of the four branches
    the validation partition offers, plus the lexicographically second
    ``escalate_unapproved``. The fifth is an escalate case because with four
    branches the decision split would otherwise be act 3 / escalate 1, and
    over- and under-escalation is the failure mode this workflow exists to
    expose.

    The **evaluation** partition is never touched.
    """
    part, w2 = _partitions(), _w2()
    pool = sorted(k for k in w2 if part[k] == "validation")
    by_branch: dict[str, list[str]] = collections.defaultdict(list)
    for k in pool:
        by_branch[w2[k].branch].append(k)

    ids = [min(v) for _, v in sorted(by_branch.items())]
    ids.append(sorted(by_branch["escalate_unapproved"])[1])
    return Selection(
        partition="validation",
        rule=(
            "first scenario id per branch (lexicographic), plus the second "
            "escalate_unapproved to balance act against escalate"
        ),
        scenario_ids=tuple(ids),
    )


def manifest() -> dict[str, Any]:
    """Everything a reader needs to know what produced a number."""
    train, val = training_selection(), validation_selection()
    w2 = _w2()
    return {
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION},
        "reward": {"name": REWARD_NAME, "note": "pilot training objective, not a benchmark metric"},
        "decoding": {
            "eval_temperature": EVAL_TEMPERATURE,
            "train_temperature": TRAIN_TEMPERATURE,
            "seed": EVAL_SEED,
            "max_prompt_tokens": MAX_PROMPT_TOKENS,
            "max_new_tokens": MAX_NEW_TOKENS,
            "max_actions": MAX_ACTIONS,
        },
        "termination": (
            "the episode ends on a terminal action (finish/escalate/abstain), "
            f"or after {MAX_ACTIONS} environment steps, whichever comes first; "
            "a step-limited episode is recorded as such and kept in the results"
        ),
        "training_scenarios": {
            "partition": train.partition, "rule": train.rule,
            "count": len(train.scenario_ids),
            "branches": dict(collections.Counter(w2[i].branch for i in train.scenario_ids)),
            "ids": list(train.scenario_ids),
        },
        "validation_scenarios": {
            "partition": val.partition, "rule": val.rule,
            "count": len(val.scenario_ids),
            "branches": dict(collections.Counter(w2[i].branch for i in val.scenario_ids)),
            "ids": list(val.scenario_ids),
        },
    }
