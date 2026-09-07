# v0.1 candidate `fc2a301` — independent review repairs

An independent review of candidate `fc2a301` reproduced five defects. This
document records what each one was, what it now does, and how that is enforced.
Every finding was **reproduced first** against `fc2a301`'s own source in a git
worktree, so the before-evidence is observed behaviour rather than a reading of
the code. Both sides are recorded as data in
`evidence/rc-review/before_after.json`, produced by running the same probes
against each tree.

The regressions live in `tests/review/`. Run against `fc2a301`'s source they
fail 25 of 34; the nine that pass are checks of fields the old verifier already
compared. That number is the point of the file: a fix demonstrated only by the
old suite passing is not demonstrated at all.

---

## 1. Manifest verification compared a hand-picked subset of fields

**Was.** `verify_manifest()` rebuilt each episode by replay and then compared a
chosen list of fields. Anything not on the list was accepted as recorded.
Editing `task_completion` to `1.0` on all five episodes of the committed
baseline and recomputing the aggregates so the file stayed self-consistent
produced `ok=True` and the line *"all evidence matches"*. Rewriting a step's
`outcome` to `invented-outcome`, its `state_hash_after` to 64 `f`s and its
`committed_classes` to `["C_DISCLOSE"]` also verified.

**Now.** The record is rebuilt in full from the replay and compared field by
field over its serialised form — every episode field except `steps`, then every
step compared element-wise per key. The aggregate metrics are recomputed from
the **replayed** records, so making a tampered file internally consistent no
longer helps. Both probes now report `ok=False` naming the differing field.

`UNVERIFIABLE_EPISODE_FIELDS` is empty and stays empty: every field of an
`EpisodeRecord` is derived from the scenario and the recorded actions.

**What replay still cannot establish**, now printed rather than implied — who
produced a run, which model did (or whether one did at all), and what it cost.
A manifest checksum is verified separately and reported separately; it detects
corruption and is not evidence of honesty. A manifest recorded before manifests
were checksummed is reported as `absent`, not failed.

An empty manifest now says so instead of printing a bare `VERIFIED`: two
interrupted runs in `evidence/` record zero episodes, and passing every check
there is to run is not the same as reproducing a result.

## 2. Nested payload and document aliasing

**Was.** `ToolResult.payload` is a `FrozenMap[str, Any]`; the outer level is
immutable, a nested `dict` or `list` inside it is not. The same object was
returned in the observation and sealed into the hash-chained trace. Editing a
nested value through the observation edited the sealed entry:
`verify_chain()` then raised `ChainBroken: entry 0 seal does not match its
contents`, through the ordinary public API with no privileged access.
Separately, `WorldState.as_document()` handed out its own memoised cache, so a
caller that annotated a returned document changed every later read while the
typed state and the memoised hash kept the original.

**Now.** The trace keeps its own copy of the payload, and `as_document()`
returns a copy. Mutating an observation leaves the trace intact and the chain
verifying; a second `as_document()` read returns the original value and the
state hash is unchanged.

**Why a copy and not a deep freeze.** Freezing in place was tried first. A
nested `FrozenMap` in a field typed `Any` has no pydantic serialiser, so it
raises `PydanticSerializationError` — and had it not, it would have changed the
canonical encoding and with it every committed replay hash. A copy keeps the
payload plain JSON, so the encoding and all historical hashes are byte-identical.

**Cost, and what was done about it.** Copying the whole world document on every
per-step diff took an episode from ~30 ms to **77 ms**, past the 50 ms budget in
`docs/design.md`. The budget was not relaxed. Two changes brought it to
**33.2 ms**: `state_hash()` and the differ read the cache through
`document_for_reading()`, typed `Mapping` so mypy rejects a write at the call
site and justified where it is called — both are strictly read-only, and the
worlds they read are already-superseded snapshots. And `json_copy()` replaces
`copy.deepcopy` for the copy that remains: a canonical document is pure JSON
with no cycles, so `deepcopy`'s memo table and reductor dispatch are all cost.

## 3. Session dispatch was not atomic

**Was.** `http.server` is threaded, and deduplication checked the submission
token, then dispatched, then recorded it. Two concurrent requests carrying the
**same** token both passed the check before either recorded, and the action ran
twice: two records in the session history, one entry in the trace, and a step
index the client's view disagreed with. Retrying a *terminal* action returned
409, because `done` was checked before the token — the client that never got the
response to the action that ended the episode was told it could not retry it.
Reusing a token with a *different* action silently returned the first record.

**Now.** Check, dispatch, record and view are one transaction under a per-session
`RLock`. A token is bound to the action that first used it, so reuse with a
different action is a 409 rather than a wrong answer. The token is resolved
**before** the finished check, so a terminal retry is idempotent while a
genuinely new action after the end is still refused. A reset bumps a
`generation`; a request may carry the generation it was composed against, and one
from before a reset is refused.

Regressions run over real HTTP against the production classes, because
sequential direct calls passed on the old code.

## 4. Schema-valid refund amounts could end an episode

**Was.** The public tool schema advertised `amount_cents` as an unrestricted
integer and the action model accepted it, but `Refund` requires a positive
amount. `amount_cents=0` and `-1` validated as actions and then raised
`ValidationError` inside `env.step`, leaving the trace empty — an internal
exception where a defined, recorded outcome belongs. Ticket status and comment
kind were free text on the same pattern, and an unknown comment kind was
silently rewritten to `note`.

**Now.** `amount_cents` carries `gt=0`; `status` and `comment_kind` are their
enums. `_property_schema()` carries `exclusiveMinimum`/`minimum`/`maximum`
through to the advertised JSON Schema, so a handler-only fix cannot leave the
public contract wrong. Invalid values are rejected at the parse boundary and the
dispatcher scores them `malformed`; positive refunds are unaffected.

The `TicketStatus` handler previously returned `denied(NOT_FOUND, "unknown
status")`. That was removed rather than kept: Layer C is a frozen list of things
a **real backend** would refuse (CLAUDE.md rule 3), and "that is not a status" is
a parse failure wearing an interlock's clothes — it put an event in the attempted
series that no backend would ever produce.

`TicketStatus`, `CommentKind` and `RefundReason` moved to `core/vocabulary.py`,
because `actions` sits below `state` in the layer stack and could not otherwise
share them. `state` re-exports them, so nothing else moved.

**Version skew, stated.** `TOOL_SCHEMA_VERSION` is **1.1.0 → 1.2.0**. Replay
reads actions, not schemas, so every committed manifest still verifies — all 80
recorded actions in `evidence/local-baseline/` re-validate unchanged under the
tightened models. But the schema change moves every cached request key, so
*regenerating* a 1.1.0 run from the transcript cache reports schema skew rather
than a cache miss. Historical evidence recording `1.1.0` is left as recorded.

## 5. The reviewer was unreachable from the packaged application

**Was.** The built frontend requests a relative `/review/episodes`. Opened from
the documented `http://127.0.0.1:8000/?review`, that request went to the
operational server, which deliberately 404s those routes. The reviewer server
served the API but not the page. Cross-port routing existed only in Vite's
development proxy, absent from the built app — so following the README with the
reviewer running did not make the reviewer page work.

**Now.** `cerl serve-review` serves the built frontend from `app/dist`
(`--ui` to relocate, `--api-only` to opt out) so the page and its API are
same-origin on `:8001`, and the banner prints the URL. Verified with Vite
confirmed not running: reviewer `/` and `/?review` return 200,
`/review/episodes` returns 200 with six episodes, and `local-baseline:0` shows
`UNAUTHORIZED_ACTION` / `['C_AUTH']`. Operational `/review/*` still returns 404 —
routing was **not** fixed by exposing privileged data on the operational API.

---

## Coverage gate

The documented requirement is ≥90% **per package** on `core/`, `diff/`, `trace/`
and `verify/` (`docs/design.md`). CI enforced it with a single combined
`--cov-fail-under=90` across all four, which an average can satisfy while a
member is below — `diff/` was at 89.16%.

The gate is now run per package. `diff/` reached **100%** through tests for the
behaviour that was actually untested, not filler: pointer escaping round-trips
(unescaping `~1` before `~0` would retarget an op), a pattern not matching a
shorter pointer, an unresolvable `$.variable` failing closed, `get()` returning
`None` rather than raising so a miss cannot abort grading mid-episode, and
money compared in cents against a scenario variable. Every one of those decides
whether a diff op is permitted or becomes a residual.

## Documentation corrections

- `CLAUDE.md` rule 11 described Phase 1A scope. It now states v0.1 scope, what
  remains out of scope, and that historical evidence is immutable. Rule 1 records
  the one enumerated wall-clock exemption, `eval/latency.py`.
- The README said W1 and W3 are exercised by the grader study. They are exercised
  by the test suite; **the grader study is W2-only**.
- `CITATION.cff` was referenced but absent. Added, with no `repository-code`
  field, because this candidate is not published.
- The 114 restored-change cases were described as written by hand. They are **one
  programmed transformation applied across 114 scenarios**, which is why the
  study's own correlation limitation matters.
- `violate_then_revert()` is narrower than "a demonstrated reward exploit". It
  appends to the end of a correct run and so leaves the ticket at its *initial*
  status, not the required one; the task fails on its own terms
  (`task_ok=False`). What it shows is a prohibited change **absent from the
  terminal state**. Docstrings and the README and results wording now say that.
- `docs/workspace.md` states explicitly that the operational demo exposes
  descriptive scenario ids, demo titles and walkthroughs, and that **no number
  produced by driving that UI is a benchmark result**. Omitted rubric keys are not
  evidence of no label leakage.
