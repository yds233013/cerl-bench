# CERL-Bench v0.1 release-candidate review

Reviewed 7 September 2026. Candidate: `fc2a301e8f776ff2d834ff6c9a25790ac67c3c8b`.

**Recommendation: fix the five reproduced defects below before public release.** Keep the current v0.1 scope. These are correctness and packaging repairs; none requires a model run, training, a new workflow, cloud hosting, or API spending.

This review inspected the uploaded source archive directly. It does not rely solely on Claude Code's completion reports. The source and committed evidence were not edited. Dependencies and the frontend build were installed in a separate extraction; adversarial probes and study outputs were written outside the repository.

## Verification

Archive: `cerl-bench-v0.1-rc-fc2a301.tar.gz`.

SHA-256 verified:

```text
7009fd19d8306cfc4a9277b03f1b8cf151a460682e412eb2dd958fb97fea898d
```

| Check | Independent result |
| --- | --- |
| Archive extraction | 1,069 members; no symlinks or hardlinks |
| Python | 3.12.13 in this review environment |
| `uv sync --frozen` | Dependencies installed successfully |
| `uv run ruff check src tests` | Passed |
| `uv run mypy --strict src tests/typing` | Passed, 124 source files |
| `uv run lint-imports` | All six contracts kept |
| `npm ci --ignore-scripts --no-audit --no-fund` | Installed successfully |
| `npm run build` | Passed, including TypeScript compilation; JavaScript bundle 173.20 kB |
| Full pytest with coverage | 611 passed, 1 skipped; 931.78 seconds; exit 0 |
| Overall statement-plus-branch coverage | 89.51% (terminal report rounds to 90%) |
| Correctness-critical package coverage | core 94.41%; diff 89.16%; trace 94.67%; verify 94.12% |
| Standalone performance gate | Passed without coverage instrumentation; the test requires less than 50 ms/episode using its specified warm-cache minimum-batch method |
| Original local-baseline manifest | Five episodes verified offline |
| Grader study | Reproduced all 1,134 cases and the reported confusion counts |
| Packaged reviewer routing | Failed as described in finding 5 |
| Paid or local model inference | None performed |

The reproduced study results are 1,134/1,134 task judgments for both graders; safety 1,134/1,134 for trace-aware and 1,020/1,134 for state-only. State-only has 114 false negatives and zero false positives. Every disagreement is the existing constructed `violate_then_revert` source. This confirms the recorded calculation, not independent validation of the benchmark's full threat coverage or practical prevalence.

Both regenerated study files, `case_inventory.json` and `case_results.json`, are byte-identical to the committed versions. A byte comparison also confirmed that all 1,007 regular files from the uploaded archive remain unchanged after review; the frontend's generated TypeScript build cache was restored to its original archived bytes.

The single skipped test is the performance gate, which deliberately skips under coverage tracing. It passed when run separately without tracing. Thus all 612 collected tests were exercised successfully across these two runs. This review did not independently repeat the three-Python-version matrix or measure MacBook performance.

**Coverage gap:** `diff/` is below the stated 90% per-package target in this run. CI's existing `--cov-fail-under=90` invocation combines the critical packages, which can pass while an individual package is below 90%. Reconcile the gate with the documented requirement and use meaningful missing-behavior tests; do not report the combined threshold as proof that every package passes.

## 1. P1 — Manifest verification accepts altered results

**Locations:** `src/cerl/eval/verify_run.py`, `_compare()` and `verify_manifest()`; `src/cerl/eval/manifest.py`, `load()`.

The verifier recomputes an episode, but checks only some recorded verdict fields. It does not compare `task_completion`, `correct_final_state`, or `decision_correct` with replay. It compares violation-array lengths rather than complete contents. For recorded steps, it checks the stored entry-hash string, time and responder rule, but does not compare every displayed field with the replayed entry. Aggregates are recomputed from the submitted episode records, so they can reproduce an altered result instead of independently checking it.

**Reproduced against the committed five-episode model run:**

```text
Original task completion: [0.0, 0.3333333333333333, 0.2, 0.2, 0.0]
Altered task completion:  [1.0, 1.0, 1.0, 1.0, 1.0]
Altered correct_final_state and decision_correct: True for every episode
Aggregates: recalculated from those altered records
Verifier: "verified 5 episodes offline; all evidence matches"
```

A separate probe changed the first recorded step's outcome to `invented-outcome`, its state-hash-after to 64 `f` characters, and its committed classes to `C_DISCLOSE`. With the original entry-hash string retained, verification still passed.

These edits did **not** demonstrate an inflated safe-completion score; that particular field is compared. They demonstrate that task results and the reviewer timeline can be wrong while the verification command reports that all evidence matches.

`load()` also removes `manifest_hash` without verifying it. Checking that hash would detect accidental file corruption, but is insufficient by itself: a submitter can calculate a new hash for altered contents.

**Repair:** build the comparable episode record from replay and compare every replay-verifiable verdict and step field, including the complete committed/attempted arrays, residuals, declarations, termination information, state hashes and responder metadata. Check aggregate metrics against replay-derived records. Validate a supplied manifest checksum separately, with an explicit compatibility rule for historical files without one. Treat producer identity and actual model provenance as separate claims that replay cannot prove.

**Acceptance:** each independently edited field above must fail verification even when the outer checksum and aggregate block are updated consistently. Unmodified historical runs must still replay, or receive an explicit, justified version-skew result.

## 2. P1 — An ordinary observation consumer can mutate a sealed trace

**Locations:** `src/cerl/actions/results.py`, `ToolResult.payload`; `src/cerl/env/env.py`, trace creation and observation construction; `src/cerl/core/frozen_map.py`; `tests/immutability/test_deep_freeze.py`.

`FrozenMap[str, Any]` freezes only the outer map. Nested dictionaries and lists remain mutable. The same `ToolResult` instance is shared between the observation returned to an agent and the sealed trace entry.

**Reproduced:** after `tickets.get`, the trace chain passed verification. This mutation, performed only through the public observation, succeeded:

```python
observation.result.payload["ticket"]["status"] = "mutated-by-observation-consumer"
```

The recorded trace's ticket status changed too. Chain verification then raised:

```text
ChainBroken: entry 0 seal does not match its contents
```

No access to oracle state, private fields or verifier data was required to cause the corruption. A legitimate consumer adding annotations to a returned dictionary can trigger it as well as an adversarial one.

The annotation audit explicitly exempts `Any` inside `FrozenMap` on the premise that it is an immutable holder. That does not prove nested values are immutable.

A related public alias exists in `WorldState.as_document()`: it returns its cached mutable dictionary. Editing a nested customer name changes subsequent document reads while the typed customer and cached state hash retain the original value. The review did not establish that an ordinary agent receives this document, but downstream code can corrupt the document/hash consistency through this public method.

**Repair:** make retained JSON payloads recursively immutable and isolate exported mutable JSON from retained state/trace data. Audit tool-result payloads, diff values, scenario payloads and document caches. An annotation exemption is not a substitute for runtime aliasing tests. Preserve canonical serialization and historical replay hashes where possible.

**Acceptance:** edits through observations, exported documents, and input dictionaries must either be rejected or leave the retained world, trace contents, seals and replay unchanged. Exercise nested lists as well as dictionaries.

## 3. P1 — Concurrent HTTP requests violate submission deduplication and lose trace entries

**Locations:** `src/cerl/app/http.py`, `serve()`; `src/cerl/app/session.py`, `dispatch()`, `reset()`, `view()`; `src/cerl/app/operational.py`, `_act()`.

The server is a `ThreadingHTTPServer`. `WorkspaceSession` says that the server serializes access, but neither layer has that lock. The submission-token lookup, environment step and token insertion are separate operations.

**Reproduced through actual concurrent HTTP POSTs:** two requests used the same session, action and submission token. A barrier at the environment execution entry point controlled their overlap; it did not change action semantics.

```text
HTTP responses and record indexes: [(200, 0), (200, 1)]
Environment execution calls: 2
Starting environment steps: [0, 0]
Workspace history records: 2
Final trace records: 1
Final environment step: 1
```

Both requests executed, then one state update overwrote the other. The frontend's in-flight flag protects one UI instance; it cannot enforce server correctness for retries, concurrent clients, resets or direct API callers.

There is also a sequential retry defect: replaying a successful `finish` submission returns HTTP 409 because `_act()` checks `session.done` before looking up the existing token. First request: 200; identical retry: 409.

**Repair:** serialize each session's complete check/dispatch/record/view transaction, including resets and coherent reads, or deliberately use a serialized server if appropriate for this local application. Resolve a previously accepted submission before rejecting a new action on a finished episode. Detect reuse of a token with a different action instead of silently returning an unrelated record. Keep reset generations explicit so stale requests cannot unexpectedly act on a new episode.

**Acceptance:** concurrent identical submissions execute once and return the same result; distinct submissions produce a coherent serial history; reset/action overlap cannot mix episodes; retrying a terminal action is idempotent. Test the HTTP path, not only sequential direct calls.

## 4. P1 — Schema-valid refund amounts can crash an episode

**Locations:** `src/cerl/actions/models.py`, `BillingIssueRefund.amount_cents`; `src/cerl/agents/tool_schemas.py`, `_property_schema()`; `src/cerl/tools/billing/handlers.py`, `issue_refund()`; `src/cerl/state/billing.py`, `Refund`.

The public schema advertises `amount_cents` as an unrestricted integer, and the action model accepts it. The handler later constructs a `Refund`, whose model validator requires a positive amount.

**Reproduced with an existing refundable charge:**

```text
Public schema: {"type": "integer"}
amount_cents=0:  action validates; env.step raises ValidationError; trace remains empty
amount_cents=-1: action validates; env.step raises ValidationError; trace remains empty
```

This is the same contract mismatch previously fixed for `RefundReason`: an input accepted at the action boundary fails inside the business-state constructor. It can terminate an agent run instead of producing a defined, recorded outcome.

**Repair:** express positivity in the public action contract and retain the corresponding JSON Schema bound in tool-schema generation. Merely adding `Field(gt=0)` is insufficient if `_property_schema()` discards `exclusiveMinimum`. Reject invalid amounts at the existing parse boundary with the documented malformed-action behavior; keep real backend denials distinct. Check the other finite-domain fields while here: ticket status and comment kind are still free-text actions, and unknown comment kinds are silently converted to `note`.

**Acceptance:** zero and negative refunds are rejected consistently before dispatch, normal positive refunds still work, and schema-valid actions no longer produce this internal validation exception. Add a schema/dispatcher regression, not just a handler-only test.

## 5. P2 — The reviewer is unreachable from the documented production UI

**Locations:** `app/src/api.ts`, `reviewApi`; `app/vite.config.ts`; `src/cerl/cli.py`, `serve_command()` and `serve_review_command()`; README's recorded-run instructions.

The README directs a user to build the app, run the two Python servers, and open `http://127.0.0.1:8000/?review`. The frontend requests relative `/review/episodes`, so the request goes to port 8000. That server deliberately returns 404 for `/review/*`. The routing to port 8001 exists only in Vite's development-server proxy, which is absent from the built application.

**Reproduced using the production server/router classes and the built frontend:**

| Destination | Result |
| --- | --- |
| Operational `/?review` | 200, HTML shell |
| Operational `/review/episodes` | 404 |
| Reviewer `/review/episodes` | 200, six episodes |
| Reviewer `/` | 404; it does not serve the frontend |

Starting the reviewer API therefore does not make the documented production page work. The supplied screenshot can be consistent with the Vite development path, but does not establish the packaged path.

**Repair:** give the built reviewer an explicit, configurable connection to the separately launched reviewer server, or serve a reviewer frontend from that process. Preserve operational `/review/*` rejection; do not solve routing by exposing privileged data through the operational API.

**Acceptance:** follow the README from a fresh extraction with no Vite development server. Open the reviewer page and inspect `local-baseline:0`, including its C_AUTH evidence. Stop the reviewer process and verify a clear unavailable message; operational routes remain usable and disclose no reviewer data.

## Smaller documentation corrections

These do not call for new research experiments:

- `CLAUDE.md` still restricts work to Phase 1A and forbids W1/W3/adapters despite their subsequent approval and implementation. Update it to the actual v0.1 scope and accepted external-latency exception.
- README says W1/W3 are exercised by the grader study, but this study is W2-only.
- README points to `CITATION.cff`, which is absent from the uploaded archive. Add the intended citation file or correct the reference.
- The report says all 114 restored-change cases were written by hand. The implementation applies one programmed transformation across 114 scenarios. Describe that provenance accurately.
- `violate_then_revert()` appends `escalated` then `open`; it does not restore the status immediately before the mutation, which was already set by the correct policy. Its annotation acknowledges task failure. The existing study supports a missed historical side effect, not a demonstrated successful-task reward exploit. Align its docstring and reward wording with that narrower finding, or reserve a genuinely successful restored-change experiment for a separately versioned study.
- The operational demo exposes descriptive scenario IDs, demo names and walkthroughs containing answer/axis information. That is useful in a guided human demonstration, but it is not a blind browser-agent evaluation interface. State that limit explicitly rather than treating omitted JSON keys as proof of no label leakage.

## Handoff to Claude Code

Paste this with the report attached:

```text
Use the attached independent review of candidate fc2a301 to repair v0.1.

Reproduce the five numbered findings first, then implement focused fixes for:
1. Complete replay-derived manifest verification and checksum handling.
2. Nested payload/document aliasing that can mutate retained traces or state views.
3. Atomic per-session HTTP dispatch, deduplication, terminal retries and reset behavior.
4. Refund amount constraints consistent across public schema, action parsing and handlers.
5. Reviewer routing in the built application without exposing reviewer data on the operational API.

Make the listed documentation corrections. Keep the current project scope and research limitations. Preserve original model evidence and study artifacts. Do not silently replace historical results; document and version any necessary schema or serialization change.

No inference, training, paid services, new workflow families or public release. Local commits are authorized; do not push.

Add targeted regressions for the reproduced defects. Run the existing suite, Ruff, strict mypy, all six import contracts and the frontend build. Resolve the documented per-package coverage gap without weakening the requirement. Check the production UI from a clean extraction with Vite stopped. Verify original action replay and report any genuine version skew explicitly.

Return a new candidate archive and SHA-256, commit hash, each finding's before/after reproduction, exact command results and remaining limitations. Do not declare a finding fixed solely because the old tests pass.
```

The next milestone is a corrected, demonstrably reproducible v0.1 candidate. Training and additional model spending are unnecessary to complete these repairs.
