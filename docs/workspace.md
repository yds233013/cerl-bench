# The support & billing workspace

An interactive enterprise application over the W2 duplicate-charge workflow. A
person can work a ticket, see the real consequences, reset, and inspect recorded
runs.

## Setup and start

```bash
uv sync                       # Python side (no new runtime dependencies)
cd app && npm install         # frontend, once

# Three processes. Two servers on purpose -- see "The boundary" below.
uv run cerl serve             # operational API   :8000
uv run cerl serve-review      # reviewer API      :8001   (privileged)
cd app && npm run dev         # workspace UI      :5173
```

Then open **http://127.0.0.1:5173/**. The reviewer is at
**http://127.0.0.1:5173/?review**.

`cerl serve` builds the demo fixture on first run if it is missing.

### The packaged path -- two origins, not one

The development layout above works because Vite proxies `/api` to `:8000` and
`/review` to `:8001`. **That proxy is a development-server feature and is not in
the built application.** Built and served by the Python processes, the two
surfaces are two origins:

```bash
cd app && npm run build       # once; writes app/dist
uv run cerl serve             # workspace  http://127.0.0.1:8000/
uv run cerl serve-review      # reviewer   http://127.0.0.1:8001/?review
```

Each server serves its **own** copy of the page, so each page's relative
requests land on the server that answers them. Opening `?review` on `:8000`
does not work and is not supposed to: the operational server 404s every
`/review/*` route, and it must keep doing so. That rejection is the boundary;
the routing fix was to give the reviewer a page of its own, never to relax it.

`cerl serve-review --api-only` serves the API alone; `--ui PATH` points at a
different build.

## Architecture

```
  browser :5173  ──/api──▶  cerl serve        :8000   operational
        │                     └── WorkspaceSession ── CerlEnv (the simulator)
        └────────/review──▶  cerl serve-review :8001   reviewer (privileged)
                              └── recorded manifests under evidence/
```

The HTTP layer is `http.server` and a small router. The project carries three
runtime dependencies and a web framework would be the largest thing in it; a
thin adapter is both the requirement and the honest shape of the problem.

**The simulator is the only source of truth.** The adapter creates a `CerlEnv`,
dispatches validated actions into it, and reads nothing else. It applies no
policy of its own.

## What the workspace shows

Only what a tool call has returned. The inbox starts **empty** — you have a
ticket reference in the brief and must go and read it, the same way the
environment's observation interface works for an agent. A workspace that
rendered the whole world would hand the operator free information and quietly
break the boundary that makes an evaluated score mean anything.

## Logical time

Advances **only** when an action is dispatched, by that action's documented tick
cost. Rendering, polling, refreshing and waiting do not move it. This matters:
approvals expire in logical time, so a workspace that advanced the clock while
you read would expire an approval you were about to use.

## The boundary

Two servers, not two routers behind a flag.

| | operational :8000 | reviewer :8001 |
|---|---|---|
| Branch, rubric, permitted diffs | never sent | shown |
| Violation flags, verdicts | never sent | shown |
| Imports `cerl.verify` / `cerl.reference` | **no** | yes |

The operational module has no code path to a verdict, and its responses are
assembled field by field rather than dumped, so a field added to
`FrozenScenario` later cannot leak in by default. During agent evaluation the
reviewer process is simply not running, which is stronger than hiding a panel.
Asserted in `tests/app/test_workspace.py` over the serialised bytes.

### What that boundary does not claim

**This is a guided human demonstration, not a blind evaluation interface.** The
operational API withholds the *verdict machinery* — branch, rubric, permitted
diffs, violation flags. It does not withhold the answer.

Three things it sends are, in substance, labels:

- **Descriptive scenario ids.** `…__appr-expired__amt-above_threshold__…`
  names the approval axis and the amount band in plain text.
- **Demo titles.** "Escalate when authorisation is missing" states the required
  decision.
- **Walkthroughs.** Each demo ships an ordered list of the steps to take.

That is correct for what the app is for: showing a person what the workflow
feels like, and letting them see a violation latch. It would be wrong for
measuring an agent. **No number produced by driving this UI — by a human or a
browser agent — is a benchmark result**, and the absence of rubric keys in the
JSON is not evidence to the contrary. Blind evaluation runs through
`cerl eval`, against `Observation` objects, with scenario ids the agent never
reads. Making this UI evaluation-safe would mean opaque ids, no titles and no
walkthroughs, and is not in v0.1 scope.

## What the app deliberately does not do

It adds **no** policy enforcement. An unauthorised refund succeeds, exactly as
it does in the environment, and latches `C_AUTH`. A UI guard would silently
remove the dependent variable the benchmark exists to measure. Genuine backend
interlocks are untouched — an over-refund is still denied by the simulator.

## Duplicate submissions and concurrency

The client generates one `submission_id` per user gesture. The server records
it and returns the original result if it sees it again, so a double click, a
re-render or a network retry cannot dispatch an action twice. A request without
one is refused.

Deduplication alone is not enough, because `http.server` is threaded: two
requests carrying the same token can both pass the "have I seen this?" check
before either records it, and the action then executes twice. The check, the
dispatch, the recording and the view are therefore **one transaction under a
per-session lock** — the whole sequence, not the lookup alone. Three
consequences, each with a regression over real HTTP in
`tests/review/test_http_concurrency.py`:

- A token is bound to the **action** that first used it. Reusing it with a
  different action is a `409`, not a silent return of an unrelated record.
- The token is resolved **before** the episode-finished check, so retrying a
  terminal action is idempotent while a genuinely new action after the end is
  still refused.
- A reset bumps a **generation** counter. A request may carry the generation it
  was composed against; one from before a reset is refused rather than applied
  to the new episode.

## The three demonstrations

| Demo | Scenario source | Provenance |
|---|---|---|
| Refund with a valid approval | `corpus/train` | frozen corpus 2.0.0, training partition, `refund_now` |
| Obtain approval, then refund | **demo fixture** | authored at seed 90001, **development-exposed** |
| Escalate when authorisation is missing | `corpus/train` | frozen corpus 2.0.0, training partition, `escalate_unapproved` |

`request_then_refund` has **no training-partition scenario** — all 13 sit in
sibling groups carrying a registered holdout — so rather than spend an
evaluation scenario on a demo, one was materialised into `scenarios/demo/`.
It is **not held-out evaluation data** and no result may be reported from it.
The graded corpus, its hashes and the split assignments are untouched.

Each demo resets to its exact initial state, verified by state hash.
