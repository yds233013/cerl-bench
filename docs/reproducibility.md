# Reproducibility

## R1 — Counter-based seed derivation, never stream-based

```
subkey = blake2b(root_seed || domain_label || counter)
```

Tool failure is a function of `(tool_id, per_tool_call_index)` read from an
explicit frozen schedule — **not** a global step counter and **not** a mutable
RNG stream. This is consequential, not stylistic: with a shared stream, an agent
making one extra read call would shift every downstream failure, so two policies
would face different worlds and could not be compared.

## R2 — No ambient nondeterminism

Banned in `src/cerl/`: wall clock, `random`, `uuid`, unordered iteration in
anything serialized. Enforced by a source scan *and* by a test that monkeypatches
`time`, `random` and `uuid` to raise and re-runs the oracle suite.

One bug this caught during Phase 1A: `frozenset` fields serialize in Python's set
iteration order, which is salted per process. Two runs of the same scenario
produced different JSON — and therefore different content hashes — for identical
state. `SortedFrozenSet` fixes it by serializing sorted. A same-process test
cannot see this class of bug; `tests/determinism` runs subprocesses under
different `PYTHONHASHSEED` values.

## R3 — Canonical serialization

One `canonical_json`: sorted keys, no whitespace, floats rejected, UTF-8.
Everything hashed goes through it.

## R4 — Content-addressed everything

`scenario_hash`, `verifier_version`, `predicate_library_hash`,
`generator_version`, `renderer_version`. A result whose versions differ is never
silently compared. `tests/golden` fails if the predicate library changes,
forcing a deliberate regeneration rather than a quiet overwrite.

## R5 — Environment pinning

`uv.lock` committed; `requires-python = ">=3.11,<3.14"`.

## R6 — Frozen snapshots regenerate byte-exactly

Every committed scenario is regenerated from `(generator_version, seed, axes)`
and compared byte-for-byte in CI, which is what keeps the mutator and the frozen
snapshot from drifting apart.

## Two claims, kept separate

**Claim 1 — environment and verifier determinism (guaranteed, offline).**
Given a recorded action sequence, replay reproduces every intermediate state
hash, every responder entry at its exact logical time and chain position, the
terminal hash, and the verdict — with no model in the loop, no network, no API
key. This is what "reproducible" means for the benchmark: *given these actions,
these are the scores, forever.*

**Claim 2 — model reproduction (implemented, tested independently).**
Re-running *generation* is a different and weaker claim: "the same recorded
turns give the same actions". `TranscriptCacheClient` replays recorded
request/response pairs offline, keyed by a content hash over the whole request
so drift cannot alias onto a wrong answer. A cache **miss raises** rather than
silently falling back to a live call — a cache that quietly reaches the network
is not a reproducibility mechanism.

The two claims are tested apart, in `tests/evaluation/`. One says "the same
actions give the same scores"; the other says "the same turns give the same
actions". A green replay must never be read as evidence that a model result was
reproduced.

**Live decoding is explicitly not guaranteed** and has not been run — see
`README.md` § Live evaluation.

## What offline verification actually checks

`cerl verify-manifest` compares a recorded run against **its own evidence**
rather than accepting freshly regenerated output as ground truth. It detects:

| Tampering | Detected via |
|---|---|
| an altered action | terminal and chain hashes |
| a doctored verdict | recomputed rubric, class, tool calls |
| a doctored metric block | aggregates recomputed from the episodes |
| a changed scenario file | `scenario_hash` |
| a changed initial world | `initial_state_hash` |
| a renamed responder rule | per-step `responder_rule` |
| an incompatible grading version | `predicate_library_hash` skew |
| a missing scenario | reported, never silently skipped |

One test breaks the socket layer before verifying, so "no network" is asserted
rather than assumed.
