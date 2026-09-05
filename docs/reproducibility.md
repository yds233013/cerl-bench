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

**Claim 2 — model reproduction (best-effort, Phase 1B+).**
Re-running *generation* is a different and weaker claim. A committed transcript
cache makes it byte-exact offline; live pinned decoding is explicitly **not
guaranteed**, because provider-side changes can alter outputs. Phase 1A ships no
evaluated agent, so only Claim 1 is currently exercised.
