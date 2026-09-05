# Budget accounting — assumptions, limits, and residual exposure

## The withdrawn claim

An earlier version of this work said the pre-flight estimate was "biased high, so
the realistic failure mode is stopping early, not overrunning," and treated that
as a spending guarantee.

**That was not a guarantee and the claim is withdrawn.** A heuristic that leans
high on typical input is not a bound on atypical input, and it says nothing at
all about requests whose outcome we never learn. What follows is what the system
actually provides: an *operational limit* with its assumptions written down.

## What is actually enforced

Before every request — including every retry — the client:

1. counts the **entire** request: system prompt, every prior turn in the
   conversation, and the tool schemas;
2. inflates that count by the counter's declared uncertainty margin;
3. prices it at the full `max_tokens` as output, since actual output (thinking
   included) is unknown until the response arrives;
4. refuses the request outright if that figure does not fit in the spendable
   balance.

The request is not sent when it does not fit. That is a real property, tested in
`tests/live/test_budget.py`, and it is the strongest statement available.

## The four quantities

Collapsing these is how an accounting system lies, so they are separate fields
and separate columns everywhere.

| Quantity | Meaning | Can it decrease? |
|---|---|---|
| **confirmed** | Billed usage read back from a response. Fact. | No |
| **reserved** | Held for a request in flight right now. | Yes, on a definite outcome |
| **unresolved** | Held **permanently** for a request that may have reached the provider but whose usage we never learned. | **No, ever** |
| **estimated** | A projection for a request not yet sent. Not money. | n/a |

Spendable balance = `cap − (confirmed + reserved + unresolved)`.

**Unresolved is the important one.** A timeout, a connection dropped after send,
or a response that arrives without usage all mean the same thing: we were
probably billed and cannot say how much. The charge is kept against the cap
forever. Only a request that provably never reached the provider releases its
hold, via `release_unsent`. Treating an ambiguous outcome as free is the single
easiest way to overspend, and it is the mistake this design exists to avoid.

## Token counting

| Counter | Used when | Margin | Accuracy |
|---|---|---|---|
| `ApiTokenCounter` | The provider exposes `messages.count_tokens` | 5% | Exact for input |
| `HeuristicTokenCounter` | Fallback when it does not | 25% | Approximate, unbounded error |
| `MockTokenCounter` | Offline tests only | 0% | Deterministic fixture |

The provider counter is preferred automatically. The heuristic's 25% is a
working allowance, **not a proof**: dense JSON tool schemas tokenize very
differently from prose, and its error is not bounded in either direction. A run
using it carries wider exposure, and the ledger records which counter was used
so the report says so.

The counter used for a run appears in `ledger.report()["token_counter"]` and in
the manifest's spend block. **The published cost projections in
`docs/live-pilot-proposal.md` were produced with the heuristic counter**, because
the `anthropic` SDK is an optional extra that is not installed in this
environment. They are estimates, and are labelled as such.

## What is accounted for

- **The full request.** System prompt, complete conversation history, and tool
  schemas. Counting only the newest message would under-reserve by the size of
  the whole transcript, which on a 40-step episode is most of the bill.
- **Thinking tokens.** Billed as output and drawn from the same `max_tokens`
  allowance, so reserving the full `max_tokens` as output already covers them.
  They are not a separate line and must not be added twice.
- **Pricing modifiers.** Cache reads (0.1×) and cache writes (1.25×) are priced
  at their own rates from `response.usage`, not folded into the input rate.
- **Application retries.** Each attempt reserves in its own right. A loop that
  reserved once per logical call would underbill by up to 3×.
- **SDK automatic retries.** The SDK retries by default; an automatic retry is a
  billable request the ledger never saw. `AnthropicTransport` is constructed
  with `max_retries=0` so that every retry passes through the ledger. A test
  asserts that default so a future change fails loudly.
- **Timeouts.** Charged as unresolved, never released.
- **Resumption.** The ledger persists and is resumed by
  `SpendLedger.resume`. A run restarting from zero would grant itself the whole
  cap again, turning "a $50 cap" into $50 per attempt. Resume refuses a changed
  cap or a different model rather than silently adopting it. An in-flight
  reservation is not restored as spendable — its outcome is unknown, so it is
  already recorded as unresolved.

## Crash safety

Two files, two jobs. The **snapshot** (`ledger.json`) carries cap and model
provenance to check a resume against. The **journal** (`ledger.jsonl`) carries
the money, one record per request.

Ordering is the guarantee: the reservation is written, flushed, and `fsync`ed
**before** the request is sent, and the outcome is appended after it returns.

Recovery applies one rule: **a reservation with no outcome becomes an unresolved
charge.** After the fact we cannot distinguish "never sent" from "sent and
billed", so the conservative reading is the only safe one.

| On recovery | Behaviour |
|---|---|
| Confirmed usage | Preserved exactly. It was really billed. |
| Reservation with no outcome | Charged as unresolved, listed in `orphaned_requests` |
| The allowance | **Not reset.** Recovery restores the spend position |
| Uncertain requests | **Never replayed.** Retrying is a person's decision, not a side effect of restarting |
| Cap and model | Still checked; a changed cap or model is refused |
| Truncated final record | Discarded; everything before it is intact |
| Snapshot vs journal disagreement | The journal wins — it is the finer record |

Covered by `tests/live/test_crash_recovery.py`, which interrupts at the two
places money is actually at risk: **inside `transport.create`** (the request is
on the wire and no outcome is recorded) and **between the journal write and the
snapshot**. Recovery is then performed from the files on disk alone, as a
restarted process would see them. The crash is raised as a `BaseException` so it
is not caught by the client's retry handler — a real crash gets no chance to
record an outcome, and a catchable one would test the wrong path.

### Remaining limitation

The journal makes recovery consistent with **what this process observed**. It
cannot make it consistent with what the provider billed. If a request is sent and
the provider bills it but the process dies before any outcome is recorded, the
journal charges the worst-case reservation, which is an over-estimate whenever
the true response was shorter. In the other direction, a request that never left
the socket is also charged. Both errors are conservative — they shrink the
remaining allowance — and neither can be resolved without reading the provider's
own billing record.

`fsync` durability is also only as good as the storage stack beneath it; a drive
that lies about flushing can still lose the tail of the journal.

## Sequential execution

The pilot issues one request at a time. This is a deliberate accounting choice,
not a performance limitation: with a single request outstanding, the spendable
balance at each decision point is unambiguous and the ledger reads as an
auditable line-by-line account. Concurrency would buy wall-clock time at the cost
of the one property that makes the spending claim checkable. The ledger is
nonetheless thread-safe, so the invariant does not depend on that policy holding
forever.

## Residual exposure — what could still go wrong

Stated plainly, because a limit whose failure modes are hidden is not a limit.

1. **Input tokenization error, when using the heuristic counter.** If the real
   token count exceeds the heuristic + 25%, each request costs more than
   reserved. The overrun per request is bounded by the error, not by the cap.
   *Mitigation:* install the SDK so `ApiTokenCounter` is used; its input count is
   exact. *Unmitigated risk if you do not:* proportional to the error.
2. **Unresolved charges are estimates of an unknown.** An unresolved request is
   charged at its worst-case reservation. The real charge could be lower (the
   request never landed) or, if the provider billed a longer response than
   `max_tokens` allows — which should be impossible — higher. The cap is
   enforced against the estimate.
3. **Pricing drift.** `PRICING` is a table cached on 2026-09-05. If published
   prices rise, every figure here understates. *Mitigation:* re-check before a
   run; an unpriced model raises rather than costing zero.
4. **Provider-side additions.** The request the provider prices is not
   byte-identical to what we serialise (server-side scaffolding). The 5% margin
   on the API counter is an allowance for this, not a measurement of it.
5. **~~A crash between the provider billing us and the ledger being written.~~**
   **Resolved.** A per-episode snapshot left a loss window an episode wide: a
   crash after twelve confirmed requests recovered as zero, and the resumed run
   handed itself the allowance back — measured at 147¢ invisible on a 1000¢ cap.
   A write-ahead journal now records every request. Remaining exposure is
   described below.

## The honest bottom line

**A strict dollar guarantee cannot be established from outside the provider.**
We do not control billing, cannot observe it in real time, and cannot know the
cost of a request whose response never arrived.

What can be stated:

> With the provider token counter, the ledger refuses any request whose
> worst-case price does not fit in the remaining balance, charges every retry and
> every ambiguous outcome against that balance, and carries spend across
> resumption. For the proposed 20-episode pilot the modelled worst case is
> **$50.54** and the configured cap is **$51.00**. Exceeding it requires one of
> the five failure modes above, of which only pricing drift could plausibly
> exceed it by a large factor.

Anyone authorising the run should read that as an operational limit around $51,
not a contractual ceiling — and should set a provider-side spending limit as
well, which is the only actual ceiling available.
