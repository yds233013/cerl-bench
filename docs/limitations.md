# Limitations

Stated plainly, because a benchmark that hides its limits is worse than one that
has fewer of them.

- **Prose quality is not graded.** Ticket comments and Slack messages are scored
  by structural predicates only (does it cite the charge id, does it name the
  approval, does it avoid sensitive literals). Whether the writing is *good* is
  out of scope, because grading it would require an LLM judge and destroy
  reproducibility.
- **Language is templated.** Realism of surface text is a separate axis from
  safety; conflating them would make both unmeasurable.
- **One domain, three families.** W1, W2 and W3 all sit in enterprise billing
  support. Cross-*domain* transfer is untested.
- **`C_DISCLOSE` is exercised only by W3**, and only against a fixed list of
  sensitive literals. An agent could leak by paraphrase — "the card ending in
  one-eight-eight-one" — and the content check would miss it. The check is a
  floor, not a proof of non-disclosure.
- **Difficulty matching is approximate.** CF variants preserve entity counts and
  brief length, and oracle call counts within one call — except where obtaining
  an approval genuinely costs turns, which is allowed a wider band and documented
  in the test.
- **No model has been evaluated.** The prompt-only integration is complete and
  tested offline, but no live run has occurred and no spending budget is
  authorised, so no generalization number exists and none is claimed. Every
  fixture is stamped `synthetic`.
- **Verification mechanisms have limits worth stating plainly.** Import-linter
  contracts prove a module was not imported; they are not a security isolation
  proof, and a determined implementation could still read a file off disk.
  Observation-string scans check the rendered text against a curated list of
  privileged tokens; they would not catch privileged information that had been
  paraphrased or encoded. Both are defence in depth, not guarantees.
- **W1's `identity_evidence` is stratified, not held out.** It cannot serve as a
  counterfactual axis without distorting the comparison — see
  `docs/pairing.md`. This narrows what W1 can measure.
- **The oracle defines the efficiency denominator.** `r_efficiency` is normalized
  against the oracle's call count, which embeds our view of the right amount of
  work. Whether it belongs in the headline metric at all is open decision O1.
- **Single process, in memory.** No concurrency, no persistence, no adversarial
  multi-tenancy.
