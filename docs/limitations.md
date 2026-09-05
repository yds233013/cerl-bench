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
- **One domain, one family in Phase 1A.** W2 only. Cross-family transfer is a
  Phase 1B/5 question.
- **`C_DISCLOSE` is not exercised by W2.** It belongs to W3. The class exists in
  the vocabulary but Phase 1A makes no claim about it, and a test asserts W2 does
  not pretend to cover it.
- **Difficulty matching is approximate.** CF variants preserve entity counts and
  brief length, and oracle call counts within one call — except where obtaining
  an approval genuinely costs turns, which is allowed a wider band and documented
  in the test.
- **No evaluated agent yet.** Phase 1A ships the oracle and scripted fixtures
  only, so no generalization number exists and none is claimed.
- **The oracle defines the efficiency denominator.** `r_efficiency` is normalized
  against the oracle's call count, which embeds our view of the right amount of
  work. Whether it belongs in the headline metric at all is open decision O1.
- **Single process, in memory.** No concurrency, no persistence, no adversarial
  multi-tenancy.
