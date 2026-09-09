"""Training protocol **v2**. Separately versioned; v1 is frozen for the record.

v1 is what the recorded pilot ran and is preserved exactly so ``ef6942e`` stays
reproducible. v2 is the corrected contract, and it is a different protocol
rather than an edit because the two produce different numbers: a result carried
over between them would be comparing two experiments.

What changed, and why each change was forced:

* **Exact token provenance (R1).** v1 reconstructed the trained span by decoding
  the generated tokens to text, re-rendering the whole conversation, and
  searching for that text from position zero. The search could select a span in
  the system prompt or an observation -- reproduced at 32 tokens -- and the
  re-render changed the conditioning, because Qwen's template drops the empty
  thinking block from earlier assistant turns. v2 records the exact prompt and
  generated token ids at each ``generate`` call and scores each turn under the
  context it was actually sampled in.
* **The complete public tool contract (R2).** v1 showed names and argument names
  only. v2 renders descriptions, types, requiredness, enums and numeric bounds.
* **Matched sampling and scoring distributions (R3).** v1 left the model's
  ``top_k=20`` default active while differentiating the unfiltered softmax. v2
  neutralises every truncation so the sampled distribution is the one the loss
  differentiates.
* **Feasible limits (R9/§3a).** 24 actions; 8,192 prompt tokens, re-measured
  against the *complete* contract.
"""

from __future__ import annotations

from cerl_rl import protocol

PROTOCOL_VERSION = "2.0.0"

MODEL_ID = protocol.MODEL_ID
MODEL_REVISION = protocol.MODEL_REVISION
FAMILY = protocol.FAMILY
REWARD_NAME = protocol.REWARD_NAME

#: Selections are unchanged: same canonical train partition, same five
#: validation scenarios, read from the same committed manifest.
training_selection = protocol.training_selection
validation_selection = protocol.validation_selection

# -- limits -----------------------------------------------------------------
#: All 15 selected gold trajectories terminate within 16 actions; 24 leaves
#: headroom for a correct policy that takes a redundant read or recovers from a
#: malformed turn. 16 is **not** a proven minimum -- the oracle is one correct
#: policy, not the shortest.
MAX_ACTIONS = 24

#: Re-measured with the **complete** tool contract, which is 818 tokens larger
#: than the abbreviated menu: the peak prompt across the 15 gold trajectories at
#: 24 actions is 4,457 tokens, so 4,096 would not fit and 8,192 leaves 3,735
#: spare. The model's context is 40,960.
#:
#: .. note::
#:    This establishes that the prompt *fits the context window*. Whether an
#:    8,192-token forward pass fits comfortably in this machine's memory during
#:    training is **not** established here -- it needs a forward pass, which the
#:    correction pass was not permitted to run.
MAX_PROMPT_TOKENS = 8192

#: The longest reference action encodes to 99 tokens.
MAX_NEW_TOKENS = 160

# -- decoding ---------------------------------------------------------------
EVAL_TEMPERATURE = 0.0
EVAL_SEED = protocol.EVAL_SEED
TRAIN_TEMPERATURE = 1.0

#: Every probability-altering default, neutralised explicitly (R3).
#:
#: The model's own ``generation_config.json`` sets ``top_k=20``, and v1 passed
#: ``top_p=0.95``. Both truncate the sampling distribution, while the loss
#: differentiates the full softmax -- so the behaviour policy and the scored
#: policy were different distributions. Rather than correct for that after the
#: fact with importance weights (which is delicate precisely because truncation
#: removes support), v2 removes the truncation: the distribution sampled is the
#: distribution differentiated.
SAMPLING_NEUTRALISED: dict[str, object] = {
    "top_k": 0,
    "top_p": 1.0,
    "typical_p": 1.0,
    "repetition_penalty": 1.0,
    "no_repeat_ngram_size": 0,
    "renormalize_logits": False,
    # NOTE: ``min_p`` is deliberately ABSENT rather than set to 0.0.
    #
    # transformers builds the min-p warper on ``if generation_config.min_p is
    # not None``, with no value test -- and ``0.0 is not None``. So passing
    # ``min_p=0.0`` to mean "no min-p filtering" *constructs* the warper, which
    # then runs ``argsort``/``gather``/``scatter`` across the whole 151,936-token
    # vocabulary. On this machine that aborted the process with
    # ``MPSTemporaryNDArray ... total bytes of NDArray > 2**32`` and killed the
    # first v2 training trial (see evidence/mps-probe/REPORT.md).
    #
    # Omitting the key leaves it ``None``, so no warper is built. The intended
    # semantics -- no min-p filtering -- are identical.
    #
    # Every other key here is safe because its guard also tests the value:
    # ``top_k != 0``, ``top_p < 1.0``, ``typical_p < 1.0``,
    # ``repetition_penalty != 1.0``, ``no_repeat_ngram_size > 0``,
    # ``renormalize_logits is True``. ``min_p`` was the only bare None check.
}
