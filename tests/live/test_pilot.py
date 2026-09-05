"""The pilot selection must be deterministic and cover every outcome branch."""

from __future__ import annotations

import pytest

from cerl.eval import pilot
from cerl.reference.registry import families as oracle_families


@pytest.fixture(scope="module")
def projection(all_frozen):
    return pilot.project(list(all_frozen))


def test_selection_covers_every_branch_of_every_family(all_frozen, projection):
    """A pilot that skips a branch cannot report on that branch. Assert coverage."""
    corpus = {(s.family, s.branch) for s in all_frozen}
    assert projection.branches == tuple(sorted(corpus))
    assert len(corpus) == 10, sorted(corpus)
    assert {f for f, _ in corpus} == set(oracle_families())


def test_every_branch_gets_the_declared_episode_count(projection):
    counts: dict[tuple[str, str], int] = {}
    for episode in projection.episodes:
        counts[(episode.family, episode.branch)] = (
            counts.get((episode.family, episode.branch), 0) + 1
        )
    assert set(counts.values()) == {pilot.EPISODES_PER_BRANCH}


def test_selection_is_deterministic(all_frozen):
    """No sampling and no seed: the set is a function of the corpus alone."""
    first = pilot.select(list(all_frozen))
    shuffled = list(reversed(list(all_frozen)))
    second = pilot.select(shuffled)
    assert [s.scenario_id for s in first] == [s.scenario_id for s in second]


def test_worst_case_exceeds_expected_by_a_wide_margin(projection):
    """The gap is the finding, not a rounding detail.

    Input grows with the transcript, so an episode that flails to its step budget
    costs far more than one solved in eleven steps. A cap set from the mean would
    not bound the run.
    """
    expected = projection.total_cents(worst_case=False, cached=True)
    worst = projection.total_cents(worst_case=True, cached=True)
    assert worst > expected * 5


def test_recommended_cap_bounds_the_worst_case(projection):
    cap = projection.recommended_cap_cents()
    assert cap > projection.total_cents(worst_case=True, cached=True)
    assert cap % 100 == 0


def test_caching_is_cheaper_than_not_caching(projection):
    assert projection.total_cents(worst_case=True, cached=True) < projection.total_cents(
        worst_case=True, cached=False,
    )


def test_projection_matches_the_published_proposal(projection):
    """Guards the doc against silent drift as the corpus changes."""
    assert len(projection.episodes) == 20
    assert projection.model == "claude-opus-5"
    assert projection.max_tokens == 2048
    expected = projection.total_cents(worst_case=False, cached=True)
    worst = projection.total_cents(worst_case=True, cached=True)
    assert expected == pytest.approx(403.0, abs=10.0), expected
    assert worst == pytest.approx(4936.0, abs=60.0), worst
    assert projection.recommended_cap_cents() == 5000
