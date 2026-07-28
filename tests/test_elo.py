"""Bradley-Terry harness tests (Phase 4 chunk 3).

The two tests that look odd are the two the spec singles out:

- stability across ITERATION COUNTS (200/2k/20k), never a successive-
  difference tolerance — an undefeated player creeps at ~constant Elo per
  decade of iterations with 1/k step sizes, which a tolerance reads as
  slow convergence and converts into a finite, wrong number. Refusal
  (Ford) plus cross-count agreement is the discriminating pair.
- the all-draws matrix must bootstrap to CI width EXACTLY zero: each
  (pair, colour) cell resamples its own fixed total, so a degenerate cell
  is reproduced verbatim. An i.i.d. bootstrap over pooled games reports
  sd 0.021 here (measured, PLAN.md) — nonzero width on this fixture means
  the stratification broke.
"""

import numpy as np
import pytest

from rl.selfplay.elo import (
    ELO_PER_LOG,
    bootstrap,
    fit_bt,
    ford_connected,
    rate,
)


def bt_counts(rng, strengths, games=100):
    """Sample a full colour-alternated round robin from known BT strengths
    (no draws): the ground truth the fits are compared against."""
    names = sorted(strengths)
    counts = {}
    for a_idx, a in enumerate(names):
        for b in names[a_idx + 1 :]:
            p_win = strengths[a] / (strengths[a] + strengths[b])
            for first, second, chance in ((a, b, p_win), (b, a, 1 - p_win)):
                wins = int(rng.binomial(games // 2, chance))
                counts[(first, second)] = (wins, 0, games // 2 - wins)
    return counts


def test_two_player_ratings_are_analytic():
    """60/40 head-to-head: the BT MLE is p_a/p_b = 60/40 exactly, so the
    gap must be 400*log10(1.5) Elo regardless of colour split."""
    counts = {("a", "b"): (35, 0, 15), ("b", "a"): (25, 0, 25)}
    result = rate(counts, anchor="b")
    assert result.ratings["b"] == 0.0
    assert result.ratings["a"] == pytest.approx(400 * np.log10(1.5), abs=1e-6)


def test_draws_count_exactly_half():
    """50 wins + 50 draws must fit identically to 75/25 decisive."""
    with_draws = rate({("a", "b"): (50, 50, 0)}, anchor="b")
    decisive = rate({("a", "b"): (75, 0, 25)}, anchor="b")
    assert with_draws.ratings["a"] == pytest.approx(decisive.ratings["a"], abs=1e-9)


def test_ratings_are_order_independent():
    """Same games under permuted names -> identical ratings (MM updates
    every player from the same frozen iterate; the bookkeeping must not
    reintroduce an order)."""
    rng = np.random.default_rng(0)
    strengths = {"p0": 1.0, "p1": 2.0, "p2": 4.0, "p3": 8.0}
    counts = bt_counts(rng, strengths)
    renamed = {("z" + a, "z" + b): cell for (a, b), cell in counts.items()}
    base = rate(counts, anchor="p0").ratings
    flipped = rate(renamed, anchor="zp0").ratings
    for name, value in base.items():
        assert flipped["z" + name] == pytest.approx(value, abs=1e-9)


def test_undefeated_player_is_refused_by_the_fit_and_dropped_by_rate():
    counts = {
        ("a", "b"): (10, 0, 0),  # a is undefeated
        ("b", "c"): (5, 0, 5),
        ("c", "a"): (0, 0, 10),
    }
    with pytest.raises(ValueError, match="Ford"):
        fit_bt(np.array([[0.0, 10.0], [0.0, 0.0]]))
    result = rate(counts, anchor="b")
    assert result.ceilinged == ["a"]
    assert set(result.ratings) == {"b", "c"}


def test_perfect_scorer_drop_cascades():
    """Removing the undefeated top can leave the next player undefeated
    among the remainder — the drop must iterate, not single-pass."""
    counts = {
        ("a", "b"): (10, 0, 0),
        ("a", "c"): (10, 0, 0),
        ("b", "c"): (10, 0, 0),  # b undefeated once a is gone
        ("c", "d"): (5, 0, 5),
        ("d", "e"): (6, 0, 4),
    }
    result = rate(counts, anchor="d")
    assert result.ceilinged == ["a", "b"]
    assert set(result.ratings) == {"c", "d", "e"}


def test_anchor_dropped_raises():
    counts = {("a", "b"): (10, 0, 0), ("b", "c"): (5, 0, 5)}
    with pytest.raises(ValueError, match="anchor"):
        rate(counts, anchor="a")


def test_ford_connected_is_about_direction_not_contact():
    """a and b played (a swept), so the graph is connected as an UNDIRECTED
    graph — but no score flows b -> a and Ford must say no."""
    assert not ford_connected(np.array([[0.0, 10.0], [0.0, 0.0]]))
    assert ford_connected(np.array([[0.0, 9.5], [0.5, 0.0]]))  # one draw back


def test_stability_across_iteration_counts():
    """The locked test shape: 200 vs 2k vs 20k iterations agree to well
    under an Elo on a Ford-connected ladder spanning ~360 Elo. Creep would
    move ~370 Elo per decade; agreement at this tolerance refutes it."""
    rng = np.random.default_rng(1)
    strengths = {f"p{k}": 1.6**k for k in range(6)}
    counts = bt_counts(rng, strengths, games=100)
    fits = [rate(counts, anchor="p0", iterations=n).ratings for n in (200, 2000, 20000)]
    for name in fits[0]:
        values = [fit[name] for fit in fits]
        assert max(values) - min(values) < 0.5, f"{name}: {values}"


def test_recovers_known_strengths():
    """With plenty of games the fit should land near the generating truth:
    p ratio 8 between the ends = ~361 Elo."""
    rng = np.random.default_rng(2)
    strengths = {"p0": 1.0, "p1": 2.0, "p2": 4.0, "p3": 8.0}
    counts = bt_counts(rng, strengths, games=2000)
    ratings = rate(counts, anchor="p0").ratings
    for name, p in strengths.items():
        expected = ELO_PER_LOG * np.log(p)
        assert ratings[name] == pytest.approx(expected, abs=25)


def test_bootstrap_all_draws_has_exactly_zero_width():
    counts = {("a", "b"): (0, 40, 0), ("b", "a"): (0, 40, 0),
              ("b", "c"): (0, 40, 0), ("c", "b"): (0, 40, 0),
              ("a", "c"): (0, 40, 0), ("c", "a"): (0, 40, 0)}
    result = bootstrap(counts, anchor="a", B=50, iterations=500)
    assert result.failed == 0
    for name, (lo, hi) in result.intervals.items():
        assert lo == 0.0 and hi == 0.0, name


def test_bootstrap_is_seeded_and_has_width_on_real_data():
    rng = np.random.default_rng(3)
    counts = bt_counts(rng, {"p0": 1.0, "p1": 2.0, "p2": 4.0}, games=200)
    first = bootstrap(counts, anchor="p0", B=40, seed=11, iterations=500)
    again = bootstrap(counts, anchor="p0", B=40, seed=11, iterations=500)
    assert first.intervals == again.intervals
    lo, hi = first.intervals["p2"]
    assert hi > lo
    truth = ELO_PER_LOG * np.log(4.0)
    assert lo < truth < hi


def test_bootstrap_flags_failed_resamples_instead_of_fitting_them():
    """A cell with a 1-in-30 upset resamples to a sweep ~36% of the time;
    those resamples drop the swept player (reduced rated_in) or, when the
    anchor is on the wrong end of the sweep, fail outright. Either way no
    non-Ford matrix is ever fitted, and the counts must say which path
    fired."""
    counts = {
        ("a", "b"): (15, 0, 15), ("b", "a"): (15, 0, 15),
        ("b", "c"): (29, 0, 1), ("c", "b"): (1, 0, 29),
    }
    result = bootstrap(counts, anchor="a", B=60, seed=5, iterations=500)
    assert result.failed == 0  # the anchor's own cells are never swept
    assert result.rated_in["a"] == 60
    assert result.rated_in["b"] == 60
    assert result.rated_in["c"] < 60  # dropped whenever the resample sweeps it

    # Anchored at the sweep-side player instead, the same drops become
    # whole-resample failures — and they must be COUNTED, not swallowed.
    anchored_at_c = bootstrap(counts, anchor="c", B=60, seed=5, iterations=500)
    assert anchored_at_c.failed > 0
    assert anchored_at_c.rated_in["c"] == 60 - anchored_at_c.failed
