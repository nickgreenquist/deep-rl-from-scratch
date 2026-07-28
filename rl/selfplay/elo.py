"""Bradley-Terry ratings for the chunk-3 tournament (Phase 4).

Input is the tournament's raw material: a dict mapping `(first, second)`
player-name pairs — first player listed first, so each key is one COLOUR
of one matchup — to `(first_wins, draws, second_wins)` counts. Draws count
half a win each (the locked convention), which the score matrix absorbs so
the MM update never sees them specially.

Everything here follows Hunter (2004) and the locked spec, and the guards
exist because the failure modes are quiet:

- **Ford's condition (Assumption 1) is checked before every fit, and it is
  necessary, not merely sufficient** (Hunter Lemma 1(a)): without it the
  MLE does not exist. An undefeated player does not diverge to inf in a
  way a test would notice — it creeps at a constant ~372 Elo per decade of
  iterations while the step size decays like 1/k, so a successive-
  difference convergence test reads it as slow convergence and returns a
  finite, wrong, tolerance-dependent number. Hence `fit_bt` REFUSES a
  non-Ford matrix, and the committed test asserts stability across
  iteration counts (200/2k/20k), never finiteness.

- **Perfect scorers are dropped iteratively and reported with a
  floor/ceiling** (Ordo's approach) rather than smoothed away: the locked
  spec rejects pseudo-count priors outright (if one is ever wanted it is
  phantom-player rho, named explicitly — "half a virtual win and loss" is
  ambiguous by a factor of J between Glickman's two readings). Dropping
  is iterative because removing the top player can make the next one
  undefeated among the remainder.

- **The bootstrap is stratified by (pair, colour)** — each input cell is
  resampled as its own multinomial with its total fixed, preserving the
  campaign's exact colour balance; an i.i.d. bootstrap over pooled games
  destroys that balance and reports sd 0.021 where the truth is 0
  (measured, PLAN.md). Resamples that fail the fit's preconditions
  (anchor dropped, Ford violated) are FLAGGED AND SKIPPED, never fitted
  anyway — Hunter p. 402's trap: at B = 1000 assume at least one resample
  violates Assumption 1.

Ratings are Elo-scaled log-strengths, `400/ln(10) * log p`, reported
relative to a caller-named anchor (the tournament anchors `alphabeta2`
at 0).
"""

import math
from dataclasses import dataclass

import numpy as np

ELO_PER_LOG = 400.0 / math.log(10.0)

Counts = "dict[tuple[str, str], tuple[int, int, int]]"


def _score_matrix(counts, players: list[str]) -> np.ndarray:
    """S[i, j] = i's score against j, both colours pooled, draws as 0.5.
    Colour itself carries no BT term — the tournament alternates first
    player exactly N/2 each way, so seat advantage cancels by design."""
    index = {name: i for i, name in enumerate(players)}
    S = np.zeros((len(players), len(players)))
    for (first, second), (first_wins, draws, second_wins) in counts.items():
        i, j = index[first], index[second]
        S[i, j] += first_wins + 0.5 * draws
        S[j, i] += second_wins + 0.5 * draws
    return S


def ford_connected(S: np.ndarray) -> bool:
    """Assumption 1: the digraph with an edge i -> j wherever S[i, j] > 0
    is strongly connected — for every split of the players into two
    nonempty groups, someone in each scored against someone in the other.
    Strong connectivity via reachability from node 0 in the graph and its
    transpose."""
    n = len(S)
    if n <= 1:
        return True
    positive = S > 0

    def reaches_all(adj) -> bool:
        seen = {0}
        frontier = [0]
        while frontier:
            for j in np.flatnonzero(adj[frontier.pop()]):
                if int(j) not in seen:
                    seen.add(int(j))
                    frontier.append(int(j))
        return len(seen) == n

    return reaches_all(positive) and reaches_all(positive.T)


def drop_perfect_scorers(
    S: np.ndarray, players: list[str]
) -> "tuple[list[int], list[str], list[str]]":
    """Indices to keep, plus the dropped names: ceiling = nobody scored
    against them (undefeated, not even a draw), floor = they scored against
    nobody. Iterative on purpose — removing an undefeated player can leave
    the next one undefeated among the remainder."""
    kept = list(range(len(players)))
    floored: list[str] = []
    ceilinged: list[str] = []
    while True:
        sub = S[np.ix_(kept, kept)]
        scored = sub.sum(axis=1)
        conceded = sub.sum(axis=0)
        drop = [
            (i, ceilinged if conceded[k] == 0 else floored)
            for k, i in enumerate(kept)
            if scored[k] == 0 or conceded[k] == 0
        ]
        if not drop:
            return kept, floored, ceilinged
        for i, bucket in drop:
            bucket.append(players[i])
            kept.remove(i)


def fit_bt(S: np.ndarray, iterations: int = 2000) -> np.ndarray:
    """Log-strengths by Hunter's MM: p_i <- W_i / sum_j N_ij / (p_i + p_j).
    Order-independent by construction (every player updates from the same
    frozen iterate). Runs exactly `iterations` steps — convergence for
    Ford-connected matrices is what the 200/2k/20k stability test pins,
    and non-Ford matrices are refused here rather than trusted to any
    stopping rule (see the module docstring for why a tolerance cannot
    detect the creep)."""
    if not ford_connected(S):
        raise ValueError("score matrix violates Ford's condition; no BT MLE exists")
    N = S + S.T
    W = S.sum(axis=1)
    p = np.ones(len(S))
    for _ in range(iterations):
        P = p[:, None] + p[None, :]
        p = W / (N / P).sum(axis=1)
        p /= np.exp(np.log(p).mean())  # rescale only: MM is scale-invariant
    return np.log(p)


@dataclass
class EloResult:
    ratings: "dict[str, float]"  # anchor at exactly 0.0
    floored: "list[str]"  # dropped: scored nothing; below every rated player
    ceilinged: "list[str]"  # dropped: conceded nothing; above every rated player


def rate(counts, anchor: str, iterations: int = 2000) -> EloResult:
    """Fit every player named in `counts`, anchored at `anchor` = 0."""
    players = sorted({name for pair in counts for name in pair})
    for pair, cell in counts.items():
        if sum(cell) == 0:
            raise ValueError(f"pair {pair} has zero games")
    S = _score_matrix(counts, players)
    kept, floored, ceilinged = drop_perfect_scorers(S, players)
    if anchor not in (players[i] for i in kept):
        raise ValueError(f"anchor {anchor!r} is missing or was dropped "
                         f"(floored {floored}, ceilinged {ceilinged})")
    kept_names = [players[i] for i in kept]
    log_p = fit_bt(S[np.ix_(kept, kept)], iterations)
    elo = ELO_PER_LOG * (log_p - log_p[kept_names.index(anchor)])
    return EloResult(dict(zip(kept_names, map(float, elo))), floored, ceilinged)


@dataclass
class BootstrapResult:
    intervals: "dict[str, tuple[float, float]]"  # 2.5/97.5 percentile
    rated_in: "dict[str, int]"  # resamples in which the player was rated
    failed: int  # resamples skipped: anchor dropped or Ford violated


def bootstrap(counts, anchor: str, B: int = 1000, seed: int = 0,
              iterations: int = 2000) -> BootstrapResult:
    """Seeded stratified bootstrap: every (pair, colour) cell is resampled
    as its own multinomial, then refitted with the full `rate` pipeline —
    including the drop and Ford guards, whose failures are counted in
    `failed` (or, for a single dropped non-anchor player, in a reduced
    `rated_in`) instead of ever fitting anyway."""
    rng = np.random.default_rng(seed)
    samples: "dict[str, list[float]]" = {}
    failed = 0
    for _ in range(B):
        resampled = {
            pair: tuple(rng.multinomial(sum(cell), np.asarray(cell) / sum(cell)))
            for pair, cell in counts.items()
        }
        try:
            result = rate(resampled, anchor, iterations)
        except ValueError:
            failed += 1
            continue
        for name, value in result.ratings.items():
            samples.setdefault(name, []).append(value)
    intervals = {
        name: (float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5)))
        for name, values in samples.items()
    }
    return BootstrapResult(intervals, {n: len(v) for n, v in samples.items()}, failed)
