"""Connect 4 board and env fixtures.

These are the phase's GATE (PLAN.md chunk 1): deterministic per-site probes,
not a training run. The rejected alternative — "beats random >=90%" — caught
0 of 4 seeded defects, because `RandomOpponent.move` ignores the observation
entirely (so a wrong opponent perspective is a provable no-op) and both
swapped learner planes and a dropped epoch mask scored *higher* than clean.

Fixtures are hand-pinned semantics; `tests/test_connect4_oracle.py` fuzzes
against open_spiel for the discrepancies nobody thought to name. The two are
complementary and neither replaces the other: the mask is all-True at 63.8%
of decision points and single-legal-column positions are 0.53%, so fuzz alone
is flaky on exactly the cases that matter.
"""

import numpy as np
import pytest

from rl.envs.connect4 import COLS, ROWS, Connect4Board

# Win fixtures: column sequences, alternating players, the LAST move winning.
# Each is cross-checked against open_spiel in the oracle test file.
WIN_FIXTURES = {
    "vertical_col0": [0, 1, 0, 1, 0, 1, 0],
    "vertical_col6_edge": [6, 5, 6, 5, 6, 5, 6],
    "horizontal_row0_right_edge": [3, 0, 4, 1, 5, 2, 6],
    "diagonal_up_right": [0, 1, 1, 2, 6, 2, 2, 3, 6, 3, 6, 3, 3],
    "diagonal_up_left": [6, 5, 5, 4, 0, 4, 4, 3, 0, 3, 0, 3, 3],
}

# Found by random search (scratch), not by hand: 809 games to the first draw,
# 2603 to the first win on the 42nd disc. Draws are 0.27% of random games, so
# this branch is otherwise never exercised — PLAN.md lists it as a named
# degeneracy of this env.
DRAW_42 = [5, 5, 5, 0, 5, 4, 4, 6, 6, 3, 5, 1, 3, 5, 2, 1, 2, 3, 0, 4, 1,
           4, 3, 0, 2, 2, 2, 1, 0, 0, 0, 2, 1, 6, 1, 3, 4, 4, 3, 6, 6, 6]
WIN_ON_42 = [3, 4, 1, 0, 1, 1, 3, 3, 2, 2, 2, 2, 4, 4, 6, 5, 0, 3, 3, 6, 4,
             2, 4, 4, 1, 2, 0, 0, 5, 3, 6, 0, 1, 6, 0, 6, 1, 6, 5, 5, 5, 5]


def play(cols):
    """Play a column sequence; return (board, won_on_last_move). Raises if a
    win happens before the last move — a fixture that wins early is silently
    testing a different position than its name claims."""
    board = Connect4Board()
    won = False
    for i, col in enumerate(cols):
        assert not won, f"fixture won at move {i}, before its last move"
        won = board.drop(col)
    return board, won


def test_row_zero_is_the_bottom():
    """The convention pin. An inverted board passes every vertical and
    horizontal test — only the diagonals and this assertion catch it."""
    board = Connect4Board()
    board.drop(3)
    filled = np.argwhere(board.board != 0)
    assert filled.tolist() == [[0, 3]], "first disc must land on row 0"
    board.drop(3)
    assert board.height(3) == 2
    assert sorted(np.argwhere(board.board != 0).tolist()) == [[0, 3], [1, 3]]


def test_empty_board_state():
    board = Connect4Board()
    mask = board.legal_mask()
    assert mask.shape == (COLS,) and mask.dtype == np.bool_
    assert mask.all()
    assert not board.full()
    assert board.moves == 0
    assert all(board.height(c) == 0 for c in range(COLS))


def test_full_column_mask():
    board = Connect4Board()
    for _ in range(ROWS):  # alternating players: 6 discs, no line of 4
        board.drop(0)
    mask = board.legal_mask()
    assert not mask[0]
    assert mask[1:].all()
    assert not board.full()  # a full COLUMN is not a full board
    assert board.height(0) == ROWS


def test_single_legal_column():
    """0.53% of real decision points (~1 in 190) — fuzz reaches it rarely
    enough to be flaky, which is why it is pinned by hand."""
    board = Connect4Board()
    for col in range(COLS - 1):
        for _ in range(ROWS):
            board.drop(col)
    mask = board.legal_mask()
    assert mask.tolist() == [False] * (COLS - 1) + [True]
    assert np.count_nonzero(mask) == 1


def test_drop_on_full_column_raises():
    board = Connect4Board()
    for _ in range(ROWS):
        board.drop(0)
    with pytest.raises(ValueError, match="full"):
        board.drop(0)


@pytest.mark.parametrize("name", sorted(WIN_FIXTURES))
def test_win_fixtures_win_exactly_on_the_last_move(name):
    board, won = play(WIN_FIXTURES[name])
    assert won, f"{name}: last move should win"
    assert not board.full()


def test_diagonals_win_on_the_intended_diagonal():
    """A 'diagonal' fixture that actually won horizontally would leave the
    diagonal code path untested while staying green. Pin the cells.

    After the winning drop the board has been negated, so the winner's discs
    are -1 (drop() reports the win; the board moves on).
    """
    up_right, _ = play(WIN_FIXTURES["diagonal_up_right"])
    assert [int(up_right.board[i, i]) for i in range(4)] == [-1] * 4
    up_left, _ = play(WIN_FIXTURES["diagonal_up_left"])
    assert [int(up_left.board[i, COLS - 1 - i]) for i in range(4)] == [-1] * 4


def test_draw_on_a_full_board():
    board, won = play(DRAW_42)
    assert not won
    assert board.full()
    assert board.moves == ROWS * COLS
    assert not board.legal_mask().any()


def test_win_on_the_final_disc():
    """Win-before-full ordering is load-bearing: the 42nd disc can complete a
    line, so an env checking "board full -> draw" first silently converts this
    win into a draw. The board reports both true at once; the env's ORDER is
    what resolves it (see test_env_win_on_the_final_disc_is_a_win)."""
    board, won = play(WIN_ON_42)
    assert won
    assert board.full()


# Positions where a ray running off the board would WRAP. numpy's negative
# indices are silent: a scan off column 0 reads column 6, a scan off row 0
# reads row 5. Both positions below have no line of four, but report one if
# either low bound is dropped from the ray guard. Found by search against a
# deliberately mutated win check (scratch), not by hand.
NO_WIN_COL_WRAP = [4, 1, 5, 2, 6, 3, 0]  # discs at cols 4,5,6 then col 0
NO_WIN_ROW_WRAP = [2, 0, 3, 0, 6, 3, 3, 3, 0, 2, 3, 1, 3, 0, 1]


@pytest.mark.parametrize(
    "name,cols",
    [("col_wrap", NO_WIN_COL_WRAP), ("row_wrap", NO_WIN_ROW_WRAP)],
)
def test_rays_do_not_wrap_around_the_board_edges(name, cols):
    """The phantom-win case. Three same-colour discs on one edge plus one on
    the opposite edge is NOT a line of four — but numpy indexing says it is
    unless both low bounds are guarded."""
    board, won = play(cols)
    assert not won, f"{name}: ray wrapped around the board edge"


def test_planes_are_egocentric_and_freshly_allocated():
    board = Connect4Board()
    board.drop(3)  # mover A plays; board flips, so B is now the mover
    planes = board.planes()
    assert planes.shape == (2, ROWS, COLS) and planes.dtype == np.bool_
    # Plane 0 is the MOVER's discs. B has none yet; A's single disc is plane 1.
    assert planes[0].sum() == 0
    assert planes[1].sum() == 1 and planes[1][0, 3]
    # Never a view of internal state: the env contract forbids handing out
    # aliases of the board, and np.stack allocates.
    assert not np.shares_memory(planes, board.board)
    board.drop(3)
    assert planes[1].sum() == 1, "planes must not track later mutations"


def test_board_stays_canonical_after_every_drop():
    """+1 is always the player to move. After each drop the counts must
    satisfy: mover has played as many discs as the opponent, or one fewer."""
    board = Connect4Board()
    rng = np.random.default_rng(0)
    for _ in range(20):
        col = int(rng.choice(np.flatnonzero(board.legal_mask())))
        if board.drop(col):
            break
        mover = int((board.board == 1).sum())
        other = int((board.board == -1).sum())
        assert other - mover in (0, 1), f"non-canonical: {mover} vs {other}"


def test_copy_is_independent():
    board = Connect4Board()
    board.drop(3)
    clone = board.copy()
    clone.drop(3)
    assert board.moves == 1 and clone.moves == 2
    assert not np.shares_memory(board.board, clone.board)
