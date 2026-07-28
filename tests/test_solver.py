"""Solver tests (Phase 4 chunk 3): bitboard vs Connect4Board, search vs
brute force.

The two differential halves deliberately trust different things. The
bitboard half trusts `Connect4Board` — chunk 1 validated it against the
open_spiel oracle, so ply-by-ply agreement chains the bitboard to that
oracle. The search half (added with the solver) trusts `brute_force`, a
negamax with nothing in it that can be subtly wrong — no pruning, no
transposition table, no ordering.

The named fixtures ride through the bitboard too: the random playouts are
the fuzz, the fixtures are the shapes chunk 1 learned to pin by hand (edge
columns, both diagonals, the win on the 42nd disc) — numpy's wrap bug there
is the same failure class as a bitboard shift crossing a column boundary,
which is exactly what the sentinel row exists to stop.
"""

import numpy as np
import pytest

from rl.envs.connect4 import COLS, Connect4Board
from rl.selfplay.solver import Bitboard
from tests.test_connect4 import DRAW_42, WIN_FIXTURES, WIN_ON_42

# ---------------------------------------------------------------- bitboard


def replay(cols):
    """Play a column sequence through BOTH representations, comparing at
    every ply; return the final (board, bitboard, last_move_won)."""
    board, bb = Connect4Board(), Bitboard()
    won = False
    for col in cols:
        assert not won, "sequence continues past a win"
        assert [c for c in range(COLS) if bb.can_play(c)] == list(
            np.flatnonzero(board.legal_mask())
        )
        predicted = bb.is_winning_move(col)
        won = board.drop(col)
        assert predicted == won, f"win flag disagrees on column {col}"
        bb = bb.play(col)
        assert Bitboard.from_board(board) == bb, f"state diverged after column {col}"
    return board, bb, won


def test_random_playout_differential():
    """Fuzz: full random games, every ply compared — legal columns, the win
    flag, and the (current, mask, moves) triple via from_board."""
    rng = np.random.default_rng(0)
    for _ in range(40):
        board, bb = Connect4Board(), Bitboard()
        while True:
            assert [c for c in range(COLS) if bb.can_play(c)] == list(
                np.flatnonzero(board.legal_mask())
            )
            col = int(rng.choice(np.flatnonzero(board.legal_mask())))
            predicted = bb.is_winning_move(col)
            won = board.drop(col)
            assert predicted == won
            bb = bb.play(col)
            assert Bitboard.from_board(board) == bb
            if won or board.full():
                break


@pytest.mark.parametrize("name", sorted(WIN_FIXTURES))
def test_win_fixtures_through_the_bitboard(name):
    """The hand-pinned win shapes — edge columns and both diagonals — must
    win on their last move and never earlier, in the bitboard's own win
    check (replay asserts the flag at every ply)."""
    _, _, won = replay(WIN_FIXTURES[name])
    assert won


def test_draw_and_win_on_42_through_the_bitboard():
    board, bb, won = replay(DRAW_42)
    assert not won and bb.moves == 42
    _, bb, won = replay(WIN_ON_42)
    assert won and bb.moves == 42


def test_to_board_round_trips():
    """to_board is the inverse of the playout: array, moves counter, and
    a second from_board all agree mid-game (canonical perspective intact
    after an odd number of plies)."""
    for cols in (WIN_FIXTURES["diagonal_up_right"][:9], DRAW_42[:17]):
        board, bb, _ = replay(cols)
        back = bb.to_board()
        assert np.array_equal(back.board, board.board)
        assert back.moves == len(cols)
        assert Bitboard.from_board(back) == bb


def test_keys_are_unique_across_distinct_positions():
    """`current + mask` must collide only for identical positions: walk a
    few hundred random positions and assert the key map is injective."""
    rng = np.random.default_rng(1)
    seen: dict[int, tuple[int, int]] = {}
    for _ in range(30):
        board, bb = Connect4Board(), Bitboard()
        while True:
            state = seen.setdefault(bb.key(), (bb.current, bb.mask))
            assert state == (bb.current, bb.mask), "key collision"
            col = int(rng.choice(np.flatnonzero(board.legal_mask())))
            if board.drop(col):
                break
            bb = bb.play(col)
            if board.full():
                break
    assert len(seen) > 300
