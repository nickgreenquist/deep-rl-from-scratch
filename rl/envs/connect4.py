"""Connect 4: the Phase 4 self-play on-ramp env.

Two objects, deliberately separated:

- `Connect4Board` — pure NumPy game logic, no gymnasium. Testable against the
  `open_spiel` oracle and (Phase 4 chunk 3) against a negamax solver without
  dragging in a Box/Discrete space. NumPy is the right representation *here*
  because the observation is a pair of 6x7 planes; it is the wrong one for
  search, so the chunk-3 solver uses a Python-int bitboard instead and the
  two cross-check each other (measured: NumPy 351k nodes/s vs the bitboard's
  894k — NumPy is the slowest of the three representations tried).
- `Connect4Env` — a learner-centric single-agent view of a two-player game,
  with the frozen opponent living inside. See its docstring.

Two conventions are stated loudly because getting either wrong produces tests
that pass while the game is broken:

- **Row 0 is the BOTTOM row.** Discs stack upward from row 0. An inverted
  convention makes every diagonal test pass while the real diagonals are
  mirrored, because the board is symmetric under a vertical flip but the
  diagonals are not.
- **The board is CANONICAL: +1 is always the player to move**, -1 the other
  player, 0 empty. `drop()` negates the whole board after placing a disc, so
  the mover is +1 again for the next ply. Nothing in this file needs to know
  who "red" or "yellow" is, which is exactly what makes the observation
  egocentric for free — and what makes the perspective bugs in
  `Connect4Env` possible, so read that docstring too.
"""

import numpy as np

ROWS, COLS = 6, 7
CONNECT = 4
CELLS = ROWS * COLS
# Half-directions through a placed disc: horizontal, vertical, and the two
# diagonals. Each is scanned both ways, so four entries cover all eight rays.
_DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))


class Connect4Board:
    """Canonical Connect 4 position: `board[row, col]`, row 0 = bottom,
    +1 = player to move.

    Height is derived from the board rather than cached: discs stack with no
    gaps, so a column's height is just its nonzero count, and a derived value
    cannot desynchronize from the position it describes.
    """

    def __init__(self, board: np.ndarray | None = None, moves: int = 0):
        self.board = np.zeros((ROWS, COLS), dtype=np.int8) if board is None else board
        self.moves = moves

    def copy(self) -> "Connect4Board":
        return Connect4Board(self.board.copy(), self.moves)

    def legal_mask(self) -> np.ndarray:
        """bool [COLS], True = playable. A column is playable iff its TOP cell
        is empty. Freshly allocated: callers put this in `info` and the env
        contract forbids handing out views of internal state."""
        return self.board[ROWS - 1] == 0

    def height(self, col: int) -> int:
        return int(np.count_nonzero(self.board[:, col]))

    def full(self) -> bool:
        return self.moves == CELLS

    def drop(self, col: int) -> bool:
        """Play the mover's disc in `col`; return True if that move won.

        The win check runs BEFORE the negation, against +1 — the player who
        just moved. After the negation the mover is +1 again, so the returned
        flag is the only record of who won, and `Connect4Env` must consume it
        immediately.
        """
        row = self.height(col)
        if row >= ROWS:
            raise ValueError(f"column {col} is full")
        self.board[row, col] = 1
        won = self._is_win(row, col)
        self.board *= -1
        self.moves += 1
        return won

    def _is_win(self, row: int, col: int) -> bool:
        """Does the +1 disc at (row, col) complete a line of CONNECT?"""
        for dr, dc in _DIRECTIONS:
            count = 1
            for sign in (1, -1):
                r, c = row + sign * dr, col + sign * dc
                while 0 <= r < ROWS and 0 <= c < COLS and self.board[r, c] == 1:
                    count += 1
                    r += sign * dr
                    c += sign * dc
            if count >= CONNECT:
                return True
        return False

    def planes(self) -> np.ndarray:
        """Egocentric observation: bool [2, ROWS, COLS] — plane 0 the mover's
        discs, plane 1 the opponent's. Freshly allocated by np.stack (the env
        contract: never a view of `self.board`).

        No turn-indicator plane. First-player identity is recoverable from the
        plane sums (the mover has played either as many discs as the opponent
        or one fewer), and no reference carries one: alpha-zero-general uses 1
        canonical plane, PettingZoo 2, open_spiel 3 absolute.
        """
        return np.stack([self.board == 1, self.board == -1])

    def render(self) -> str:
        """Human-readable, printed TOP row first (open_spiel's `ToString` does
        the same). Debug aid only — never compare renderings in a test, and
        never use one to check the row convention: that is precisely the
        comparison that hides a flipped board.
        """
        glyphs = {1: "x", -1: "o", 0: "."}
        return "\n".join(
            "".join(glyphs[int(v)] for v in self.board[row]) for row in reversed(range(ROWS))
        )
