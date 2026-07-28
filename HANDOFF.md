# Handoff — Phase 4 chunk 3: solver, anchors, tournament, Elo

Instructions to a fresh session of Claude Code after a context clear. Trust
`PLAN.md` § Phase 4 over anything here if they disagree — it is the locked
spec and the session log holds every finding. Do not re-litigate locked
decisions; if one looks wrong, say so and ask.

## State
- Clean at `fd7e880`, **10 commits UNPUSHED** (origin is at `9717797`) — ask
  the maintainer before pushing. **208 tests green.**
- **Chunks 1 AND 2 are COMPLETE**, including chunk 2's pathfinder and the
  mechanism diagnostics. The five `2026-07-27` session-log entries in
  PLAN.md are the record: chunk-1 implementation, chunk-1 gate, chunk-2
  implementation, chunk-2 pathfinder, mechanism diagnostics.
- Headline findings you must not rediscover from scratch: the self-play
  agents are coverage-collapsed (probe arms replay 8–11 distinct games out
  of 200; critic 10–30× worse one step off its own distribution), the
  pre-registered vs-random ≥0.95 recovery FAILED (0.87–0.90, declining),
  entropy collapsed by 2M. All pre-registered escapes — findings, not bugs.
  Three chunk-4 probe levers are named in falsifiable form in PLAN.md.

## Read, in order
1. `CLAUDE.md` (note the command-handover format rule — one command per
   fenced block, no inline comments, `WANDB_MODE=offline` in run commands).
2. `PLAN.md` § Phase 4 **in full** — the solver/Elo paragraphs ARE the
   chunk-3 spec (TT `(value, EXACT|LOWER|UPPER)` flags, Python-int bitboard,
   chapter-8 null-window mandatory, Pons URLs plain-http into gitignored
   `data/` never committed, Bradley-Terry by MM with Ford check and
   stratified bootstrap). Then the five 2026-07-27 session-log entries.
3. Code: `rl/selfplay/pool.py` (AgentOpponent is the checkpoint→opponent
   adapter the tournament needs — deepcopy inside the constructor, own
   `torch.Generator` per instance for matchup replay), `rl/selfplay/
   opponents.py` (the `OPPONENTS` registry chunk 3 extends with
   `alphabeta2`/`alphabeta4`), `rl/envs/connect4.py` docstrings,
   `scripts/score_ladder.py`, `scripts/mutate.py` + `scripts/mutations/
   chunk2_pool.py` (the battery pattern to repeat for chunk 3).

## Task: chunk 3, per the maintainer-approved file plan
1. `rl/selfplay/solver.py` — bitboard negamax + alpha-beta (centre-first,
   bounded TT with flags) PLUS a brute-force no-pruning negamax as the
   differential oracle; tests cross-check bitboard ↔ `Connect4Board` and
   solver ↔ brute force on random positions.
2. Chapter-8 iterative deepening on top; Pons downloader; validation runs
   (Begin sets ~12 min) go to the maintainer's terminal.
3. `AlphaBetaOpponent(depth)` with uniform random tie-breaking, registered.
4. `rl/selfplay/elo.py` — BT-MLE by MM, Ford check before every fit,
   perfect scorers dropped with floor/ceiling, seeded stratified bootstrap
   B=1000, iteration-stability test at 200/2k/20k.
5. `scripts/tournament.py` — ladder + four anchors, first player alternated
   exactly, both sides stochastic; the ~40-min campaign runs in the
   maintainer's terminal.

## Gotchas not in the plan
- `deep-rl` conda env only: `/opt/anaconda3/envs/deep-rl/bin/pytest tests/`.
- Ladders to feed the tournament are ON DISK: `runs/connect4_pool_s0`,
  `runs/connect4_pool_lam1_s0`, `runs/connect4_pool_k4_s0` — 10 rungs each
  plus `history.csv` and `ladder_scores.json`.
- The mutation battery temporarily mutates tracked source: never run it
  while the maintainer might be launching a run, or the run stamps
  `git_dirty: true` (the launch blocker).
- The two mechanism-diagnostic scripts (off-distribution value MSE,
  distinct-game coverage) were session scratch and are GONE; their methods
  and controls are recorded in the diagnostics session-log entry. The
  kernel-fork confirmation seeds (k3 vs k4, 2 seeds, before chunk 4 locks)
  must re-log both — rebuild from the recorded method (~30 lines each) or
  fold them into committed tooling then.
- `open_spiel` stays dev-only (`pyspiel.load_game`, never
  `open_spiel.python.*`); the AST pin test enforces it.
- Solver perf notes are already measured (PLAN.md): Python-int bitboard
  894k nodes/s, NumPy is the SLOWEST representation; never subsample Pons
  sets to estimate runtime.

## First actions
1. Confirm chunk 3 step 1 is the scope for this session.
2. Read the docs and code above.
3. State which files you'll create/change and why; wait for a go-ahead.
4. Small commits, pytest green per step, mutation-test each guard.
