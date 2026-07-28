# Handoff — Phase 4: chunk 3 COMPLETE, chunk 4 next

Instructions to a fresh session of Claude Code after a context clear. Trust
`PLAN.md` § Phase 4 over anything here if they disagree — it is the locked
spec and the session log holds every finding. Do not re-litigate settled
decisions; if one looks wrong, say so and ask.

## State
- Clean at `c4464c6`, **fully pushed** (origin in sync). **254 tests green.**
- **Chunk 3 is COMPLETE** — solver (bitboard + chapter-8 null-window
  driver), Pons downloader/validator, `AlphaBetaOpponent` anchors
  (`alphabeta2`/`alphabeta4`), `rl/selfplay/elo.py` (BT-MM + Ford guard +
  stratified bootstrap + intransitivity detector w/ null band),
  `play_game` in `opponents.py`, `scripts/tournament.py`. Mutation
  batteries committed and clean: `chunk3_solver.py` (17/17 real caught,
  5/5 controls survive), `chunk3_elo.py` (9/9 + 2/2).
- **Pons validation CLOSED at 4,400/4,400, zero mismatches** (End-Easy,
  Middle-Easy, Middle-Medium, Begin-Easy full; Begin-Medium 400-position
  partial). Begin-Medium/Hard are measured-intractable at this solver
  level — the boundary the spec itself drew; chapters 9–13 are the named
  lever ONLY if a future phase needs them. Do not reopen.
- **Both forks are SETTLED and the campaign config is LOCKED:**
  `gae_lambda` 0.95 (lam1 final 60 Elo behind pool, CIs disjoint) and
  `kernel_size` 3 (3 seeds/arm: k3 finals {−124.5, −189.8, −131.7} vs k4
  {−200.8, −171.7, −231.8}; k4's vs-heuristic edge was anchor
  specialization). The four 2026-07-28 session-log entries are the record.
- **Seven trained ladders on disk, each with a `tournament.json`**
  (500 games/pair, B=1000): `runs/connect4_pool_{s0,s1,s2}`,
  `connect4_pool_k4_{s0,s1,s2}`, `connect4_pool_lam1_s0`. The pool trio
  IS the pool side of the chunk-4 campaign (locked config, seeds 0/1/2).
  The **naive arm (`pool_size: 1`) has never been trained** — its 3 seeds
  are the only campaign training left (`configs/connect4_naive.yaml`,
  `--seed`/`--run-name` overrides exist).
- Cross-cutting findings already established (do not rediscover): the
  cycling detector fires in ALL 7 tournaments (fractions 0.035–0.108 vs
  null-band tops 0.002–0.013); best rung ≠ final in 4/7 runs (late
  regression is the norm; k4_s2 −47 Elo from its 1.2M peak); within-arm
  seed spread ~65 Elo; AlphaStar min-winrate proxy from the tournament
  counts: pool {0.610, 0.458, 0.609}, k4 {0.481, 0.526, 0.567}, lam1
  0.487. Chunk-2's coverage collapse / vs-random decline / entropy
  collapse findings and the three chunk-4 probe levers (entropy floor,
  PFSP, fixed-opponent mixing) stand as recorded.

## Read, in order
1. `CLAUDE.md` — note the handed-over-command format: one command per
   fenced block, ONE line, no inline comments, wrapped in `<command>` /
   `</command>` sentinel lines OUTSIDE the fence; `WANDB_MODE=offline` on
   training commands.
2. `PLAN.md` § Phase 4: the chunk-4 checkbox, the forgetting-demonstration
   and solver/metrics paragraphs (they define the Pons agent metrics:
   value sign accuracy + Brier over decisive positions per set, MAE
   rejected; policy metrics over the solver-exhausted subset, coverage
   always reported), and the 2026-07-27/28 session-log entries.
3. Code: `rl/selfplay/{solver,elo,opponents,pool}.py`,
   `scripts/{tournament,pons_benchmark,score_ladder,mutate}.py`,
   `scripts/mutations/chunk3_*.py`.

## Task: chunk 4
1. **PENDING GO-AHEAD (asked, never answered — ask again before building):**
   commit the two mechanism diagnostics as tooling, rebuilt from the
   methods recorded in the 2026-07-27 diagnostics entry:
   `scripts/coverage_probe.py` (distinct games + mean common prefix vs
   latest snapshot, random-vs-random control; reuse `play_game`, which
   may need an optional start-position param) and
   `scripts/value_mse_probe.py` (value MSE over self-play/random/heuristic
   state distributions, targets = mean of K=8 mirror-self-play
   continuations). Then run both on the 7 finals (the confirming-seed
   "re-log both" note is still open).
2. Naive arm: 3 training runs (maintainer's terminal, ~3–5 min each),
   then tournament each ladder (~1 min each, in-session is fine).
3. Forgetting demonstration per the locked spec: AlphaStar proxy PRIMARY
   (promote the session-log scratch snippet to committed tooling),
   regression rate SECONDARY and only against its simulated null band.
4. Pons agent metrics script (the last instrument): value-head sign
   accuracy + Brier per set over decisive positions; blunder rate /
   optimal-move agreement / score regret over the solver-exhausted subset
   (End/Middle sets; blunder rate needs only child SIGNS so a weak solve
   suffices there), coverage always reported.
5. Gitignored `runs/connect4_campaign.sh`, figure, README section.
   Open decision for the maintainer: probe levers before or as campaign
   arms.

## Gotchas not in the plan
- `deep-rl` conda env only: `/opt/anaconda3/envs/deep-rl/bin/pytest tests/`.
- **Division of labor (clarified this session):** training and anything
  multi-minute goes to the maintainer's terminal; the ~1-minute class
  (tournaments, probes) may run in-session — it was measured at full
  speed there. When in doubt, hand it over.
- Never edit the tree (even uncommitted) while the maintainer may be
  LAUNCHING a training run — launches stamp `git_dirty`. Never run
  mutation batteries while any maintainer process might import mutated
  source. Coordinate via the conversation.
- zsh traps in handed-over commands: no `echo =====` (equals expansion),
  no inline `#`, no multi-line blocks.
- `AgentOpponent` freeze-at-install contract: whoever installs calls
  `freeze()` (tournament.py does; copy that pattern).
- Tournament JSONs contain full pairwise counts — the AlphaStar proxy and
  any future pairwise analysis come free from them; `best_checkpoint.pt`
  is excluded from tournaments by design (selection bias).
- The mutation batteries' `old` strings match current source exactly; a
  refactor that breaks one is a prompt to update the spec, not delete it.

## First actions
1. Confirm with the maintainer which chunk-4 item to start with (the
   diagnostics go-ahead is the natural first ask).
2. Read the docs and code above.
3. State which files you'll create/change and why; wait for a go-ahead.
4. Small commits, pytest green per step, mutation-test each guard.
