# Handoff — Phase 5 started; env plumbing DONE, collection loop next

Instructions to a fresh session of Claude Code after a context clear. Trust
`PLAN.md` over anything here if they disagree — it is the locked spec and
the session log holds every finding. Do not re-litigate settled decisions;
if one looks wrong, say so and ask.

## State
- Origin is in sync through `8f82859`; commits after that (Phase 5 step 1 +
  this doc pass) are local — offer a push, explicit go-ahead only.
- 278 tests green. **Phase 5 step 1 (env plumbing) is DONE** (see the
  2026-07-29 "Phase 5 opens" session-log entry): `poke-env==0.15.0` pinned,
  local Showdown server at pinned sha `59da482` via
  `scripts/setup_showdown.sh` (gitignored `showdown/`), and
  `rl/envs/showdown.py` — `ShowdownSingles` + the `ShowdownEnv` adapter
  (mask lifted obs→`info["action_mask"]`, `info["outcome"]` from
  `battle.won`/`lost` never term/trunc, terminal-only reward = outcome),
  registered as `Showdown-v0` through `make_env`.
- Server must be RUNNING for the integration test (it self-skips when
  :8000 is down) and any battles:
  `cd showdown && node pokemon-showdown start --no-security`.
- `embed_battle` is a 10-dim PLACEHOLDER encoder — plumbing only; the
  encoder-design step replaces it after the throughput measurements.
- Measured baseline: ~1,100 agent-steps/s, one env, one process, no policy
  net, mask-random policy. Mask-random loses 0/10 to `max_power` and
  `heuristics` (milestone-1 headroom is real).

## Read, in order
1. `CLAUDE.md` — hard rules, handed-over-command format, >5-minute rule,
   CPU-first hardware note.
2. `PLAN.md` § Phase 5 — especially the four pre-registered throughput
   measurements (a)–(d) and the single-inference-seam collection-loop
   contract (battle coroutines submit observations to one seam; batch-1 /
   micro-batched / lockstep-vector stay config choices).
3. The 2026-07-29 session-log entries (Phase 4 wrap + Phase 5 opening).

## Next steps (order NOT settled — decide with the maintainer)
1. Collection-loop seam + throughput measurements (a)–(d): they gate every
   vectorization/provisioning decision and (d) prices the encoder budget.
   Measurement (a)'s latency breakdown wants per-turn timing hooks.
2. Milestone-1 train wiring: a `configs/showdown_*.yaml`, PPO through
   `python -m rl.train`, eval vs `max_power` (`eval/win_rate` already works
   — `info["outcome"]` is emitted). Blocked-ish on 1: `SubprocVecEnv`-vs-
   asyncio is exactly what the measurements decide.
3. Encoder design (replace the placeholder) — after 1 prices it.
4. Self-play later: the pool/`Opponent` protocol transfers, but the
   opponent enters via `SingleAgentWrapper(env, opponent)` and needs a
   Player adapter driving seat 2 from our policy (not written yet).

## Gotchas not in the plan
- `deep-rl` conda env only: `/opt/anaconda3/envs/deep-rl/bin/pytest tests/`.
- Multi-minute runs go to the maintainer's terminal (`WANDB_MODE=offline`).
- Never edit the tree (even untracked files) while the maintainer may be
  LAUNCHING a run — `git_dirty` stamps flip on untracked files (measured).
- zsh traps in handed-over AND in-session commands: no `echo ===` (glob
  error), no inline `#`, one command per block, `<command>` sentinels.
- poke-env: actions must be `np.int64` (`action_to_order` calls
  `action.item()`); `SingleAgentWrapper` lives at
  `poke_env.environment.SingleAgentWrapper` (not `.player.`); opponent
  Players get `start_listening=False` (their `choose_move` is called
  directly — no websocket needed).
- Showdown config: `exports.repl = false` appended (REPL sockets EINVAL-
  crash on macOS + Node 25); setup script handles it on re-clone.
- Battles are NOT seed-reproducible (server rolls teams/damage); variance
  is handled by battle count (≥1000/matchup), not seeds.
- The mutation batteries' `old` strings match current source exactly; a
  refactor that breaks one is a prompt to update the spec, not delete it.

## First actions
1. Ask the maintainer: collection-loop seam + measurements first, or
   milestone-1 train wiring first (see Next steps).
2. Check the server is up before running the suite if you want the
   integration test to actually run.
3. State which files you'll create/change and why; wait for a go-ahead.
4. Small commits, pytest green per step, mutation-test each new guard.
