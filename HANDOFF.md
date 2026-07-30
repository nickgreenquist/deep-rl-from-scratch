# Handoff — written 2026-07-29 on explicit request; fold in, then restore the stub

Per CLAUDE.md § Plan and status: read this, fold anything durable into
PLAN.md / SESSION_LOGS.md (nothing should need it — the log is current),
then restore this file to its empty stub.

## State
- Clean and PUSHED at `32ae257`. 287 tests green, all batteries passing.
- **Phase 5 MILESTONE 1 PASSED**: 0.663 ± 0.029 vs MaxBasePowerPlayer,
  1000 battles, final checkpoint (`runs/showdown_maxbp_s0/`, headline
  artifact `final_eval_maxbp_1000.json`). Same checkpoint vs heuristics:
  0.262 ± 0.039 — milestone 2 is an ENCODER problem, as pre-registered.

## Read, in order
1. `CLAUDE.md` — hard rules, handed-over-command format, the three-doc
   scheme (PLAN.md spec / SESSION_LOGS.md history / this file ephemeral).
2. `PLAN.md` § Phase 5 — capstone spec, milestone ladder, hardware note.
3. `SESSION_LOGS.md`, the five 2026-07-29 entries (grep `2026-07-29`):
   Phase 5 opening, throughput (a)+(b), throughput (c), the three-review
   collection-fork decision + env fixes, milestone-1 result. Read these
   before proposing anything — the collection architecture is SETTLED
   (SyncVectorEnv path; lockstep facade parked behind measurement (e);
   seam-native collector rejected for milestone scope with reasons).

## Next (discussed, not committed — plan with the maintainer)
1. **Encoder design** — replace the 10-dim placeholder in
   `rl/envs/showdown.py` (`embed_battle`). The design work of the phase:
   what the acting player can observe in Gen 1 (HP fractions, status,
   boosts, revealed moves/PP, species — small embedding tables at most,
   per the hardware note). This is a maintainer design session, not a
   solo implementation.
2. **Measurement (d)** while at it: forward-pass share at the real
   encoder, batch-1 vs batched (`scripts/showdown_throughput.py` has
   (a)-(c); the seam + `rl/collect.py` exist for the probe).
3. Milestone-2 run vs `SimpleHeuristicsPlayer` (config exists as a
   pattern: `configs/showdown_maxbp_ppo.yaml`).

## Operational
- Server: `cd showdown && node pokemon-showdown start --no-security`
  (localhost:8000; setup script `scripts/setup_showdown.sh` if re-cloning).
  Live-server tests self-skip when it's down.
- Training runs: maintainer's terminal, `WANDB_MODE=offline`, CLEAN tree
  (untracked files flip `git_dirty` — CLAUDE.md rule).
- The 2M-run wall is ~34 min at ~1k steps/s on the current path; don't
  optimize collection further without a binding wall (see the fork entry).
