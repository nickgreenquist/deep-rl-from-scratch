# Handoff — PHASE 4 COMPLETE; Phase 5 (capstone) next

Instructions to a fresh session of Claude Code after a context clear. Trust
`PLAN.md` over anything here if they disagree — it is the locked spec and
the session log holds every finding. Do not re-litigate settled decisions;
if one looks wrong, say so and ask.

## State
- Clean, 271 tests green, all mutation batteries passing (every real
  mutation caught, every control surviving). Check `git log` for the tip;
  ask before pushing anything.
- **Phase 4 is COMPLETE** (chunks 1–4 all ticked; see the 2026-07-28/29
  session-log entries). The campaign, three probe-lever arms, the
  supervised-on-solver-labels diagnostic, figure
  (`assets/connect4_forgetting.png`), README section, and the gitignored
  `runs/connect4_campaign.sh` reproduction record are all landed.
- **The three durable Phase-4 findings** (do not rediscover): naive
  self-play forgets and the 20-snapshot pool prevents it (AlphaStar proxy,
  no overlap between arms); intransitivity is structural at this strength
  band (cycling detector fired in 19/19 tournaments); and the tactical
  ceiling is the VISITED STATE DISTRIBUTION, not the architecture — the
  same net reaches 0.855 optimal-move agreement supervised in-distribution
  vs 0.29–0.61 for every RL variant on the Pons sets. Lever verdicts:
  fixed-opponent mixing was the only lever to hit its target (vs-random
  decline eliminated on 2/3 seeds); entropy floor buys coverage with
  extreme seed variance; PFSP trades robustness-to-past for current
  strength.
- 17 finished run dirs under `runs/connect4_*` and `runs/supervised_*`,
  each with tournament/forgetting/coverage/value-MSE/Pons JSONs where
  applicable.

## Read, in order
1. `CLAUDE.md` — hard rules (no RL libraries; the `open_spiel` dev-only
   carve-out), the handed-over-command format (one command per fenced
   block, ONE line, no inline comments, `<command>`/`</command>` sentinels
   OUTSIDE the fence), the >5-minute-runs-go-to-the-maintainer rule, and
   the CPU-first capstone hardware note (revised 2026-07-28).
2. `PLAN.md` § Phase 5 — the capstone spec: poke-env `SinglesEnv` (the old
   `Gen*EnvSinglePlayer` API is DEAD, removed in 0.8.4), action mask
   arrives in the OBSERVATION and needs one adapter wrapper lifting it
   into `info`, `truncated` is load-bearing (forfeits/ties/timer), the
   hardware section's four pre-registered throughput measurements, the
   single-inference-seam collection-loop contract, and the named-optional
   BC-on-replays diagnostic.
3. PLAN.md § Phase 4's findings entries (2026-07-28/29) — they set the
   capstone's priorities: opponent/state diversity is what matters;
   encoder capacity at this scale is not the fear.

## Phase 5 first steps (nothing is committed to yet — plan with the maintainer)
1. Env plumbing: local Node.js Showdown server + poke-env pinned dep
   (through `pyproject.toml`, exact pin), `gen1randombattle`, the
   mask-to-info adapter wrapper, `info["outcome"]` mapping.
2. The four pre-registered throughput measurements (PLAN.md hardware note)
   BEFORE any provisioning or vectorization decision.
3. Milestone ladder: beat `MaxBasePowerPlayer` → beat
   `SimpleHeuristicsPlayer` (headline metric: win rate over ≥1000 battles,
   multiple seeds) → self-play with the pool → optional ladder Elo.
4. What transfers from Phase 4: the pool/`Opponent` protocol (including
   `report()` for PFSP and `fixed_mix` — the one lever that worked),
   `eval/win_rate` from `info["outcome"]`, the forgetting instruments, and
   the lesson that opponent diversity is the thing to engineer for.

## Gotchas not in the plan
- `deep-rl` conda env only: `/opt/anaconda3/envs/deep-rl/bin/pytest tests/`.
- Training and anything multi-minute goes to the maintainer's terminal
  (`WANDB_MODE=offline` on training commands); the ~1-minute class may run
  in-session. When in doubt, hand it over.
- Never edit the tree (even uncommitted) while the maintainer may be
  LAUNCHING a run — launches stamp `git_dirty`, and an untracked file is
  enough to flip it (measured: an advisory .md dirtied 8 of 9 lever-run
  stamps; the attribution note is in the 2026-07-29 lever entry). Never
  run mutation batteries while any maintainer process might import mutated
  source.
- zsh traps in handed-over commands: no `echo =====`, no inline `#`, no
  multi-line blocks; state-changing steps one block at a time.
- The mutation batteries' `old` strings match current source exactly; a
  refactor that breaks one is a prompt to update the spec, not delete it.
- `scripts/pons_agent_metrics.py` refuses non-self-play checkpoints unless
  `--allow-non-selfplay`; `best_checkpoint.pt` feeds no reported number
  anywhere (selection bias — tournaments select).

## First actions
1. Confirm with the maintainer that Phase 5 is starting and which step
   comes first (env plumbing is the natural opening; dependency changes go
   through `pyproject.toml` with exact pins and a go-ahead).
2. Read the docs above.
3. State which files you'll create/change and why; wait for a go-ahead.
4. Small commits, pytest green per step, mutation-test each guard.
