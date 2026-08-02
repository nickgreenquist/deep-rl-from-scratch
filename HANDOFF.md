# Handoff — written 2026-08-02 on explicit request; fold in, then restore the stub

STATUS.md and the log are CURRENT (two 2026-08-02 entries: P4 pre-registration,
P4 verdict; locked spec lives in PLAN.md Phase 5). This file is a map plus the
few things not yet durable. Read, verify, restore the stub.

## State
- Clean at `47ebfef`. 2 commits ahead of origin (`50c6a44` P4 pre-registration,
  `47ebfef` P4 verdict) — ask the maintainer before pushing.
- Doc layout is the restructured one: session start = this file if non-empty →
  STATUS.md; PLAN.md holds Phase 5 + the locked P4 spec; Phases 0–4 live in
  PLAN_ARCHIVE.md.
- **P4 COMPLETE — the plateau is training-side.** BC clone through the exact
  capstone encoder + [512,512] plays 0.453–0.465 vs SimpleHeuristics (R3 passed
  in both batteries; +0.045–0.057 over the 0.408 RL best). Encoder exonerated;
  one-directional caveat stands (representable + supervised-learnable ≠
  PPO-reachable under terminal-only reward). R2 closed PARTIAL at 0.90 — data
  still binding after the one spent doubling, curve extrapolates onto the
  audit's predicted ~0.97; capacity probe skipped as uninterpretable (disclosed
  deviation, reasoned in the verdict entry).

## Next (in recommended order — none started)
1. Milestone-3 write-up (README section) + the stop-rule decision, with P4's
   answer in hand. Keep the 0.5 bar unmoved, cross-play co-reported, the bar's
   date attached; verify every number programmatically against run artifacts
   before commit. The arc: three arms converge on the same ~0.4 plateau, and P4
   shows that plateau sits below a demonstrated-representable policy.
2. Queued behind: P3 (team-luck variance, ~20 min), P5 (rollout_steps 512), and
   the NEW question P4 created — BC-warm-start from the 0.465 clone. That one
   is a ladder-design decision (it would change the from-scratch narrative run
   3 established); flagged in the P4 spec, deliberately not taken.

## Watch items / small
- poke-env 0.15.0 upstream bug found during the P4 audit: SimpleHeuristics'
  setup branch is dead (`move.target == "self"` compares an int enum to a
  string) — SH is a pure damage-maximizer + matchup-switcher everywhere.
  Internal comparability unaffected. Possibly worth an upstream report.
- `test_full_episode_contract_against_live_server` flakes on ordering when the
  full suite runs with the server up; passes alone; pre-existing (2026-08-01).
- data/bc_p4_{main,sub10k,40k}.npz ≈ 3.9 GB, gitignored — deletable; full
  regeneration is ~10 min via scripts/make_bc_dataset.py.
- P4 run scripts + logs are in the old session's tmp dir (jobs/acc47a2b/tmp/
  p4_run*.{sh,log}) — ephemeral, everything durable is in runs/ and the log.

## Operational
- wandb is now OFFLINE BY DEFAULT in code (`e53323a`) — the per-shell
  WANDB_MODE=offline export is no longer needed; explicit WANDB_MODE wins.
- Server: `cd showdown && node pokemon-showdown start --no-security`.
- Handed-over command sets go in bash scripts under the session tmp dir; one
  command per fenced block, `<command>` sentinels outside the fence.
