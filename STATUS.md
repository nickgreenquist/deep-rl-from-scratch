# Status

Current-state board. Rewritten in place (never appended) as work lands — update it in the
same commit that appends a `SESSION_LOGS.md` entry; hard cap ~80 lines. If this file
conflicts with the newest session-log entry, the log wins — say so and fix this file.

## Where we are (updated 2026-08-02)

- **Phase 5 (Pokémon Showdown capstone), milestone-3 campaign COMPLETE** — from-scratch
  self-play LEARNS on Showdown: finals 0.380 ± 0.009 pooled vs SimpleHeuristics (above the
  pre-registered 0.20–0.35 band), 0.484 head-to-head vs the equal-budget fixed-bot parent,
  without ever seeing the eval bot. Everything approaches the same ~0.4 plateau; the plateau,
  not the training distribution, binds. (Three 2026-08-01 SESSION_LOGS entries: run-3 design,
  run-3 gates, run 3 complete.)
- Milestone ladder: M1 (MaxBasePowerPlayer) PASSED 2026-07-29 · M2 (SimpleHeuristics, 0.5
  bar) NOT passed — best 0.408 at 12M with [512,512], curve not flattened · M3 (self-play
  pool) campaign complete; stop-rule decision for shipping it still OPEN.
- **P4 COMPLETE 2026-08-02 — the plateau is TRAINING-SIDE.** The BC clone through the exact
  capstone encoder + [512,512] trunk plays 0.453–0.465 vs SimpleHeuristics (R3 passed in both
  batteries; +0.045–0.057 over the 0.408 RL best, above the 0.42 asymptote), so the encoder is
  exonerated for the plateau. R2 closed partial (0.90, data still binding after the spent
  doubling; curve extrapolates onto the audit's predicted ~0.97; capacity probe skipped as
  uninterpretable — disclosed deviation). One-directional caveat stands: representable and
  supervised-learnable ≠ PPO-reachable under terminal-only reward. The passing clone is a
  warm-start candidate above the RL best — a ladder decision, NOT taken. (Two 2026-08-02
  entries: pre-registration + verdict; locked spec in PLAN.md Phase 5.)
- Git: 2 commits ahead of origin (P4 pre-registration, P4 verdict). Standing rule: ask the
  maintainer before any push.

## Next, in order

1. **Milestone-3 write-up (README section) + stop-rule decision**, now with P4's answer in
   hand: the three-arm arc plus "the plateau is training-side, demonstrated" — keep the 0.5
   bar unmoved, cross-play co-reported, the bar's date attached. Force the stop rule here.
2. P3 (team-luck variance decomposition, ~20 min) and P5 (rollout_steps 512 — the config's
   only true SNR knob) queued behind; the BC-warm-start question joins the ladder-design
   queue (flagged in the P4 spec, not decided).

## Watch items

- s0 late regression is real (last-2M eval dip 0.395→0.365; cross-play 0.434 vs sp6m) —
  Phase-4 "best rung ≠ final" recurring on 1 of 3 seeds; a write-up caveat, not a defect.
- Pre-existing test failure: `test_full_episode_contract_against_live_server` fails only when
  the whole suite runs with a server up; passes when its file runs alone. Reproduced on an
  unmodified tree (2026-08-01 P4 entry).

## Operational

- Server: `cd showdown && node pokemon-showdown start --no-security`
- wandb defaults to offline as of `e53323a` (an explicit `WANDB_MODE` still wins).
- Stage-0 pattern for concurrent seeds; ≥5-min runs in the maintainer's terminal; steps/s
  from meta.yaml → checkpoint.pt mtimes; clean tree at every launch; commit docs BEFORE
  launches. Handed-over command sets go in bash scripts under the session tmp dir — long
  single-line chains mangle on paste (measured 2026-08-01).
- Throughput: from-scratch 3-wide 568–575 steps/s shakeout, ~546 effective over 12M with
  evals.
- Run artifacts: `runs/showdown_scratch12m_s{0,1,2}/` (finals, xplay JSONs, offline
  histories); parent cells `runs/showdown_{sp6m,cont6m}_s*/`, `runs/showdown_heur_512_s0/`.
