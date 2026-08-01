# Handoff — written 2026-08-01 on explicit request; fold in, then restore the stub

The session log is CURRENT: three 2026-08-01 entries (run-3 design session,
run-3 gates, run-3 complete) carry every number, verdict and decision from
this session. This file is a map. Read, verify, restore the stub.

## State
- Clean at `d9fb4ba`. NOT pushed — origin sits 5 commits back at `798c306`;
  ask the maintainer before pushing.
- Milestone-3 campaign COMPLETE, three-arm arc coherent (run-3-complete
  entry): (1) warm-started self-play at matched budget moved nothing
  (0.5050 vs parent, se 0.0065); (2) continued fixed-bot gained only on its
  own anchor (+0.024; 0.5098 head-to-head — specialization); (3)
  from-scratch self-play 12M×3 LEARNS (finals 0.3800 ± 0.0089, above the
  pre-registered 0.20–0.35 band; 0.4837 head-to-head vs the equal-budget
  parent) without ever seeing the eval bot. Everything approaches the same
  ~0.4 plateau; the plateau, not the training distribution, binds.
- The run-3 design entry holds the load-bearing re-analysis (verified
  in-session from the wandb histories): CT was flat in-run too; fixed-bot
  asymptote extrapolates to ~0.42 (bar 0.5 unreachable by budget); run-1
  pool proven strength-homogeneous; recipe-level claims were ~5×
  underpowered; +0.018 deterministic-seat effect. Candidate kills with
  reasons (latest_prob at any dose, pre-seeded pool, bar-chasers) live
  there too — do not re-propose them without new evidence.

## Next (in recommended order — none started)
1. P4, the encoder-ceiling BC diagnostic: clone SimpleHeuristics from
   (obs, bot_action) pairs through the same 611-dim encoder + [512,512].
   Decisive if it FAILS (plateau explained, encoder must change),
   one-directional if it passes — pre-register that caveat. Diagnostic
   outside the milestone ladder (Phase-4 contamination framing,
   allow-non-selfplay-flag precedent). ~1 machinery session (collection
   script + supervised trainer; Phase-4 train_supervised is the pattern).
2. Milestone-3 write-up (README section): the three-arm arc is a fully
   pre-registered story; keep the 0.5 bar unmoved with cross-play
   co-reported and the bar's date attached. Stop-rule decision
   (milestone 3 ships after a bounded set) still OPEN — force it here.
3. P3 (team-luck variance decomposition, ~20 min) and P5 (rollout_steps
   512 — the config's only true SNR knob) queued behind.

## Watch items / small
- s0 late regression is real (last-2M eval dip 0.395→0.365, cross-play
  0.434 vs sp6m) — Phase-4 "best rung ≠ final" recurring on 1 of 3 seeds;
  a write-up caveat, not a defect.
- wandb offline default still undone (one line in rl/common/logging.py +
  test); until then WANDB_MODE=offline per shell on every launch.
- Handed-over command sets go in bash scripts under the session tmp dir —
  a ~2.7k-char && chain mangled on paste (measured this session); never
  hand over long single-line chains again.
- Run-3 artifacts: runs/showdown_scratch12m_s{0,1,2}/ (finals, xplay JSONs,
  offline histories); parent-cell JSONs in runs/showdown_{sp6m,cont6m}_s*/
  and runs/showdown_heur_512_s0/.

## Operational (unchanged)
- Server: `cd showdown && node pokemon-showdown start --no-security`.
- Stage-0 pattern for concurrent seeds; ≥5-min runs in the maintainer's
  terminal; steps/s from meta.yaml → checkpoint.pt mtimes; clean tree at
  every launch; commit docs BEFORE launches.
- From-scratch 3-wide throughput now measured: 568–575 steps/s shakeout,
  ~546 effective over 12M with evals.
