# Status

Current-state board. Rewritten in place (never appended) as work lands — update it in the
same commit that appends a `SESSION_LOGS.md` entry; hard cap ~80 lines. If this file
conflicts with the newest session-log entry, the log wins — say so and fix this file.

## Where we are (updated 2026-08-01)

- **Phase 5 (Pokémon Showdown capstone), milestone-3 campaign COMPLETE** — from-scratch
  self-play LEARNS on Showdown: finals 0.380 ± 0.009 pooled vs SimpleHeuristics (above the
  pre-registered 0.20–0.35 band), 0.484 head-to-head vs the equal-budget fixed-bot parent,
  without ever seeing the eval bot. Everything approaches the same ~0.4 plateau; the plateau,
  not the training distribution, binds. (Three 2026-08-01 SESSION_LOGS entries: run-3 design,
  run-3 gates, run 3 complete.)
- Milestone ladder: M1 (MaxBasePowerPlayer) PASSED 2026-07-29 · M2 (SimpleHeuristics, 0.5
  bar) NOT passed — best 0.408 at 12M with [512,512], curve not flattened · M3 (self-play
  pool) campaign complete; stop-rule decision for shipping it still OPEN.
- P4 machinery (BC-clone encoder-ceiling instrument) built, smoke-tested end to end, priced:
  the whole diagnostic runs in-session in minutes (collection 2,825 decisions/s, ~0.7 s/epoch
  at 40k rows × [512,512], 1,000-battle re-eval ~50 s). Pre-registration deliberately NOT yet
  taken. (2026-08-01 P4 entry.)
- Git: local is ahead of origin — ask the maintainer before pushing.

## Next, in order

1. **P4 — encoder-ceiling BC diagnostic:** pre-register the read, then run (clone
   SimpleHeuristics from (obs, bot_action) pairs through the same 611-dim encoder +
   [512,512]). Decisive if it FAILS (plateau explained, encoder must change); one-directional
   if it passes — carry that caveat into the pre-registration.
2. **Milestone-3 write-up (README section):** the three-arm arc is fully pre-registered; keep
   the 0.5 bar unmoved, cross-play co-reported, the bar's date attached. Force the stop-rule
   decision here.
3. P3 (team-luck variance decomposition, ~20 min) and P5 (rollout_steps 512 — the config's
   only true SNR knob) queued behind.

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
