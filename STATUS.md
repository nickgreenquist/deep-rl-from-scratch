# Status

Current-state board. Rewritten in place (never appended) as work lands — update it in the
same commit that appends a `SESSION_LOGS.md` entry; hard cap ~80 lines. If this file
conflicts with the newest session-log entry, the log wins — say so and fix this file.

## Where we are (updated 2026-08-03)

- **Milestone-3 write-up SHIPPED**: README has the Phase-5 results section (milestones 1–3
  + the cloning diagnostic) with a house-style figure; every number verified
  programmatically against run artifacts; three-Opus review pass folded in. The arc as
  shipped: fixed-bot / warm-started / from-scratch arms all converge on ~0.4 vs
  SimpleHeuristics, and the BC clone at 0.453 through the same stack locates the plateau
  training-side (one-directional caveat attached; the clone's R2 fit gate miss disclosed).
- **Stop rule RATIFIED by the maintainer (2026-08-02 evening).** As adopted:
  (1) M3 ships now; (2) P3 = post-ship analysis appendix, P5 = real training probe needing
  its own pre-registration (entropy probe queued same tier); (3) the 0.5 bar stops being
  chased under this recipe class — bar stays unmet and unmoved; 16×-budget from-scratch is
  untested-not-excluded, deferred on cost; (4) BC-warm-start deferred to its own
  pre-registered design session. Full wording in the 2026-08-02 write-up log entry.
- Milestone ladder: M1 PASSED 0.663 · M2 NOT passed (fixed-bot 12M pooled 0.417 ± 0.009,
  3 seeds, best seed 0.432; the 18M continuation's 0.432 reads as specialization) · M3
  complete and shipped.
- Pushed through `17ae11b` 2026-08-03 (maintainer's go) — the milestone-3 section is
  PUBLIC. Standing rule unchanged: ask before any future push.
- **P3 COMPLETE (2026-08-03):** observable draw explains ~4% of outcome variance (CV R²
  0.0375, p < 0.005; lower bound) — the draw does not decide battles at species level.
- **P5 CREDITED (2026-08-03):** rollout_steps 128→512 lifts the 6M win rate 0.355 → 0.392
  pooled (z = 3.0, pre-registered; whole 4–6M band shifted, approx_kl halved). First
  credited lever since capacity; at half the budget it nearly matches the base recipe's
  12M value. README closing paragraph amended per the pre-stated condition.

- **Prior-work verification COMPLETE (2026-08-03, three-Opus dig; full entry in the log):**
  the maintainer's external briefing largely survives. Key calibrations: Wang's gen4 agent
  (tuned + annealed lr) was at **0.575 vs SH at our 6M budget** (digitized Fig 4.1; crosses
  0.5 at ~4M); our 0.39–0.42 is in-band for scratch PPO (VGC-Bench scratch SP 0.48 at 5M;
  BC-initialized 0.62–0.78 — **BC init +25–30 pts at matched budget** is the best-evidenced
  lever, LR annealing the only controlled ablation: 0.55→0.80). ps-ppo's "MLP stuck at 1100
  Elo" claim has zero code support (anecdote); its 2102-Elo pure-policy result is real but
  confounded (BC init + shaping + arch, no ablation). Gen-1 favorable: MCTS placed #8 in
  Gen1OU (PokéAgent) where pure policy took #1/#2; SH is weakest vs humans in Gen 1 (~0.21).
  Wang + ps-ppo both used NO opponent pool. Our poke-env 0.15.0 SH has a +1-boost stat bug
  (evaluates as +2) — stock-vs-patched comparability caveat only. All Showdown runs train
  flat lr (anneal machinery exists; the 2026-08-01 lr-test rejection was about magnitude,
  not schedule).

## Next, in order

1. **Maintainer direction decision:** (a) 12M r512 extension (~5.2 h; plateau vs approach
   speed) vs (b) LR-anneal probe (new candidate from verification; needs pre-registration
   + reconciling MinAtar's "anneal cost 35%") vs (c) straight to the BC-warm-start design
   session (now the best-evidenced big lever; ps-ppo's BC-as-architecture-screen folds in).
2. Push decision — commits since `17ae11b` are local.
3. Optional cheap check: audit our poke-env bridge vs the pokejax bug list (stale
   `available_moves` after switches, PP not decrementing, sleep off-by-one).

## Watch items

- s0 late regression (0.396→0.365 last 2M; weak seed in cross-play) — now disclosed in the
  shipped section; keep an eye on recurrence in any future run.
- Pre-existing test flake: `test_full_episode_contract_against_live_server` fails only when
  the whole suite runs with a server up; passes alone (2026-08-01).
- poke-env 0.15.0 upstream bug (SH setup branch dead: int enum vs string) — shipped as a
  README finding; an upstream report is still unfiled.
- `data/bc_p4_{main,sub10k,40k}.npz` ≈ 3.9 GB, gitignored — deletable; regeneration ~10 min
  via `scripts/make_bc_dataset.py`.

## Operational

- Server: `cd showdown && node pokemon-showdown start --no-security`
- wandb defaults to offline as of `e53323a` (an explicit `WANDB_MODE` still wins).
- `runs/*/history.csv` (gitignored) now exist for heur_512, heur_6m, scratch12m ×3 — the
  figure script reads them; regenerate via `scripts/extract_history.py`.
- Stage-0 pattern for concurrent seeds; ≥5-min runs in the maintainer's terminal; clean
  tree at every launch; commit docs BEFORE launches. Handed-over command sets go in bash
  scripts under the session tmp dir.
- Run artifacts: `runs/showdown_scratch12m_s{0,1,2}/`, parents
  `runs/showdown_{sp6m,cont6m}_s*/`, `runs/showdown_heur_512_s0/`, clones
  `runs/bc_p4_512{,_40k}_s{0,1,2}/`.
