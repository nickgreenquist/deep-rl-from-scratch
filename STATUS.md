# Status

Current-state board. Rewritten in place (never appended) as work lands — update it in the
same commit that appends a `SESSION_LOGS.md` entry; hard cap ~80 lines. If this file
conflicts with the newest session-log entry, the log wins — say so and fix this file.

## Where we are (updated 2026-08-03)

- **Milestone-3 write-up SHIPPED**: README Phase-5 section (milestones 1–3 + the cloning
  diagnostic), every number verified against run artifacts. The shipped arc: all three arms
  converge on ~0.4 vs SimpleHeuristics; the BC clone at 0.453 locates the plateau
  training-side (one-directional caveat attached).
- **Stop rule RATIFIED (2026-08-02):** the 0.5 bar stops being chased under this recipe
  class (unmet, unmoved; 16×-budget untested-not-excluded); training probes need their own
  pre-registration. Full wording in the 2026-08-02 write-up log entry.
- Milestone ladder: M1 PASSED 0.663 · M2 NOT passed (fixed-bot 12M pooled 0.417 ± 0.009,
  3 seeds, best seed 0.432; the 18M continuation's 0.432 reads as specialization) · M3
  complete and shipped.
- Pushed through `957b4c1` 2026-08-03 (maintainer's go); milestone-3 + P3/P5 + the
  verification fold are PUBLIC. Ask before any push; commits since are local.
- **P3 COMPLETE (2026-08-03):** observable draw explains ~4% of outcome variance (CV R²
  0.0375, p < 0.005; lower bound) — the draw does not decide battles at species level.
- **P5 CREDITED (2026-08-03):** rollout_steps 128→512 lifts the 6M win rate 0.355 → 0.392
  pooled (z = 3.0, pre-registered; whole 4–6M band shifted, approx_kl halved). First
  credited lever since capacity; at half the budget it nearly matches the base recipe's
  12M value. README closing paragraph amended per the pre-stated condition.

- **Prior-work verification COMPLETE (2026-08-03; full entries in the log):** Wang's gen4
  agent (tuned + annealed lr) was at **0.575 vs SH at our 6M budget**; our 0.39–0.42 is
  in-band for scratch PPO (VGC-Bench scratch 0.48 at 5M; **BC-init +25–30 pts at matched
  budget** — best-evidenced lever; LR annealing the only controlled ablation, 0.55→0.80).
  ps-ppo's "MLP stuck at 1100 Elo" has zero code support; its 2102-Elo pure-policy result
  is real but confounded. Gen-1 favorable: MCTS placed #8 in Gen1OU where pure policy took
  #1/#2; SH weakest vs humans in Gen 1 (~0.21). Wang + ps-ppo both used NO opponent pool.
  Sources archived in `prior_work/` (PDFs local-only, gitignored; index tracked).
- **Bridge audit CLEAN (2026-08-03):** 866 decisions (402 post-switch) vs heuristics — 0
  stale `available_moves`, 0 mask mismatches, PP decrements; the pokejax bug class does
  not reproduce on our request-driven path. Headline number not depressed by it.
- **P5b LAUNCHED (2026-08-03 ~20:56, 3 seeds up):** LR-anneal probe — one variable,
  `lr_anneal_steps` 0→6M (linear) on the r512 recipe, 6M × 3 seeds vs the r512 pooled
  control 0.3923 ± 0.0089. **CREDITED iff pooled ≥ 0.418.** Locked read in
  `configs/showdown_r512_lra.yaml`. Annealed ckpts cannot be warm-extended.
- **Scope decisions (2026-08-03, in PLAN.md):** MCTS is an OPEN follow-up phase (inference-
  only, Wang pattern) — the "no forward model" premise is revised: serialization is upstream
  in our own Showdown checkout (verified file:line); Wang's forks read + archived
  (`prior_work/wang_fork_diffs.md`). "Pure self-play" retired as an identity constraint —
  BC init, shaping, teacher data are first-class for the BC design session.

## Next, in order

1. **P5b read when finals land** (~3 h from launch; per the locked config header). No `rl/`
   source edits until `lra_probe.sh` exits fully — its finals stage imports `rl` fresh.
2. **Throughput session (maintainer-directed 2026-08-03, goals in the PLAN scope block):**
   server-port knob + one server per lane (measured basis: shared server peaks at W=2;
   per-server ~7.5k dec/s at W=4–8), lane-scaling W=3–6 through the full loop, facade
   go/no-go via measurement (e). Cheap steps compound into every later pre-registration.
3. **BC-warm-start design session** — the stack (BC init + shaping + anneal verdict) as one
   pre-registered package per the PLAN scope block.
4. 12M flat-lr r512 extension decision — still open (interacts with P5b and the facade).

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
