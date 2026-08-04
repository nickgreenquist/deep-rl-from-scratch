# Status

Current-state board. Rewritten in place (never appended) as work lands — update it in the
same commit that appends a `SESSION_LOGS.md` entry; hard cap ~80 lines. If this file
conflicts with the newest session-log entry, the log wins — say so and fix this file.

## Where we are (updated 2026-08-04)

- **Milestone-3 write-up SHIPPED**: README Phase-5 section (milestones 1–3 + the cloning
  diagnostic), every number verified against run artifacts. The shipped arc: all three arms
  converge on ~0.4 vs SimpleHeuristics; the BC clone at 0.453 locates the plateau
  training-side (one-directional caveat attached).
- **Stop rule RATIFIED (2026-08-02):** the 0.5 bar stops being chased under this recipe
  class (unmet, unmoved; 16×-budget untested-not-excluded); training probes need their own
  pre-registration. Full wording in the 2026-08-02 write-up log entry.
- Milestone ladder: M1 PASSED 0.663 · M2 NOT passed (fixed-bot 12M pooled 0.417 ± 0.009,
  3 seeds, best seed 0.432) · M3 complete and shipped.
- Pushed through `957b4c1` 2026-08-03 (maintainer's go). Ask before any push; the four
  commits since (P5b pre-reg → scope decisions → throughput reprioritization → this read)
  are local.
- **P3 COMPLETE (2026-08-03):** observable draw explains ~4% of outcome variance (CV R²
  0.0375, p < 0.005; lower bound) — the draw does not decide battles at species level.
- **P5 CREDITED (2026-08-03):** rollout_steps 128→512 lifts the 6M win rate 0.355 → 0.392
  pooled (z = 3.0, pre-registered; whole 4–6M band shifted, approx_kl halved).
- **P5b CREDITED (2026-08-04):** LR anneal (linear 2.5e-4 → 0 over 6M) on the r512 recipe —
  pooled finals **0.4433 ± 0.0091** vs control 0.3923 ± 0.0089; delta +0.051, twice the
  +0.025 credit line (z ≈ 4.0); per seed 0.416/0.468/0.446. All R0 gates passed; whole
  4–6M band up. **First result above the ~0.42 plateau; beats 12M flat-lr (0.417) at half
  budget; within noise of the BC clone (0.453).** README closing paragraph amended per the
  pre-stated condition. Annealed ckpts cannot be warm-extended — any 12M anneal test is a
  from-scratch `lr_anneal_steps: 12000000` run.
- **Prior-work verification COMPLETE (2026-08-03; full entries in the log):** our 0.39–0.42
  was in-band for scratch PPO (VGC-Bench scratch 0.48 at 5M); **BC-init +25–30 pts at
  matched budget** — best-evidenced remaining lever; Wang's LR-anneal ablation now
  replicated directionally in-repo (P5b). Wang + ps-ppo both used NO opponent pool. Gen-1
  favorable for pure policy (MCTS #8 vs policy #1/#2 in Gen1OU). Sources in `prior_work/`
  (PDFs local-only, gitignored; index tracked).
- **Bridge audit CLEAN (2026-08-03):** 866 decisions — 0 stale masks, 0 mismatches; the
  pokejax bug class does not reproduce on our request-driven path.
- **Scope decisions (2026-08-03, in PLAN.md):** MCTS is an OPEN follow-up phase (inference-
  only, Wang pattern; serialization verified upstream in our Showdown checkout). "Pure
  self-play" retired as an identity constraint — BC init, shaping, teacher data are
  first-class for the BC design session.

## Next, in order

1. **Throughput session (maintainer-directed 2026-08-03, goals in the PLAN scope block):**
   server-port knob + one server per lane (measured basis: shared server peaks at W=2;
   per-server ~7.5k dec/s at W=4–8), lane-scaling W=3–6 through the full loop, facade
   go/no-go via measurement (e). Cheap steps compound into every later pre-registration.
2. **BC-warm-start design session** — the stack (BC init + shaping + anneal verdict) as one
   pre-registered package per the PLAN scope block. The anneal is now a credited component.
3. 12M r512 extension decision — still open; if taken, the annealed arm must be from-scratch
   (`lr_anneal_steps: 12000000`), and flat-vs-annealed at 12M is the natural framing.

## Watch items

- s0 late regression pattern (r512 flat-lr): weak seed in cross-play; P5b's s0 (0.416) was
  also the weak seed — watch seed spread in any 12M run.
- Pre-existing test flake: `test_full_episode_contract_against_live_server` fails only when
  the whole suite runs with a server up; passes alone (2026-08-01).
- poke-env 0.15.0 upstream bug (SH setup branch dead: int enum vs string) — shipped as a
  README finding; an upstream report is still unfiled.
- `data/bc_p4_{main,sub10k,40k}.npz` ≈ 3.9 GB, gitignored — deletable; regeneration ~10 min
  via `scripts/make_bc_dataset.py`.

## Operational

- Server: `cd showdown && node pokemon-showdown start --no-security` (one on :8000 now).
- wandb defaults to offline as of `e53323a` (an explicit `WANDB_MODE` still wins).
- `runs/*/history.csv` (gitignored) exist for heur_512, heur_6m, scratch12m, r512_lra ×3 —
  regenerate via `scripts/extract_history.py`.
- Stage-0 pattern for concurrent seeds; ≥5-min runs in the maintainer's terminal; clean
  tree at every launch; commit docs BEFORE launches. Handed-over command sets go in bash
  scripts under the session tmp dir.
- Run artifacts: `runs/showdown_r512_lra_s{0,1,2}/` (P5b), `runs/showdown_scratch12m_s*/`,
  `runs/showdown_heur_512_s0/`, clones `runs/bc_p4_512{,_40k}_s{0,1,2}/`.
