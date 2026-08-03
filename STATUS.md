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

- **Both hardening steps CLOSED (2026-08-02 later + overnight entries):** clone
  final-vs-best measured a non-issue (12 evals, all within noise), and the heur_512
  replication ran overnight — s1 0.411 / s2 0.432, pooled p_RL 0.417 ± 0.009 (spread
  0.024). Retention rule fired RESOLVED: clone − p_RL = +0.036, z = 2.81. README amended
  per the locked rule (wedge, milestone table, figure now 3 fixed-bot seeds); the n=1
  caveat on the lineage's key number is retired.

## Next, in order

1. **Maintainer decision:** 12M r512 extension (~5.2 h; would test whether the credited
   SNR lever moves the ~0.42 plateau or only the approach speed — and a better base recipe
   changes what the BC warm start grafts onto) vs straight to the BC-warm-start design
   session. Extension needs its own pre-registration either way.
2. Push decision — this session's commits (P3/P5 results, README amendment) are local.
3. BC-warm-start design session (next chapter's opener, pre-registered meaning first).

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
