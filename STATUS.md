# Status

Current-state board. Rewritten in place (never appended) as work lands — update it in the
same commit that appends a `SESSION_LOGS.md` entry; hard cap ~80 lines. If this file
conflicts with the newest session-log entry, the log wins — say so and fix this file.

## Where we are (updated 2026-08-02, evening)

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
- Milestone ladder: M1 PASSED 0.663 · M2 NOT passed (best 0.408 n=1; 0.432 pooled reads as
  specialization) · M3 complete and shipped.
- Git: 4 commits ahead of origin after this session (P4 pre-reg, P4 verdict, write-up,
  docs). Standing rule: ask the maintainer before any push — and the push now publishes
  the milestone-3 section.

- **Both write-up hardening steps taken (2026-08-02 later entry):** clone final-vs-best
  asymmetry measured a non-issue (all 12 evals within noise; README parenthetical added);
  `heur_512_s{1,2}` 12M replication PRE-REGISTERED and handed over for overnight — the
  3-seed pooled replaces the n=1 0.408 in the wedge, retention rule locked (resolved iff
  0.4530 − p_RL ≥ 2·se_diff ≈ 0.026).

## Next, in order

1. **Overnight (maintainer's terminal): `heur512_seeds.sh`** — heur_512 s1+s2, 12M each
   2-wide, finals included (~5–6 h). Needs server up + the already-clean tree.
2. **Morning read**: pre-registered rule from the 2026-08-02 later entry; amend the README
   wedge sentence + milestone table with the 3-seed number either way.
3. Push decision (maintainer) — publishes the milestone-3 section.
4. P3 team-luck variance decomposition (~20 min, analysis appendix).
5. P5 rollout_steps 512 — pre-registration first, then run; BC-warm-start design session
   behind it.

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
