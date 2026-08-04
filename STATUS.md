# Status

Current-state board. Rewritten in place (never appended) as work lands — update it in the
same commit that appends a `SESSION_LOGS.md` entry; hard cap ~80 lines. If this file
conflicts with the newest session-log entry, the log wins — say so and fix this file.

## Where we are (updated 2026-08-04)

- **Milestone-3 write-up SHIPPED**: README Phase-5 section (milestones 1–3 + the cloning
  diagnostic), every number verified against run artifacts. All three arms converge on ~0.4
  vs SimpleHeuristics; the BC clone at 0.453 locates the plateau training-side.
- **Stop rule RATIFIED (2026-08-02):** the 0.5 bar stops being chased under this recipe
  class; training probes need their own pre-registration. Wording in the 2026-08-02 entry.
- Milestone ladder: M1 PASSED 0.663 · M2 NOT passed (fixed-bot 12M pooled 0.417 ± 0.009,
  best seed 0.432) · M3 complete and shipped. Pushed through `422f9ee` 2026-08-04.
- **P3 COMPLETE (2026-08-03):** observable draw explains ~4% of outcome variance (CV R²
  0.0375, p < 0.005; lower bound) — the draw does not decide battles at species level.
- **P5 CREDITED (2026-08-03):** rollout_steps 128→512 lifts the 6M win rate 0.355 → 0.392
  pooled (z = 3.0). Rollout length and batch size moved together; neither was isolated.
- **P5b CREDITED (2026-08-04):** LR anneal (linear 2.5e-4 → 0 over 6M) on r512 — pooled
  **0.4433 ± 0.0091** vs control 0.3923 ± 0.0089, delta +0.051 (2× the credit line, z ≈ 4.0);
  per seed 0.416/0.468/0.446. First result above the ~0.42 plateau; beats 12M flat-lr (0.417)
  at half budget; within noise of the BC clone (0.453). README amended. Annealed ckpts cannot
  be warm-extended — any 12M anneal arm is from-scratch `lr_anneal_steps: 12000000`.
- **THROUGHPUT SESSION COMPLETE (2026-08-04), no `rl/` source changed.** Work item (1) was a
  one-line edit to gitignored `showdown/config/config.js`: `simulator: 1 → 4`. Shared-server
  collection now 2,237/5,246/7,096/9,233/11,024/11,313/9,966 dec/s at W=1/2/3/4/6/8/12 —
  plateau W=6–8; **+81% at W=4 vs simulator:1, beats one-server-per-lane by 26–50%. Server
  sharding RETIRED.** Full loop: **W=3 → 659 steps/s per lane, W=6 → 556** — the ≥685 goal
  **NOT met** (~4% short); lane scaling met (3→6 costs 15.6%/lane, returns +41% aggregate).
- **LOAD-BEARING FINDING: collection-only benchmarks overstate full-loop gain ~7×.**
  simulator:4 bought ~29% collection-side but **+3.7% end-to-end** (615 → 638 mean vs the
  P5b lanes, same recipe/3-wide/machine). **The loop is update-and-encode bound, not
  collection bound** — contradicting the Phase-5 hardware note, the collection-loop
  architecture work, and the surrogate-tuning interest, all of which assume the opposite.
- **Facade CLOSED (2026-08-04) as a self-play-scoped item, not on new measurement.** Prize 1
  measured: [512,512] batch-1 83.1 µs vs batch-8 41.4 µs/sample = 2.04×, worth 2.5% under
  self-play and **exactly 0% under `opponent: heuristics`** (every queued run); prize 2 is
  bounded by server-wait, which the 3.7% says barely exists. **Record corrected:** late-July's
  "headroom ~zero at [512,512]" is wrong — right verdict, wrong arithmetic; the `[64,64]`
  hardcode in `showdown_throughput.py` has now caused two misreads. Revisit only when a
  self-play chapter is designed, priced as a code-cost tradeoff.
- **Prior work + scope (2026-08-03, in PLAN.md):** our 0.39–0.42 was in-band for scratch PPO;
  **BC-init +25–30 pts at matched budget** is the best-evidenced remaining lever (Wang's
  anneal ablation is now replicated in-repo by P5b). MCTS is an OPEN follow-up phase; "pure
  self-play" retired as an identity constraint. Sources in `prior_work/`.

## Next, in order

1. **Loop-split instrumentation** — collect / encode / update / eval as separate timers, so
   the update-and-encode split is measured rather than inferred. Highest-leverage targets in
   order: the observation encoder (our Python, per decision), then the PPO update (the one
   place a GPU could matter at [512,512]). Wang needed the same instrument and could not get
   it from stock SB3 (7 of his 8 fork commits); see the 2026-08-04 log entry.
2. **BC-warm-start design session** — the stack (BC init + shaping + anneal verdict) as one
   pre-registered package per the PLAN scope block. The anneal is now a credited component.
3. 12M r512 extension decision — still open. If taken, the annealed arm is from-scratch, and
   flat-vs-annealed at 12M is the natural framing. ~5.2 h at 3-wide post-`simulator: 4`.

## Watch items

- **Startup-crash hazard (2026-08-04):** a lane can die with SIGSEGV in torch lazy static
  init before writing any log or run dir, and a naive launcher reports success over a
  short-handed result (hit W=6: 5 of 6). Not memory. A related SIGABRT at teardown hit the
  P5b finals (results unaffected). **Every launcher must stagger starts and assert all W run
  dirs exist with complete histories before reporting done.**
- s0 late regression pattern; P5b's s0 was also the weak seed (0.416) — watch seed spread.
- Pre-existing test flake: `test_full_episode_contract_against_live_server` fails only when
  the whole suite runs with a server up; passes alone (2026-08-01).
- poke-env 0.15.0 upstream bug (SH setup branch dead); upstream report still unfiled.
- `data/bc_p4_*.npz` ≈ 3.9 GB, gitignored — deletable; regen ~10 min.

## Operational

- Server: `cd showdown && node pokemon-showdown start --no-security` — now `simulator: 4`
  (gitignored file; re-set it if the checkout is ever recreated).
- wandb defaults to offline as of `e53323a`. `runs/*/history.csv` via `extract_history.py`.
- Stage-0 pattern for concurrent seeds; ≥5-min runs in the maintainer's terminal; clean tree
  at every launch; commit docs BEFORE launches; handed-over command sets go in bash scripts
  under the session tmp dir.
- Run artifacts: `runs/showdown_r512_lra_s{0,1,2}` (P5b), `runs/showdown_scratch12m_s*`,
  clones `runs/bc_p4_512{,_40k}_s{0,1,2}`. `runs/showdown_tput_w*` are disposable.
