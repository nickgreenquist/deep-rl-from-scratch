# Status

Current-state board. Rewritten in place (never appended) as work lands — update it in the
same commit that appends a `SESSION_LOGS.md` entry; hard cap ~80 lines. If this file
conflicts with the newest session-log entry, the log wins — say so and fix this file.

## Where we are (updated 2026-08-04)

- **Milestone-3 write-up SHIPPED**: README Phase-5 section (milestones 1–3 + the cloning
  diagnostic), numbers verified against run artifacts. All three arms converge on ~0.4 vs
  SimpleHeuristics; the BC clone at 0.453 locates the plateau training-side. Ladder: M1 PASSED
  0.663 · M2 NOT passed (fixed-bot 12M pooled 0.417 ± 0.009) · M3 shipped, pushed `422f9ee`.
- **Stop rule RATIFIED (2026-08-02):** the 0.5 bar stops being chased under this recipe
  class; training probes need their own pre-registration. Wording in the 2026-08-02 entry.
- **P3 COMPLETE (2026-08-03):** observable draw explains ~4% of outcome variance (CV R²
  0.0375, p < 0.005; lower bound) — the draw does not decide battles at species level.
- **P5 CREDITED (2026-08-03):** rollout_steps 128→512 lifts the 6M win rate 0.355 → 0.392
  pooled (z = 3.0). Rollout length and batch size moved together; neither was isolated.
- **P5b CREDITED (2026-08-04):** LR anneal (linear 2.5e-4 → 0 over 6M) on r512 — pooled
  **0.4433 ± 0.0091** vs control 0.3923 ± 0.0089, delta +0.051 (2× the credit line, z ≈ 4.0);
  per seed 0.416/0.468/0.446. First result above the ~0.42 plateau; beats 12M flat-lr (0.417)
  at half budget; within noise of the BC clone (0.453). README amended. Annealed ckpts cannot
  be warm-extended — any 12M anneal arm is from-scratch `lr_anneal_steps: 12000000`.
- **THROUGHPUT (2026-08-04), no `rl/` source changed.** `simulator: 1 → 4` in gitignored
  `showdown/config/config.js` is the whole of it: +81% collection at W=4, beats
  one-server-per-lane by 26–50%, **server sharding RETIRED**. Full loop W=3 → 659 steps/s per
  lane, W=6 → 556; the ≥685 goal NOT met (~4% short), lane scaling met. Curve in the log entry.
- **LOOP SPLIT MEASURED (2026-08-04; instrument in `rl/train.py`, always-on):** **collect
  94.5–95.0%, update 5.0–5.5%, eval negligible** — six lanes, 3- and 6-wide, all agreeing.
  **Supersedes the morning's "update-and-encode bound" inference: the update is not a
  bottleneck and a GPU at [512,512] buys at most ~5%.** Reconciles 29%-collection →
  3.7%-end-to-end: `showdown_throughput.py` measures server-side decisions/s, a small slice of
  collect; our encode + inference is the bulk. **All headroom is in collect** — hence next-2.
- **Facade CLOSED (2026-08-04), self-play-scoped.** Prize 1 is 2.04× but only 2.5% under
  self-play and **0% under `opponent: heuristics`** (every queued run). The `[64,64]` hardcode
  in `showdown_throughput.py` has caused two misreads — quote it only with its width.
- **Prior work + scope (2026-08-03/04, PLAN.md + `prior_work/`):** our 0.39–0.42 is in-band for
  scratch PPO; **BC-init +25–30 pts at matched budget** is the best-evidenced lever. Action
  space CLOSED as a lever (ps-ppo 14, Metamon 9, both positional; Wang's 494-identity outlier
  needed MCTS). Encoder is the live question — see the ps-ppo move-token fields.
- **P6 RUNNING (launched 2026-08-04 ~18:26):** flat vs annealed at 12M on r512, 3 seeds/arm,
  6-wide, both from scratch. Pre-registration committed in `configs/showdown_r512_12m.yaml`.
  524–548 steps/s per lane (inside R0). Result pending.

## Next, in order

1. **P6 finals + read** — run `p6_finals.sh` (1000 battles/seed, locked protocol) once all six
   lanes exit, then `p6_read.py`. Both are staged in the session tmp dir. Do NOT run them while
   training is live; they need the same server.
2. **Decompose collect Showdown-side** — NOW TRIGGERED by the split. Re-run measurement (a)
   (`scripts/showdown_throughput.py a`): it splits per-turn encode vs inference vs env gap but
   last ran on the 10-dim PLACEHOLDER encoder, pre-dating the real 611-dim one (2026-07-30).
   Script change, NOT a seam change — an encode timer must not go in shared code (`embed_battle`
   is inside `ShowdownSingles`; a `hasattr` branch in `rl/train.py` is what the masking contract
   bans).
3. **`DESIGN_P7.md` — PROPOSED, awaiting team review; do NOT ratify before P6 reads.** The
   BC-warm-start package (BC init + staged unfreeze + faint shaping + distributional value),
   aimed where P4 measured the bottleneck. **§10 may dominate it: a verified 109,147-replay
   `gen1randombattle` human corpus exists** — human demos are not bounded by the 0.489
   SH-imitation ceiling that caps every other arm. Phase placement is the open question.

## Watch items

- **CONCURRENT LANES MUST HAVE DISTINCT SEEDS (2026-08-04), across arms too.** Global `random`
  is seeded by `cfg.seed` and poke-env builds usernames from it, so same-seed concurrent lanes
  collide; the loser gets `|nametaken|`, surfacing as `TimeoutError: Agent is not challenging`
  at first `reset`. Killed P6's whole annealed arm.
- **Launcher hygiene (2026-08-04), three ways it has now lied:** SIGSEGV in torch lazy static
  init kills a lane before any log or run dir (W=6: 5 of 6; not memory); the run dir is written
  BEFORE the first `reset`, so `-d` is true for a lane that never trains; and unquoted `$VAR`
  does not word-split in **zsh**, so shell loops must run under `bash`. **Stagger starts, assert
  battle PROGRESS not artifacts, and verify complete histories before reporting done.**
- s0 late regression pattern; P5b's s0 was also the weak seed (0.416) — watch seed spread.
- Pre-existing test flake: `test_full_episode_contract_against_live_server` fails only when
  the whole suite runs with a server up; passes alone (2026-08-01).
- poke-env 0.15.0 upstream bug (SH setup branch dead); upstream report still unfiled.
- `data/bc_p4_*.npz` ≈ 3.9 GB, gitignored — deletable; regen ~10 min.

## Operational

- **Reference source on disk: `/Users/nickgreenquist/Documents/Projects/ps-ppo`** — full clone
  of the strongest pure-policy Showdown agent (Gen 9, 14.5M-param transformer). Read the CODE
  for encoder / action-space / reward / PPO-hparam questions; its README and Reddit thread
  describe an older system and several public claims fail against it. Provenance, verified
  numbers and two known defects: `prior_work/README.md`, ps-ppo entry — read that first.
- Server: `cd showdown && node pokemon-showdown start --no-security` — now `simulator: 4`
  (gitignored file; re-set it if the checkout is ever recreated).
- wandb offline since `e53323a`; `runs/*/history.csv` via `extract_history.py`. Stage-0 for
  concurrent seeds; ≥5-min runs in the maintainer's terminal; clean tree at every launch;
  commit docs BEFORE launches; handed-over command sets go in bash scripts under session tmp.
- Run artifacts: `runs/showdown_r512_lra_s{0,1,2}` (P5b), `runs/showdown_scratch12m_s*`,
  clones `runs/bc_p4_512{,_40k}_s{0,1,2}`. `runs/showdown_tput_w*` are disposable.
