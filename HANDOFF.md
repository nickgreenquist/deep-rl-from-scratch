# Handoff — written 2026-08-01 on explicit request; fold in, then restore the stub

Per CLAUDE.md § Plan and status: the session log is CURRENT — the 2026-07-31
design-session entry and the 2026-08-01 campaign entry carry every verdict,
number and provenance detail. This file is a map plus the one open decision.
Read, verify, restore the stub.

## State
- Clean and PUSHED at `798c306` (origin in sync). 308 tests green including
  BOTH live-server tests, run twice on a free server — the old
  `test_full_episode_contract_against_live_server` flake is CLEARED.
- Milestone-3 machinery is built, reviewed (three-Opus adversarial pass, see
  the 2026-07-31 entry) and validated live: PoolPlayer (seat-2 pool adapter),
  `init_from` warm start, `selfplay/winrate_anchor|winrate_latest|anchor_games`
  metrics, `--opponent-checkpoint` cross-play on eval_checkpoint.py,
  Showdown `fixed_mix` hard-reject.

## Where milestone 3 stands (2026-08-01 entry has all numbers)
- Run 1 (warm-started from the 12M/0.408 ckpt, 6M continuation, 3 seeds/arm):
  **self-play NOT credited at matched init + budget.** The load-bearing fact
  is R4: windowed `winrate_anchor` ≈ 0.5 in EVERY 1M window on EVERY seed —
  the learner never diverged from its frozen parent in either direction (no
  improvement, no forgetting). Cross-play SP-vs-CT 0.501 (dead even). All
  health gates passed (no stall equilibrium, entropy fine).
- **New best headline: the CONTROL arm — 0.432 ± 0.018 pooled, the first
  3-seed number in the lineage** (continued fixed-bot, 18M cumulative, curve
  still creeping, seed-std 0.010). Milestone-2 bar (0.5) not met by either arm.
- Asymmetry worth keeping in the narrative: CT's +0.024 on the anchor does
  NOT show up head-to-head — it reads as heuristics-specialization, not
  generalizable strength.

## Next: the run-3 DESIGN DECISION (open — maintainer has not chosen)
Two candidates on the table (end of the 2026-08-01 entry):
1. Pre-registered tree branch ("R1 flat and R3 flat"): from-scratch self-play
   12M×3, the narrative arm; expected value pre-registered 0.20–0.35, budget
   caveat ~6% of H&L's 192M learner transitions.
2. `latest_prob` 0.8→0.5 at 6M×3 (~3 h wall): the tree predates knowing the
   split would be CT +0.024 / SP exactly +0.000, and 80% mirror-vs-near-
   current battles is the plainest mechanism candidate for a ZERO gradient.
   Prior session's recommendation on record: this first, narrative arm queued.
Also live as bar-chasers independent of self-play: capacity step 2, or more
fixed-bot budget (the 18M curve is still creeping at +0.024/6M — price the
diminishing returns before committing).

## Read before proposing anything
1. SESSION_LOGS 2026-08-01 (campaign verdicts) and 2026-07-31 milestone-3
   design entry (three-review kills: no facade, no heuristics anchor ever —
   it is the eval bot; fixed_mix is Connect4-only; stall-equilibrium gate R0).
2. configs/showdown_sp6m.yaml header — the full R0–R5 pre-registration
   pattern; reuse it for run 3.
3. Memory file `milestone3-read-priorities` — maintainer's standing read
   discipline: windowed winrate_anchor (differenced via anchor_games, never
   the cumulative average) is the primary in-flight signal; eval rungs are
   noise at SE ≈ 0.05; the matched control is what makes results causal.

## Parked / small items
- wandb prompts for login in any fresh shell (no credentials on this machine;
  all campaigns run ambient-offline). Workaround: `export WANDB_MODE=offline`
  per shell — REQUIRED before any wandb-logger launch. To-do: make offline
  the code default in rl/common/logging.py (one line + test), not yet done.
- Concurrency headroom levers, priced but unbuilt (build when the lever queue
  has 2+ reads waiting): `simulator: 4` in showdown/config/config.js (one
  line, may also claw back the −23% at 3-wide), per-run `server_configuration`
  passthrough (~3 lines; boot precedent in showdown_throughput.py). Any new
  width needs a Stage-0-style re-probe first; wandb stub caveat stands at
  6-wide.
- Measured throughputs for planning: self-play 3-wide ~553 steps/s
  (evals included, ~8 s/5 rungs — reset latency is evidently small), solo
  717; fixed-bot 3-wide ~600; facade permanently parked (solo gate passed).
- BC diagnostic still parked (GO-WITH-CAVEATS, PLAN.md); `init_from` — its
  needed seam — now exists.

## Operational (unchanged)
- Server: `cd showdown && node pokemon-showdown start --no-security`.
- Stage-0 pattern for concurrent seeds; throughput never through Claude
  tooling; steps/s from meta.yaml → checkpoint.pt mtimes (rung ckpts include
  eval time); clean tree at every launch; commit docs BEFORE launches.
- Campaign artifacts: runs/showdown_{sp6m,cont6m}_s{0,1,2}/ —
  final_eval_heur_1000.json, xplay_*.json, wandb offline histories
  (scripts/extract_history.py reads them).
