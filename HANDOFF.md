# Handoff — written 2026-07-31 on explicit request; fold in, then restore the stub

Per CLAUDE.md § Plan and status: the session log is CURRENT (every verdict
of the last two days is a committed dated entry — nothing here should need
folding), so this file is mostly a map. Read it, verify, restore the stub.

## State
- Clean at `361229c`. **NOT pushed since `dfdfbdc`** — ~11 local commits
  (encoder-era follow-ups through the mixture verdict). Offer a push,
  explicit go-ahead only.
- 294 offline tests green + 2 live-server tests. KNOWN FLAKE TO CLEAR:
  `test_full_episode_contract_against_live_server` timed out ONCE when
  pytest ran against a server busy with a training run (2026-07-30);
  passed before and since on a free server, but rerun the full suite with
  a free server once to confirm 296 green. Never run live tests while a
  training run owns :8000.
- All phase5_env mutations caught (7 real + 1 surviving control).

## Where milestone 2 stands (all numbers = locked protocol: FINAL
checkpoint, 1000 fresh battles, seeds past the training ladder)
- Bar: 0.5 vs SimpleHeuristicsPlayer. **Best so far: 0.408 ± 0.030**
  (`showdown_heur_512_s0`, [512,512], 12M steps) — curve still creeping at
  the wall, first unflattened curve of the phase.
- Lever ledger (grep the 2026-07-30/31 log entries): encoder = enabling
  (10→611 dims, observable-state only); budget +0.066 credited-then-
  exhausted (flat from 2.5M); **capacity the big one** ([64,64]→[512,512],
  0.324@6M→0.408@12M, Phase 4's "capacity never binds" formally scoped to
  small-obs regimes); distribution-via-fixed-bots ≤+0.03 NOT credited
  (70/20/10 MixturePlayer, 3 seeds, band read fired 'at/below') — which is
  what promoted milestone-3 self-play.
- Seed-std at this budget is tiny (0.008 over 3 seeds) — n=1 reads are
  usable, 3-seed concurrent reads (~2.6 h, Stage-0 pattern) are the default.

## Next: milestone-3 self-play — DESIGN SESSION with the maintainer
Recommended model for it: Fable/high (design), Sonnet/high ok for the
implementation after. Read, in order, before proposing anything:
1. PLAN.md § Phase 5 — API corrections: the opponent lives OUTSIDE
   poke-env's env, entering via `SingleAgentWrapper(env, opponent)`; what
   Phase 4 transfers is the learner-facing contract, not the location.
   The SEAT-2 PLAYER ADAPTER (a poke-env Player driving the second seat
   from our policy snapshots) is the one piece never written.
2. SESSION_LOGS 2026-07-30 async-review entry — decisions of record that
   CONSTRAIN this design: async vec envs SHELVED (Stage-0 concurrent runs
   won); the pool-identity seam (make.py:24-29, one shared object across
   sub-envs) is load-bearing and any subprocess collection breaks it
   silently; the LOCKSTEP FACADE is the collection architecture of record
   for milestone 3 (in-process, preserves pool identity, batches opponent
   forwards); measurement (e) (challenge-to-first-request reset latency)
   still unmeasured and gates any facade work.
3. SESSION_LOGS 2026-07-29 fork entry — the seam collector's three
   preconditions if anyone re-proposes it; the 2026-07-31 mixture entry —
   why fixed-bot distribution is exonerated.
4. rl/selfplay/pool.py — Phase-4 pool (latest_prob 0.8, fixed_mix 0.05,
   PFSP): the machinery that transfers; it was validated on Connect 4.
- Design tension to surface early: pool opponents are policy SNAPSHOTS
  (need forward passes on seat 2 every turn) — that is where the facade's
  opponent-batching pays and where per-battle MixturePlayer-style
  delegation does not scale. Eval anchor stays PURE heuristics
  (make_eval_env doctrine); milestone-2 bar remains the scoreboard.
- Precedent worth re-reading in the design session: Huang & Lee §V-C
  (self-play forgetting is PUBLISHED in this domain) and Metamon's
  checkpoint-overfit finding — both already summarized in PLAN.md.

## Parked / small items
- BC diagnostic: scoping GO-WITH-CAVEATS (PLAN.md § Phase 5 BC paragraph +
  2026-07-30 log entry): ~109k-replay corpus, parser-from-scratch is the
  work, opponent-HP must round to /100. Parked until wanted.
- Replay watching works: `scripts/watch.py <ckpt>` on any Showdown
  checkpoint → replay HTMLs in runs/<run>/replays/ (p1 = our agent; open
  in Chrome). Offered but not built: readable seat account names.
- Light Screen volatile is unrepresented in the encoder (poke-env 0.15.0
  parses it to Effect.UNKNOWN) — documented in rl/envs/showdown.py.
- Async branch: if ever un-shelved, the full pre-registered protocol
  (stages A–E) is in the 2026-07-30 async-review entry; drop-don't-merge
  if the ≥2x gate fails.

## Operational
- Server: `cd showdown && node pokemon-showdown start --no-security`.
- Concurrent seeded runs: the Stage-0 pattern — one `&`-joined line +
  `wait`, ~-20%/run at 3-wide, do NOT extrapolate past 3 without re-probing;
  logger wandb is fine at 3-wide (no stub dirs observed) but the
  2026-07-24 stub caveat stands at 6-wide.
- Throughput measurements NEVER through Claude tooling (10x-slowdown rule);
  read steps/s from meta.yaml→checkpoint.pt mtimes.
- Watcher pattern for unattended finishes: background poll for the final
  ckpt file, then run `scripts/eval_checkpoint.py <ckpt> --episodes 1000
  --out <rundir>/final_eval_heur_1000.json` and log the verdict.
- Clean tree at every launch; commit docs BEFORE the maintainer launches.
