# Handoff — written 2026-08-04 ~05:45 on explicit request; fold in, then restore the stub

STATUS.md, PLAN.md and SESSION_LOGS.md are CURRENT through this entire session. The whole
throughput session is already durable in the 2026-08-04 log entry — do NOT re-derive it from
here. Tree clean at `f7d117b`; **TWO local commits are unpushed** (`8ae7796`, `f7d117b`) —
push decision open, ask first.

## FIRST ACTION — the loop-split instrument (design agreed in-session, not yet written)

STATUS "Next" item 1. The maintainer directed it after this session's load-bearing finding
(collection-only benchmarks overstate full-loop gain ~7x; the loop is update-and-encode
bound, not collection bound). Shape agreed in discussion, awaiting only a go-ahead:

- **Step 1 — `rl/train.py` only, env-agnostic:** three `perf_counter()` pairs per rollout
  (NOT per step) logging `time/collect_sec`, `time/update_sec`, `time/eval_sec` alongside the
  existing `time/steps_per_sec`. Both the vectorized and scalar paths. ~15 lines. Ship with a
  test pinning the no-op on CartPole. This alone answers whether update or collect dominates,
  which is the thing that contradicts or confirms the Phase-5 hardware note.
- **Step 2 — ONLY if step 1 says collect dominates:** decompose collect Showdown-side. Do NOT
  put an encode timer in shared code — `embed_battle` lives inside `ShowdownSingles`, and a
  `hasattr` branch in `rl/train.py` for one env is the pattern the masking contract bans.
  Instead re-run **measurement (a)** (`scripts/showdown_throughput.py a`), which already
  decomposes per-turn encode vs inference vs env gap but last ran on the 10-dim PLACEHOLDER
  encoder, before the real 611-dim one landed 2026-07-30. Script change, not a seam change.

**Two questions were put to the maintainer and are still unanswered** — get these before
writing code: (1) always-on timers, or flag-gated? (recommended: always-on; a flag is
speculative configurability and the overhead is microseconds per rollout); (2) is adding three
`time/*` keys to every run's W&B history acceptable under the locked-metric-names rule? (read
as prohibiting renames, not additions — but it is the maintainer's rule to confirm).

## Then, in order (already in STATUS "Next")

2. **BC-warm-start design session** — the stack (BC init + shaping + anneal verdict) as ONE
   pre-registered package. P5b's anneal is now a credited component of it.
3. 12M r512 extension decision — open. Annealed arm must be from-scratch; flat-vs-annealed at
   12M is the natural framing; ~5.2 h at 3-wide post-`simulator: 4`.

## Operational

- **Showdown server is UP** on :8000 and now runs `simulator: 4` (`showdown/config/config.js`
  line 111). That file is GITIGNORED — it will not show in `git status`, and it must be re-set
  if the `showdown/` checkout is ever recreated. It is the entire content of throughput work
  item (1); no `rl/` source changed this session.
- `runs/showdown_tput_w3_s{0,1,2}` + `w6_s{0..4}` (8 dirs, ~30 MB each) are DISPOSABLE — the
  numbers are all in the log entry. Delete when convenient.
- Launcher scripts live in `/Users/nickgreenquist/.claude/jobs/acc47a2b/tmp/` (`tput_lanes.sh`,
  `facade_price.sh`). That is the OLD session's job dir — absolute path, may not exist for a
  new session; both are reproducible from the log entry.
- **Startup-crash hazard is live and unmitigated** (STATUS watch items): a lane can die with
  SIGSEGV in torch lazy static init before writing any log or run dir, and a naive launcher
  reports success over a short-handed result. Any future multi-seed launcher must stagger lane
  starts and assert all W run dirs exist with complete histories before printing done.
- CLAUDE.md still carries the old "no forward model" Phase-4 sentence — maintainer knows; a
  one-line edit if/when they want it aligned with the revised PLAN premise.
- `data/bc_p4_*.npz` ≈ 3.9 GB still deletable (regen ~10 min).
