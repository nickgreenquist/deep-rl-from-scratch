# Handoff — written 2026-08-03 ~21:20 on explicit request; fold in, then restore the stub

STATUS.md, PLAN.md and SESSION_LOGS.md are CURRENT through this entire session (five
2026-08-03 log entries: prior-work verification → direction set/P5b pre-reg → Wang fork
dig/scope decisions → throughput prioritized). Tree clean at `af4242b`; THREE local
commits are unpushed (`5074c1b`, `3e26b4d`, `af4242b`) — push decision open, ask first.

## FIRST ACTION — the P5b read (mechanical, fully pre-registered)

The P5b LR-anneal probe is RUNNING: launched ~20:56 on `5074c1b`, 3 seeds
(`showdown_r512_lra_s{0,1,2}`), ~2.9 h + in-script finals → expect artifacts ~00:00–00:30.
Take the read EXACTLY per the locked header in `configs/showdown_r512_lra.yaml`:
R0 gates first (late entropy [0.2,1.0] — a frozen late value is expected, lr→0; ties ≤4%;
steps/s within ~25% of 587), then PRIMARY: pooled 3-seed finals from
`runs/showdown_r512_lra_s{0,1,2}/final_eval_heur_1000.json` vs the r512 control
0.3923 ± 0.0089 — **CREDITED iff pooled ≥ 0.418**. Secondaries + amendment condition in the
header. Append the log entry, update STATUS (it sits at 83 lines — rebalance to ~80).

- **No `rl/` source edits until `lra_probe.sh` has FULLY exited** — the finals stage boots
  fresh python that imports `rl`. Check the script is done before touching source:
  launcher logs are at `/Users/nickgreenquist/.claude/jobs/acc47a2b/tmp/r512_lra_s{0,1,2}.log`
  (old session's job dir — absolute path, may still exist; the `runs/` artifacts are the
  ground truth either way).
- An annealed 6M ckpt CANNOT be warm-extended (train.py refuses; header has the note).

## Then, in order (all recorded in STATUS "Next" and the PLAN Phase 5 scope block)

1. **Throughput session** (maintainer-directed, precedes the BC chapter): server-port knob
   in the env seam + one Showdown server per lane; goal ≥685 steps/s per run restored at
   3-wide, lane-scaling W=3–6 through the full loop; facade go/no-go via measurement (e).
   Rationale of record: meta-level compounding (hypothesis turnover), PLAN scope block.
2. **BC-warm-start design session** — the stack (BC init + shaping + anneal verdict) as ONE
   pre-registered package. Scope relaxations already ratified in PLAN: pure-self-play
   retired; MCTS open as a follow-up phase (premise revised, serialization is upstream in
   our own showdown checkout — file:line refs in PLAN).
3. 12M flat-lr r512 extension decision — open, untaken.

## Operational

- Showdown server running on :8000 (shared by the probe — leave it alone until done).
- `prior_work/` holds the archived papers/forks (gitignored except README + briefing);
  `wang_fork_diffs.md` there is the maintainer-provided Wang fork extraction, read+verified.
- CLAUDE.md still carries the old "no forward model" Phase-4 sentence — maintainer knows;
  one-line edit if/when they want it aligned with the revised PLAN premise.
- data/bc_p4_*.npz ≈ 3.9 GB still deletable (regen ~10 min).
