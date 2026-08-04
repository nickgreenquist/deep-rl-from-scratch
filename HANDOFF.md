# Handoff — written 2026-08-03 on explicit request; fold in, then restore the stub

STATUS.md and the log are CURRENT through the P5 verdict (2026-08-03 entries:
push + P3 + P5 pre-reg, then P5 CREDITED). Tree clean at `b2f4be7`; commits
since the last push are local.

## FIRST ACTION — before folding anything or picking a direction

**Read `PHASE5_PRIOR_WORK_BRIEFING.md` (repo root, untracked)** — the
maintainer's external research findings from a separate no-repo-access
session (Wang thesis read in full + a flagged-unverified GitHub repo claim).
It SUPERSEDES the in-session web scan below, and it already corrects it on
one load-bearing point: Wang's actual result is **0.786 vs SH for the pure
network** (0.908 with MCTS, gen4randombattles, 1693 Elo) — the "~55% stuck"
figure this session quoted from a search snippet was a constant-lr ablation,
not the thesis result. Verify the briefing's claims per its own provenance
flags (Source B is unread by its author), discuss direction with the
maintainer, then take it from there; fold what survives into the log.
Do the verification with 2–3 **Opus high-effort subagents** (house pattern:
one per source/lens — e.g. read `Nebraskinator/ps-ppo` directly, re-verify
Wang's specifics, hunt for other pure-policy-vs-SH datapoints); the
maintainer approved that spend. Save xhigh/max effort for the design
session that follows, not the dig.

## Superseded-pending: this session's web calibration (do not treat as of-record)

One conversation produced context that was never logged (deliberately — the
maintainer's own findings take precedence): top gen1randombattle ladder
humans hold GXE ~78–92% (the format rewards skill; not a luck wash); an MIT
MEng thesis (Wang 2024, RL for randbats) reported getting stuck ~55% vs this
same SimpleHeuristicsPlayer, which it rates "beginner human" — so the unmet
0.5 bar ≈ matching a comparable academic effort; Metamon is out of reach on
data+compute (offline RL on ~1M human battles, transformers), not ideas, and
its self-play arm failed (§5.3, already cited in PLAN). Lever discussion with
the maintainer, unlogged: bigger model is evidence-dead (the clone), reward
shaping should be potential-based if tried, natural stack = BC init + r512
recipe + shaping.

## State / open decisions (all durable in STATUS.md — this is just the map)

- P5 CREDITED: rollout_steps 512 → 6M pooled 0.3923 vs control 0.3550,
  z=3.0; whole band shifted. 12M-extension decision OPEN (needs its own
  pre-registration). Other open decision: push (local commits).
- BC-warm-start design session is the queued next chapter (pre-registered
  meaning first — see stop rule, ratified 2026-08-02).
- heur_512 replication done (pooled 0.417 ± 0.009); wedge RESOLVED at 3
  seeds; README amended and public through `17ae11b`.

## Operational

- Monitoring cron loops: all stopped. Launch scripts in the session tmp dir
  (jobs/acc47a2b/tmp/): heur512_seeds.sh, r512_probe.sh — ephemeral.
- Showdown server may still be running from the probe (node, :8000).
- data/bc_p4_*.npz ≈ 3.9 GB still deletable (regen ~10 min).
