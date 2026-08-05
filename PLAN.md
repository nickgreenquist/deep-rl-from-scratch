# Plan

Working doc for this repo: the benchmark protocol and per-phase digests of completed work
(what was built, and what still binds). `STATUS.md` is the current-state board (read it
first); `README.md` is the public summary; `CLAUDE.md` is the working rules.

Locked specs for completed Phases 0–4 moved verbatim to `PLAN_ARCHIVE.md` (2026-08-01). Any
reference to a Phase 0–4 spec — here, in code/test docstrings, or in README.md — resolves
there. Grep it; never read it whole. Before re-opening any decision named in a "Still binds"
line below, read that phase's archived spec first: those decisions were paid for, and a
summary is not grounds to re-open them.

## Benchmark protocol (Phases 1–4)

Every headline result comes from **≥3 independent training seeds**, reported
as mean ± std across seeds — deep RL is brittle with respect to random seed (per Spinning Up), so
single-run numbers don't count. The per-run eval protocol (fixed eval seeds, deterministic policy, N
episodes) is unchanged.

Execution: any run longer than ~5 minutes — i.e. every real training run (a 5M-step MinAtar run is
~55 min) — is launched by the maintainer in their own terminal; Claude hands over the exact command
(env vars included) and picks up from the logged output and checkpoints. Learned on a prior project:
training launched through Claude Code tooling runs ~10x slower. Short smokes and tests stay
in-session.

## Phase 0 — shared harness + tabular warmup (COMPLETE)

**Built:** the training harness every later algorithm reuses unchanged — `python -m rl.train`
end to end (env factory → rollout → logging → periodic eval → checkpoint), seeding utility,
fixed-seed eval protocol, checkpoint save/load, `scripts/watch.py`; proven by a random-policy
pipeline check and tabular Q-learning on FrozenLake (final eval 0.67 ± 0.47 vs 0.02 random).
Full spec: `PLAN_ARCHIVE.md` §Phase 0.

**Still binds:** `tests/test_harness.py` stays green for the life of the project — the
known-good path when a reward curve goes flat. The locked metric names, eval protocol, and
agent interface live in CLAUDE.md's architecture invariants.

## Phase 1 — DQN, discrete track (COMPLETE)

**Built:** DQN with replay buffer, target network, ε-greedy, and Double/Dueling/n-step config
toggles (`rl/agents/dqn.py`); benchmarked on all 5 MinAtar games at 5M steps × 3 seeds, 5/5
replicate the paper (see README results). Full spec: `PLAN_ARCHIVE.md` §Phase 1.

**Still binds:**
- No gradient clipping in DQN — audited, deliberate, and changing it breaks comparability
  with the 63-run Phase 1 campaign (§Phase 1 for the audit against the PyTorch DQN tutorial).
- Single-threaded torch for tiny nets — the threading pathology is measured three independent
  times in this repo (MinAtar DQN, Connect 4 PPO, the SAC probe); default to `torch_threads: 1`.

## Phase 2 — PPO, both tracks (COMPLETE)

**Built:** PPO discrete and continuous — vectorized collection seam, `RolloutBuffer` + GAE,
discrete PPO on MinAtar (beats DQN on 3 of 5 games), continuous PPO on MuJoCo with running
obs/reward normalization landed as checkpointed harness surface; REINFORCE on-ramp. Chunk
3/4/5 specs locked by three-lens reviews and implemented as written, no deviations. Full
spec: `PLAN_ARCHIVE.md` §Phase 2.

**Still binds:**
- The four canonical continuous forks — unbounded `Normal` + `ClipAction`; state-independent
  free `log_std`; running obs stats, wrapper-side and checkpointed; SB3 reward-norm operation
  order — chosen for the PPO-vs-SAC hold-everything-fixed discipline (§Phase 2).
- Normalizer state is harness surface: checkpointed, restored through one helper, and
  `train()` raises on a normalize flag for scalar-path algos (§Phase 2).
- Advantage recomputed once per rollout (never per epoch), MSE critic loss, per-minibatch
  advantage norm — audited against the TorchRL tutorial and kept canonical (§Phase 2).

## Phase 3 — SAC, continuous track (COMPLETE)

**Built:** SAC (twin Q, squashed Gaussian with the stable log-prob form, auto-tuned α, Polyak
via `rl/common/polyak.py`), validated against published SAC anchors before the PPO comparison
(M3), then benchmarked against Phase 2 PPO on 3 MuJoCo envs; 64×64-Tanh architecture ablation
and PPO-raw control campaign. Full spec: `PLAN_ARCHIVE.md` §Phase 3.

**Still binds:**
- SAC runs on RAW envs while PPO's MuJoCo configs normalize — the confound is closed by the
  PPO-raw control, and any cross-algorithm claim must respect the per-anchor protocol table
  (§Phase 3).
- The `(B,)`-shape guards and their tests are load-bearing — the `(B,B)` broadcast traps
  (log-prob flatten, critic target) train plausibly and only detonate off the gate env
  (§Phase 3).
- Snapshots/copies must never alias live tensors — the `log_alpha` rebind failure class;
  round-trip persistence tests must go through `torch.save`, not in-memory (§Phase 3).

## Phase 4 — Connect 4 self-play on-ramp (COMPLETE)

**Built:** the self-play machinery, validated in isolation — `Connect4Env` with open_spiel
differential oracle, opponent pool with strided retention (`rl/selfplay/`), negamax solver
validated on Pons' benchmark (4,400/4,400), Bradley-Terry Elo tournament harness, the
forgetting demonstration (naive vs pool, one-key diff), the probe-lever campaign, and the
supervised-on-solver-labels diagnostic. Full spec: `PLAN_ARCHIVE.md` §Phase 4.

**Still binds (live Phase-5 priors):**
- ~50% is the EQUILIBRIUM of self-play, not a failure; learning is judged against fixed
  external anchors, never the pool, and `eval_opponent` never resolves to a pool member
  (§Phase 4 pre-registered expectations — carried verbatim into Phase 5 below).
- PPO hyperparameters do not port across episode-length/reward regimes; γ=1.0 with a
  terminal-only ±1 reward; lr anneal off in self-play — it would suppress the phenomenon
  under study (§Phase 4 config).
- Pool snapshots are deepcopies — `agent.state_dict()` aliases live tensors; frozen opponents
  must be probed as actually frozen (§Phase 4 self-play machinery).
- Encoder/architecture ceilings are diagnosed with a supervised instrument, never inferred
  from RL curves — the Phase 4 supervised diagnostic is the pattern (§Phase 4; SESSION_LOGS
  2026-07-29).
- Best rung ≠ final rung — report both; the checkpoint ladder exists for this (§Phase 4).

## Session log

Moved to `SESSION_LOGS.md` (2026-07-29) — every dated entry, verbatim.
Any "see the session log" reference in this file resolves there. Append
new entries THERE as work lands, not here.
