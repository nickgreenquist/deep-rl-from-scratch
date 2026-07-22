# Plan

Living working doc for this repo: per-phase definitions of done (checkboxes = current status), open decisions, and a session log. `README.md` is the public summary; `CLAUDE.md` is the working rules. Update this file as work lands.

## Phase 0 — shared harness + tabular warmup (in progress)

Goal: a training harness reused unchanged by every later algorithm, proven two ways — a random-policy pipeline check, then one agent that genuinely learns.

- [x] Repo scaffold: directory structure, README, `.gitignore`, pinned `pyproject.toml`; `deep-rl` conda env created and deps validated
- [x] Random-policy agent runs end-to-end on `CartPole-v1` through `python -m rl.train` (env → rollout → logging → periodic eval → checkpoint)
- [ ] Tabular Q-learning on `FrozenLake-v1` (slippery 4x4, lookup table — no net) visibly learns: eval return climbs above the random baseline (~0.015 success rate)
- [x] Logging wired with the locked metric names from CLAUDE.md
- [x] Seeding utility covering Python / NumPy / torch / env
- [x] Eval protocol: fixed eval seeds, N episodes, deterministic policy, mean ± std
- [x] Checkpoint save/load stub
- [x] `tests/test_harness.py`: CartPole sanity test (a few hundred steps, asserts the loop completes) — stays green for the life of the project; it's the known-good path when a reward curve goes flat

Build order: seeding → config → logger → env factory → agent interface → buffers (interface only) → eval → checkpoint → train loop → random agent → tabular Q-learning → sanity test.

Deliberately deferred: `rl/envs/wrappers.py` and `scripts/run.sh` (created when they have content); `rl/buffers/base.py` (no consumer yet — the random agent takes no buffer; the interface lands with its first user); all deep RL algorithms (Phase 1+).

## Benchmark protocol (Phases 1–4)

Every headline result comes from **≥3 independent training seeds** (5+ for the capstone), reported as mean ± std across seeds — deep RL is brittle with respect to random seed (per Spinning Up), so single-run numbers don't count. The per-run eval protocol (fixed eval seeds, deterministic policy, N episodes) is unchanged.

## Phase 1 — DQN (discrete track)

Replay buffer, target network, ε-greedy exploration; Double DQN / Dueling / n-step returns as config toggles. Benchmark on CartPole/LunarLander for sanity, then MinAtar (install `minatar` at phase start). Headline metric vs a published baseline.

Expect this phase to feel disproportionately hard — it's the first algorithm *and* the first heavy use of the harness. That's front-loaded difficulty, not a signal the project is off track.

Also in this phase: a watch script (`scripts/watch.py`: load a checkpoint, render the policy with `render_mode="human"` via a passthrough in the env factory) — deferred from Phase 0 because it first pays off with a learned policy on a visual env (CartPole/LunarLander; FrozenLake renders as a text grid).

## Phase 2 — PPO (both tracks)

GAE, clipped surrogate objective, entropy bonus, vectorized rollouts (the env-factory seam becomes real here). Runs discrete *and* continuous — the bridge between the two tracks and the likely capstone engine. Install `gymnasium[mujoco]` when the continuous track starts. Reference: "The 37 Implementation Details of Proximal Policy Optimization".

Optional on-ramp, decide at phase start: a one-evening REINFORCE/VPG on CartPole (~80 lines, not a benchmarked milestone) to meet the policy-gradient core in isolation before PPO stacks GAE + clipping + vectorization on top.

## Phase 3 — SAC (continuous track)

Twin Q critics, reparameterized stochastic actor, auto-tuned entropy temperature, soft target updates. Benchmarked against PPO on MuJoCo locomotion (HalfCheetah, Hopper, Walker2d).

## Phase 4 — capstone (env undecided)

Point the best spine algorithm at one substantial environment with published baselines, so the result stays legible even if it isn't SOTA. Runs on a rented cloud GPU; W&B merges local + cloud runs into one dashboard.

**Open decision — the env fork:** hard-exploration (Crafter, Montezuma's Revenge) vs generalization (Procgen). Current lean is **Procgen**: cleanest headline metric (train/test generalization gap), fast C++ envs that pair naturally with a PyTorch agent, and no JAX/PyTorch framework tension. Not locked; nothing in Phases 0–3 depends on it, so don't scaffold anything capstone-specific until it's decided.

## Session log

- 2026-07-21 — Repo scaffolded: structure, README, CLAUDE.md, `.gitignore`, pinned `pyproject.toml`. Initial commit.
- 2026-07-22 — Pushed to GitHub. Created `deep-rl` conda env; installed pinned deps and smoke-tested CartPole/FrozenLake/LunarLander. Added working-style and dev-env sections to CLAUDE.md. Retired the handoff doc into this file. Checked the plan against OpenAI Spinning Up: added the multi-seed benchmark protocol and the optional VPG on-ramp.
- 2026-07-22 (later) — Random-policy milestone: logger seam (W&B + TensorBoard), env factory, `Agent` interface (+ `state_dict` for checkpointing), `RandomAgent`, fixed-seed eval, checkpoint stub, `rl.train` entry point, CartPole sanity test. Verified: `pytest` green; full 5000-step `cartpole_random.yaml` run through W&B (offline) hit the ~22 random baseline (eval 24.7 ± 11.1). Post-review fix: eval episode seeds are now constants (`EVAL_SEED_OFFSET + episode`), decoupled from the training seed so multi-seed benchmark runs share one eval distribution. Next: tabular Q-learning on FrozenLake.
