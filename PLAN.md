# Plan

Living working doc for this repo: per-phase definitions of done (checkboxes = current status), open decisions, and a session log. `README.md` is the public summary; `CLAUDE.md` is the working rules. Update this file as work lands.

## Phase 0 — shared harness + tabular warmup (in progress)

Goal: a training harness reused unchanged by every later algorithm, proven two ways — a random-policy pipeline check, then one agent that genuinely learns.

- [x] Repo scaffold: directory structure, README, `.gitignore`, pinned `pyproject.toml`; `deep-rl` conda env created and deps validated
- [ ] Random-policy agent runs end-to-end on `CartPole-v1` through `python -m rl.train` (env → rollout → logging → periodic eval → checkpoint)
- [ ] Tabular Q-learning on `FrozenLake-v1` (slippery 4x4, lookup table — no net) visibly learns: eval return climbs above the random baseline (~0.015 success rate)
- [ ] Logging wired with the locked metric names from CLAUDE.md
- [ ] Seeding utility covering Python / NumPy / torch / env
- [ ] Eval protocol: fixed eval seeds, N episodes, deterministic policy, mean ± std
- [ ] Checkpoint save/load stub
- [ ] `tests/test_harness.py`: CartPole sanity test (a few hundred steps, asserts the loop completes) — stays green for the life of the project; it's the known-good path when a reward curve goes flat

Build order: seeding → config → logger → env factory → agent interface → buffers (interface only) → eval → checkpoint → train loop → random agent → tabular Q-learning → sanity test.

Deliberately deferred: `rl/envs/wrappers.py` and `scripts/run.sh` (created when they have content); all deep RL algorithms (Phase 1+).

## Phase 1 — DQN (discrete track)

Replay buffer, target network, ε-greedy exploration; Double DQN / Dueling / n-step returns as config toggles. Benchmark on CartPole/LunarLander for sanity, then MinAtar (install `minatar` at phase start). Headline metric vs a published baseline.

Expect this phase to feel disproportionately hard — it's the first algorithm *and* the first heavy use of the harness. That's front-loaded difficulty, not a signal the project is off track.

## Phase 2 — PPO (both tracks)

GAE, clipped surrogate objective, entropy bonus, vectorized rollouts (the env-factory seam becomes real here). Runs discrete *and* continuous — the bridge between the two tracks and the likely capstone engine. Install `gymnasium[mujoco]` when the continuous track starts. Reference: "The 37 Implementation Details of Proximal Policy Optimization".

## Phase 3 — SAC (continuous track)

Twin Q critics, reparameterized stochastic actor, auto-tuned entropy temperature, soft target updates. Benchmarked against PPO on MuJoCo locomotion (HalfCheetah, Hopper, Walker2d).

## Phase 4 — capstone (env undecided)

Point the best spine algorithm at one substantial environment with published baselines, so the result stays legible even if it isn't SOTA. Runs on a rented cloud GPU; W&B merges local + cloud runs into one dashboard.

**Open decision — the env fork:** hard-exploration (Crafter, Montezuma's Revenge) vs generalization (Procgen). Current lean is **Procgen**: cleanest headline metric (train/test generalization gap), fast C++ envs that pair naturally with a PyTorch agent, and no JAX/PyTorch framework tension. Not locked; nothing in Phases 0–3 depends on it, so don't scaffold anything capstone-specific until it's decided.

## Session log

- 2026-07-21 — Repo scaffolded: structure, README, CLAUDE.md, `.gitignore`, pinned `pyproject.toml`. Initial commit.
- 2026-07-22 — Pushed to GitHub. Created `deep-rl` conda env; installed pinned deps and smoke-tested CartPole/FrozenLake/LunarLander. Added working-style and dev-env sections to CLAUDE.md. Retired the handoff doc into this file.
