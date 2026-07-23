# Plan

Living working doc for this repo: per-phase definitions of done (checkboxes = current status), open decisions, and a session log. `README.md` is the public summary; `CLAUDE.md` is the working rules. Update this file as work lands.

## Phase 0 — shared harness + tabular warmup (complete)

Goal: a training harness reused unchanged by every later algorithm, proven two ways — a random-policy pipeline check, then one agent that genuinely learns.

- [x] Repo scaffold: directory structure, README, `.gitignore`, pinned `pyproject.toml`; `deep-rl` conda env created and deps validated
- [x] Random-policy agent runs end-to-end on `CartPole-v1` through `python -m rl.train` (env → rollout → logging → periodic eval → checkpoint)
- [x] Tabular Q-learning on `FrozenLake-v1` (slippery 4x4, lookup table — no net) visibly learns: eval return climbs above the random baseline (~0.015 success rate) — final eval 0.67 ± 0.47 vs 0.02 random on the same eval episodes (optimal ≈ 0.74)
- [x] Logging wired with the locked metric names from CLAUDE.md
- [x] Seeding utility covering Python / NumPy / torch / env
- [x] Eval protocol: fixed eval seeds, N episodes, deterministic policy, mean ± std
- [x] Checkpoint save/load stub
- [x] `tests/test_harness.py`: CartPole sanity test (a few hundred steps, asserts the loop completes) — stays green for the life of the project; it's the known-good path when a reward curve goes flat
- [x] `scripts/watch.py`: render a checkpointed policy live (`python scripts/watch.py runs/<run>/checkpoint.pt`) — pulled forward from Phase 1 once a learned FrozenLake policy existed to watch; toy-text envs render graphically via pygame, contrary to the earlier text-grid assumption

Build order: seeding → config → logger → env factory → agent interface → buffers (interface only) → eval → checkpoint → train loop → random agent → tabular Q-learning → sanity test.

Deliberately deferred: `rl/envs/wrappers.py` and `scripts/run.sh` (created when they have content); `rl/buffers/base.py` (no consumer yet — the random agent takes no buffer; the interface lands with its first user); all deep RL algorithms (Phase 1+).

## Benchmark protocol (Phases 1–4)

Every headline result comes from **≥3 independent training seeds** (5+ for the capstone), reported as mean ± std across seeds — deep RL is brittle with respect to random seed (per Spinning Up), so single-run numbers don't count. The per-run eval protocol (fixed eval seeds, deterministic policy, N episodes) is unchanged.

## Phase 1 — DQN (discrete track)

Replay buffer, target network, ε-greedy exploration; Double DQN / Dueling / n-step returns as config toggles. Benchmark on CartPole/LunarLander for sanity, then MinAtar (install `minatar` at phase start). Headline metric vs a published baseline.

**On-ramp (first step of the phase): linear Q-learning on CartPole.** The DQN skeleton with a zero-hidden-layer network (`nn.Linear(4, 2)`) — a linear model borrowing all the same infra. Isolates the one conceptual leap from Phase 0 (shared weights generalize across states; the same TD error now takes a gradient step instead of a table-cell nudge) and meets the deadly triad — function approximation + bootstrapping + off-policy, where tabular convergence guarantees evaporate — *before* the target network and replay buffer arrive as its stabilizers. Then add the hidden layer and it's DQN. Optional half-hour calibration first: random search over linear policies solves CartPole outright — a reminder that "solves CartPole" is a low bar, which is why the headline benchmark is MinAtar.

**Visuals (starts here, grows with the project):** better visuals serve three purposes — learning (seeing what a policy actually does), validation (a rendered rollout catches bugs metrics hide), and eventual sharing/posts (this repo is a portfolio piece). First deliverable, once DQN produces a policy worth showing: a `--record` path or `scripts/record.py` using `render_mode="rgb_array"` to save rollouts as annotated GIFs/videos (episode/step/return stamped on frames — cheap once frames are in hand) for the README and posts. Explicitly rejected: HUD overlays in the live `render_mode="human"` window (gymnasium exposes no API for it; the pygame workarounds are fragile) — `watch.py`'s terminal narration covers live viewing.

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
- 2026-07-22 (later still) — **Phase 0 complete.** Tabular Q-learning on slippery FrozenLake: `QLearningAgent` (ε-greedy, per-step Bellman update, ε annealed over first half of training), `frozenlake_q.yaml`, and the train loop's first real `update()` call (per-step, on the fresh transition; bootstraps through truncation but not termination). 200k steps → eval 0.67 ± 0.47 vs 0.02 random baseline on identical eval episodes; checkpoint restore reproduces 0.67 exactly. Also pulled `scripts/watch.py` forward from Phase 1 (render_mode passthrough in the env factory + checkpoint-driven agent rebuild) — verified live on the FrozenLake policy. Next: Phase 1 DQN (replay buffer → `rl/buffers/base.py` gets its first consumer).
- 2026-07-22 (Phase 1 start) — **Linear-Q on-ramp landed**: `rl/buffers/base.py` (thin `Buffer` ABC) + `rl/buffers/replay.py` (NumPy ring buffer), `rl/networks/mlp.py` (`hidden_sizes=[]` → single linear layer), `rl/agents/dqn.py` (online + target net, ε-greedy, Huber TD loss, agent-owned replay; `update()` stores the fresh transition then trains on a sampled batch, so the train loop stayed untouched apart from registering `algo: dqn`), `configs/cartpole_linear_q.yaml`, DQN smoke test in `test_harness.py`. First live contact with the deadly triad, on schedule: at lr 1e-3 the linear net **diverges** — Q predictions climb to ~281 vs the ~100 theoretical ceiling, eval 9.45 (below the ~22 random baseline). Same skeleton with `hidden_sizes: [64, 64]` at that lr solves CartPole (eval 500 ± 0, Q-pred mean ~104 ≈ the true ceiling — skeleton validated, not a bug). Committed config uses lr 1e-4, where linear-Q learns modestly (eval 46.5 ± 14.9, single seed, pipeline proof not a headline number). Next: promote to DQN proper (`hidden_sizes: [64, 64]` config), then Double/Dueling/n-step toggles and the multi-seed CartPole/LunarLander benchmark.
- 2026-07-22 (DQN proper) — **Toggles + sanity benchmark landed.** Double DQN / dueling / n-step as config toggles, defaults off (`rl/agents/dqn.py`: `NStepAccumulator` assembling n-step transitions with per-transition `gamma^m` discounts stored in the buffer; `rl/networks/mlp.py`: `DuelingMLP`). Transition tuple grew to include `truncated` (n-step must flush at every episode boundary, not just termination — both agents updated). `--seed`/`--run-name` CLI overrides make multi-seed runs a shell loop. `tests/test_dqn.py`: hand-computed n-step boundary cases + all-toggles smoke run (9 tests green). 3-seed vanilla benchmark, final `eval/return_mean` mean ± std across seeds: **CartPole 221 ± 243** (per-seed 500 / 105.5 / 58.2 — every seed reaches ~500 mid-run, then oscillates; Q-preds all sit at the ~100 ceiling while realized returns vary, i.e. textbook policy churn / overestimation that Double DQN exists to damp), **LunarLander 182 ± 55** (241.8 / 171.4 / 132.7; seed 0 clears the 200 "solved" bar, all still trending up at 300k steps). Numbers are the vanilla baseline for MinAtar-phase ablations, not headlines. Next: MinAtar (install `minatar`), toggle ablations, headline vs published baseline.
