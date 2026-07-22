# CLAUDE.md

Guide for Claude Code sessions on this repo. `RL_PROJECT_HANDOFF.md` (repo root, deliberately untracked) is the founding doc — if present, read it before starting work.

## What this project is

From-scratch deep RL in PyTorch, built as a portfolio piece over multiple months at ~10 hrs/week. Spine: DQN → PPO → SAC on a shared harness, benchmarked apples-to-apples, each phase independently shippable. Capstone: best algorithm on a substantial env vs a published baseline. The capstone environment is **undecided** (current lean: Procgen) — do not scaffold or assume anything capstone-specific.

## Hard rules

- **No RL libraries.** Never import or depend on Stable-Baselines3, RLlib, Tianshou, CleanRL, etc. Reading their source for reference is fine; depending on them is not.
- **Plan before editing.** State which files you'll create/change and why; wait for a go-ahead. Keep diffs clean and reviewable.
- **Small, single-purpose commits.** End every session at a green, committable state.
- **Minimal dependencies.** Stdlib where possible; config is a dataclass + YAML, no experiment frameworks. Pin versions in `pyproject.toml`.
- **CPU by default.** A device override flag exists, but do not rely on MPS — it's flaky for this workload. GPU appears only at the capstone (rented cloud instance).
- **This repo may go public.** Keep personal details (employer, etc.) out of committed files.

## Architecture invariants

- Two tracks are first-class: discrete (DQN vs PPO) and continuous (PPO vs SAC). Never hardcode discrete-action assumptions in shared code.
- Agent interface (`rl/agents/base.py`): `act(obs, deterministic=False) -> action`, `update(batch) -> dict[str, float]`. DQN, PPO, and SAC must all fit it without contortions.
- Single entry point: `python -m rl.train --config configs/<run>.yaml`. Every algorithm plugs into it.
- Logging: W&B is the default backend, TensorBoard behind a flag as the offline fallback, both wrapped by a thin logger interface. No W&B calls in algorithm code.
- **Locked metric names** — reuse these exactly in every algorithm: `rollout/episode_return`, `rollout/episode_length`, `eval/return_mean`, `eval/return_std`, `time/steps_per_sec`, plus `loss/*` for per-algorithm losses.
- Evaluation: fixed eval seeds, N episodes, deterministic policy, mean ± std; kept separate from training rollouts.
- `tests/test_harness.py` (CartPole sanity test) must stay green for the life of the project — it is the known-good path when a reward curve goes flat, since in RL a bug and a bad hyperparameter look identical.
- Env factory (`rl/envs/make.py`) keeps a clean seam for vectorized envs — PPO needs them in Phase 2.
- Buffers: DQN needs a replay buffer (off-policy), PPO a rollout buffer (on-policy); `rl/buffers/base.py` must accommodate both patterns.

## Roadmap and status

| Phase | Deliverable | Status |
|-------|-------------|--------|
| 0 | Scaffold + shared harness; random agent on CartPole; tabular Q-learning on FrozenLake | scaffold committed 2026-07-21; harness not started |
| 1 | DQN from scratch (discrete track) | not started |
| 2 | PPO from scratch (both tracks; reference "The 37 Implementation Details of PPO") | not started |
| 3 | SAC from scratch (continuous track) | not started |
| 4 | Capstone (env TBD) | blocked on env decision — do not pre-build |

Update the status column as work lands.

## Working with the maintainer

- Deep ML/DL fluency (production PyTorch recommender systems background). Do not explain gradient descent, tensors, or PyTorch basics. RL specifically is new — do explain RL concepts and algorithm design choices when they first appear.
- Direct tone. Skip superlatives and filler.
- Sessions are short evening blocks; optimize for incremental, resumable progress.

## Phase 0 build order (next up)

seeding → config → logger → env factory → agent interface → buffers (interface only) → eval → checkpoint → train loop, then the random agent on `CartPole-v1`, tabular Q-learning on `FrozenLake-v1` (eval return must climb above the random baseline), and the CartPole sanity test. Full definition of done: `RL_PROJECT_HANDOFF.md` §3.
