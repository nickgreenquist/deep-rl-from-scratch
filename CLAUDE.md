# CLAUDE.md

Guide for Claude Code sessions on this repo. Read `PLAN.md` at session start for current status and next steps.

## What this project is

From-scratch deep RL in PyTorch, built as a portfolio piece over multiple months at ~10 hrs/week. Spine: DQN → PPO → SAC on a shared harness, benchmarked apples-to-apples, each phase independently shippable. Capstone: best algorithm on a substantial env vs a published baseline. The capstone environment is **undecided** (current lean: Procgen) — do not scaffold or assume anything capstone-specific.

## Hard rules

- **No RL libraries.** Never import or depend on Stable-Baselines3, RLlib, Tianshou, CleanRL, etc. Reading their source for reference is fine; depending on them is not.
- **Plan before editing.** State which files you'll create/change and why; wait for a go-ahead. Keep diffs clean and reviewable.
- **Small, single-purpose commits.** End every session at a green, committable state.
- **Minimal dependencies.** Stdlib where possible; config is a dataclass + YAML, no experiment frameworks. Pin versions in `pyproject.toml`.
- **CPU by default.** A device override flag exists, but do not rely on MPS — it's flaky for this workload. GPU appears only at the capstone (rented cloud instance).
- **This repo may go public.** Keep personal details (employer, etc.) out of committed files.

## Development environment

- **Always run in the `deep-rl` conda env** (`/opt/anaconda3/envs/deep-rl`, Python 3.13): `conda activate deep-rl`, or call `/opt/anaconda3/envs/deep-rl/bin/python` / `.../bin/pytest` directly. Never use `base` or `pytorch_env` — the latter belongs to an unrelated project.
- The repo is installed editable in that env, so `import rl` and `python -m rl.train` work as-is; tests run with `pytest tests/` from the repo root.
- Dependency changes go through `pyproject.toml` with exact pins, then `pip install -e ".[dev]"` — no ad-hoc `pip install`, no `conda install` into the env.
- Recreate from scratch if needed: `conda create -y -n deep-rl python=3.13`, then `pip install -e ".[dev]"` in it.

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

## Plan and status

`PLAN.md` (repo root) is the living working doc — per-phase definitions of done (checkboxes = current status), open decisions, and a session log. Update it as work lands.

## Working with the maintainer

- Deep ML/DL fluency (production PyTorch recommender systems background). Do not explain gradient descent, tensors, or PyTorch basics. RL specifically is new — do explain RL concepts and algorithm design choices when they first appear.
- Direct tone. Skip superlatives and filler.
- Sessions are short evening blocks; optimize for incremental, resumable progress.
- **Git:** commit only when asked; never commit+push in one command — commit, then ask before pushing.

## Working style

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

### 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

### 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

### 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```
