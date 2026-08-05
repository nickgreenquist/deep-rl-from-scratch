# CLAUDE.md

Guide for Claude Code sessions on this repo. At session start read `HANDOFF.md` if non-empty, then `STATUS.md` — that is the only mandatory read; everything else is on demand per "Plan and status" below.

## What this project is

From-scratch deep RL in PyTorch, built as a portfolio piece over multiple months at ~10 hrs/week. DQN → PPO → SAC on a shared harness, benchmarked apples-to-apples, each phase independently shippable. **Phase 4 closes the suite with a Connect 4 self-play on-ramp** (added 2026-07-25): the self-play loop, checkpoint pool and Elo harness, PPO-only and deliberately without MCTS, since tree search needs a forward model many interesting environments do not provide. The project is complete: all five phases are implemented, benchmarked and written up.

## Hard rules

- **No RL libraries.** Never import or depend on Stable-Baselines3, RLlib, Tianshou, CleanRL, etc. Reading their source for reference is fine; depending on them is not. **One narrow carve-out (Phase 4):** `open_spiel` is a pinned **dev-only** dependency used solely as a differential-test oracle for the Connect 4 board — it also ships `open_spiel.python.algorithms.{dqn,ppo,...}`, so tests may import `pyspiel.load_game("connect_four")` and **never** `open_spiel.python.*`. Pinned by a test that greps the tree.
- **Plan before editing.** State which files you'll create/change and why; wait for a go-ahead. Keep diffs clean and reviewable.
- **Small, single-purpose commits.** End every session at a green, committable state.
- **Minimal dependencies.** Stdlib where possible; config is a dataclass + YAML, no experiment frameworks. Pin versions in `pyproject.toml`.
- **CPU by default.** A device override flag exists, but do not rely on MPS — it's flaky for this workload. No phase assumes a GPU.
- **This repo may go public.** Keep personal details (employer, etc.) out of committed files.

## Development environment

- **Always run in the `deep-rl-from-scratch` conda env** (`/opt/anaconda3/envs/deep-rl-from-scratch`, Python 3.13): `conda activate deep-rl-from-scratch`, or call `/opt/anaconda3/envs/deep-rl-from-scratch/bin/python` / `.../bin/pytest` directly. Never use `base` or `pytorch_env` — the latter belongs to an unrelated project. **This repo owns a top-level package named `rl`, so it must not share an env with another project that does the same; the first `.pth` alphabetically wins and the loser is imported silently from the wrong tree.**
- The repo is installed editable in that env, so `import rl` and `python -m rl.train` work as-is; tests run with `pytest tests/` from the repo root.
- Dependency changes go through `pyproject.toml` with exact pins, then `pip install -e ".[dev]"` — no ad-hoc `pip install`, no `conda install` into the env.
- Recreate from scratch if needed: `conda create -y -n deep-rl-from-scratch python=3.13`, then `pip install -e ".[dev]"` in it. Verified 2026-08-05: a clean env from this `pyproject.toml` installs 65 packages and runs 278/278 green.

## Architecture invariants

- Two tracks are first-class: discrete (DQN vs PPO) and continuous (PPO vs SAC). Never hardcode discrete-action assumptions in shared code.
- Agent interface (`rl/agents/base.py`): `act(obs, action_mask=None, deterministic=False) -> action`, `update(batch) -> dict[str, float]`. DQN, PPO, and SAC must all fit it without contortions.
- Action masking is a harness contract (Connect 4: a full column is illegal): Discrete-action envs always emit `info["action_mask"]` — all-True via the `ActionMask` wrapper when nothing is illegal — and algorithms mask logits/Q through `rl/common/masking` (finite `-1e8` sentinel, never `-inf`; masked path always exercised, no `mask is None` branches in algorithm code). The value head is never masked; masking applies at eval time too.
- Single entry point: `python -m rl.train --config configs/<run>.yaml`. Every algorithm plugs into it.
- Logging: W&B is the default backend, TensorBoard behind a flag as the offline fallback, both wrapped by a thin logger interface. No W&B calls in algorithm code.
- **Locked metric names** — reuse these exactly in every algorithm: `rollout/episode_return`, `rollout/episode_length`, `eval/return_mean`, `eval/return_std`, `time/steps_per_sec`, `time/collect_sec`, `time/update_sec`, `time/eval_sec`, `eval/win_rate`, plus `loss/*` for per-algorithm losses and `selfplay/*` for opponent-pool diagnostics (Phase 4+; logged from `rl/train.py`, never from algorithm or pool code). `eval/win_rate` is the fraction of eval episodes the agent **won**, counted from an env-supplied `info["outcome"] ∈ {-1, 0, +1}` and **never from the sign of the return** — a reward-sign inversion would otherwise report 100% and pass its own detector (measured). Emitted only when `eval_win_rate` is set, so every pre-Phase-4 config and run is untouched.
- Evaluation: fixed eval seeds, N episodes, deterministic policy, mean ± std; kept separate from training rollouts.
- `tests/test_harness.py` (CartPole sanity test) must stay green for the life of the project — it is the known-good path when a reward curve goes flat, since in RL a bug and a bad hyperparameter look identical.
- Env factory (`rl/envs/make.py`) keeps a clean seam for vectorized envs — PPO needs them in Phase 2.
- Buffers: DQN needs a replay buffer (off-policy), PPO a rollout buffer (on-policy); `rl/buffers/base.py` must accommodate both patterns.

## Plan and status

Session start, in this order — stop as soon as you can act:

1. `HANDOFF.md` — only if non-empty (mid-handoff). A continuation note for a session resuming after a context clear: read it, fold anything durable into STATUS.md / SESSION_LOGS.md, restore the empty stub. Written only when the maintainer explicitly asks for a handoff. Nothing persists there.
2. `STATUS.md` — always. Current phase/milestone, last verdict with numbers, next actions in order, watch items, operational commands. Rewritten in place (never appended), hard cap ~80 lines; update it in the same commit that appends a session-log entry. If it conflicts with the newest session-log entry, the log wins — say so and fix STATUS.md.
3. `PLAN.md` — the spec: benchmark protocol and per-phase digests of completed work ("Built" / "What still binds"). Never read any doc whole "for context."
4. `SESSION_LOGS.md` — dated entries (findings, decisions, run records); append as work lands. Access pattern: `grep -n '^- 20' SESSION_LOGS.md` for the index of entry titles, then Read the chosen entry by offset/limit. Never a broad keyword grep — a term like "self-play" returns tens of KB.
5. `PLAN_ARCHIVE.md` — Phases 0–4 locked specs, moved verbatim, frozen. Grep it when touching that phase's code or before re-opening any locked decision; never read whole. Any "see PLAN.md" reference to a Phase 0–4 spec resolves here.

History questions ("what did we decide / measure about X, and why?") go to the `doc-archaeologist` subagent, not to direct reads — its reads cost this session nothing, and it returns the decision with a verbatim quote, date, and file:line. Before proposing anything named in a "What still binds" line, read that phase's archived spec first: those decisions were paid for, and a summary is not grounds to re-open them.

## Working with the maintainer

- Deep ML/DL fluency (production PyTorch recommender systems background). Do not explain gradient descent, tensors, or PyTorch basics. RL specifically is new — do explain RL concepts and algorithm design choices when they first appear.
- Direct tone. Skip superlatives and filler.
- Sessions are short evening blocks; optimize for incremental, resumable progress.
- **Runs longer than ~5 minutes go in the maintainer's terminal, not through Claude.** Hand over the exact command to run (env vars, config path, seed loop included), then read the resulting logs/checkpoints from disk and take it from there. Hard-learned on a prior project: PyTorch training launched through Claude Code tooling (Bash sessions, subagents, workflows) ran ~10x slower than the same command in a plain terminal. Short smokes and pytest stay in-session.
- **Handed-over commands: one command per fenced block, never a multi-line block.** Multi-line pastes often mis-execute in the maintainer's terminal. No inline `#` comments (interactive zsh doesn't parse them — a comment after `kill` was read as PIDs); explanation goes in prose around the block. State-changing steps (kill, rm) are separate blocks run one at a time; a set of runs meant to execute together is ONE line, `&&`-chained so a failure stops the rest. Don't reach for `&`/job control unless asked.
- **Wrap every handed-over command block in `<command>` / `</command>` sentinel lines** so its boundaries are unambiguous when scrolling or copying: `<command>` on its own line, then the fenced block, then `</command>`. The sentinels go OUTSIDE the fence, never inside it — nothing but the command itself may be copyable from the block.
- **Never edit the tree (even untracked files) while the maintainer may be LAUNCHING a run** — launches stamp `git_dirty`, and one untracked .md is enough to flip it (measured: it dirtied 8 of 9 lever-run stamps; see the 2026-07-29 lever entry in SESSION_LOGS.md). Never run mutation batteries while any maintainer process might import mutated source.
- **zsh traps in ANY command (handed-over or in-session):** `echo ===` is a glob error in zsh; inline `#` comments don't parse in interactive zsh.
- **Git:** commit only when asked; never commit+push in one command — commit, then ask before pushing.
- To watch a recorded GIF, tell the maintainer to drag the file into Chrome — Preview/`open` shows a static filmstrip, not the animation.

## Working style

**Tradeoff:** these guidelines bias toward caution over speed. For trivial tasks, use judgment.

- **Think before coding.** State assumptions explicitly; if multiple interpretations exist, present them — never pick silently. If something is unclear, stop, name it, ask. Say so when a simpler approach exists; push back when warranted.
- **Simplicity first.** Minimum code that solves the problem: no unrequested features, no abstractions for single-use code, no speculative configurability, no error handling for impossible scenarios. If 200 lines could be 50, rewrite.
- **Surgical changes.** Every changed line traces to the request. Don't refactor or "improve" adjacent code; match existing style. Remove orphans YOUR change created; mention pre-existing dead code, don't delete it.
- **Goal-driven execution.** Turn tasks into verifiable goals ("fix the bug" → a test that reproduces it, then passes; "refactor" → tests green before and after). For multi-step work, state a brief step → verify plan and loop until verified.
