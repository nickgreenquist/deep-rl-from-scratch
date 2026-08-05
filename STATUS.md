# Status

Current-state board. Rewritten in place (never appended) as work lands — update it in the
same commit that appends a `SESSION_LOGS.md` entry; hard cap ~80 lines. If this file
conflicts with the newest session-log entry, the log wins — say so and fix this file.

## Where we are (updated 2026-08-05)

**The project is COMPLETE.** DQN, PPO and SAC are implemented from scratch on one shared
harness, benchmarked apples-to-apples against published baselines, and written up in
`README.md`. Phase 4 closes the suite with a Connect 4 self-play study. Nothing is in flight
and no runs are pending.

| Phase | Result |
|---|---|
| 0 — harness + tabular | Q-learning 0.67 success on slippery FrozenLake (random 0.02, optimal ~0.74) |
| 1 — DQN, discrete | reproduces the MinAtar paper's DQN on all 5 games |
| 2 — PPO, both tracks | beats DQN on 3 of 5 MinAtar games; reproduces the reference on MuJoCo |
| 3 — SAC, continuous | beats PPO on all three MuJoCo envs per sample, loses per minute |
| 4 — Connect 4 self-play | naive self-play forgets; the pool arm does not — measured on an exact-solver instrument stack |

Headline cross-phase finding: **SAC dominates per environment step and PPO dominates per
minute** — which algorithm is "better" depends entirely on whether samples or compute is the
scarce resource. Phase 4's tactical ceiling is the visited state distribution, not encoder
capacity.

## Next

Nothing scheduled. The repo is a finished portfolio piece; treat further work as new scope
and pre-register it the way `PLAN.md`'s benchmark protocol requires (≥3 seeds, mean ± std,
fixed eval seeds).

If you do pick something up, the standing invariants are in `CLAUDE.md` — action masking,
locked metric names, the single `python -m rl.train` entry point, and `tests/test_harness.py`
staying green for the life of the project.

## Watch items

- **`tests/test_harness.py` (CartPole) must stay green.** It is the known-good path when a
  reward curve goes flat, since in RL a bug and a bad hyperparameter look identical.
- `open_spiel` is **dev-only** and may only be imported as `pyspiel.load_game("connect_four")`,
  never `open_spiel.python.*` — the narrow carve-out from the no-RL-libraries rule, pinned by a
  test that greps the tree.
- Seed variance is large in self-play (~65 Elo within-arm at Phase 4 scale); every headline
  claim rests on ≥3 seeds with bootstrap intervals, and close calls are called ties.

## Operational

- Env: `/opt/anaconda3/envs/deep-rl` (Python 3.13). Never `base` or `pytorch_env`.
- Tests: `/opt/anaconda3/envs/deep-rl/bin/pytest tests/ -q` from the repo root.
- Runs: `python -m rl.train --config configs/<run>.yaml`. Anything over ~5 minutes goes in the
  maintainer's terminal, not through Claude.
- wandb offline since `e53323a`; `runs/*/history.csv` via `scripts/extract_history.py`.
- Git: commit only when asked; never commit and push in one command.
