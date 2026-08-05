# Handoff — written 2026-08-05 ~03:00 on explicit request; the capstone has moved

Everything below is CURRENT and COMMITTED through `5a32e5c`. Tree is clean. Nothing is pending,
nothing is running.

## State

- **P6 is complete and recorded — the last experiment run in this repo.** 12M flat vs annealed on
  r512, 6/6 lanes, R0 gates passed. Annealed pooled **0.4607** (0.449/0.451/0.482) vs flat
  **0.4330** (0.425/0.424/0.450); delta +0.0277, z = 2.16. **The anneal is credited at 12M, but
  narrowly** — it clears the pre-registered line by 0.003 where the 6M read cleared by double.
  0.4607 is the first RL result past the BC clone (0.453) and sits 0.028 under the measured
  SH-mirror ceiling of 0.489. Full read: last entry in `SESSION_LOGS.md`. README amended per the
  pre-stated condition (its old closing claim that the clone sat above every RL policy is now
  false and was rewritten).
- **The Pokémon Showdown capstone has moved** to
  `/Users/nickgreenquist/Documents/Projects/pokemon-showdown-rl`. That repo has `start.md`
  (bootstrap brief, carry-over manifest, distilled findings, landmines, a 7-step port verification)
  and `P6_RESULTS.md` (the final numbers and what they invalidate). A Fable session is doing the
  copy and setup there. **That is not your job.**
- **This repo reverts to its original scope**: from-scratch DQN, PPO and SAC on a shared harness.

## THE ONE TASK: execute `CAPSTONE_REMOVAL.md`

Read it. It is import-graph-verified, not guessed, and it separates:
- outright deletions (Showdown-only code, configs, data, artifacts, `prior_work/`)
- **files to EDIT, not delete** — `rl/envs/make.py`, `scripts/eval_checkpoint.py`,
  `scripts/watch.py`, `tests/test_selfplay_harness.py`, `pyproject.toml`. Deleting these breaks
  the spine.
- **shared code that looks capstone-ish but is Phase 4 Connect 4's** — `rl/selfplay/`,
  `rl/common/masking.py`, `scripts/score_ladder.py`, and the selfplay / `eval_win_rate` /
  loop-split-timer wiring in `rl/train.py`. Leave all of it.

### Blocking gate — ASK THE MAINTAINER FIRST

**Do not delete anything until the new repo has passed its port verification** (`start.md` §8b
there — 7 checks, each reproducing a recorded number). `data/bc_p4_*.npz`, the `runs/` artifacts
and `prior_work/` exist nowhere else; deleting them early is unrecoverable without regeneration.
You cannot verify this yourself — ask.

### A recommendation I made that the maintainer has NOT yet ruled on

`CAPSTONE_REMOVAL.md` §4: I recommended **against** deleting the written record — the README's
Phase 5 section and the Showdown entries in `SESSION_LOGS.md`. Deleting dated log entries rewrites
an honest history, and the Phase 5 write-up is the strongest evidence in the repo of *how this
project works* (pre-registered reads, credited and uncredited levers, corrections to its own
record). Strip it and a reader gets benchmarks with no methodology. The maintainer's stated goal
was "nothing about Showdown should be here," so §4 also gives a minimum-honesty version. **Confirm
which they want before touching docs** — do not silently pick either.

## Watch items for the removal

- `CLAUDE.md` now carries a banner marking its capstone content as historical; the body still
  describes the tree as it currently exists. Update it as part of the removal (§4), and **keep the
  action-masking contract** — that is a real harness invariant exercised by Connect 4 and MinAtar
  and pinned by `tests/test_masking.py`.
- Baseline before removal: **288 passed** with `--ignore=tests/test_showdown_env.py`. After
  removal the count drops (Showdown and collect tests are gone); what matters is zero failures and
  that `test_harness.py` — the CartPole sanity test that must stay green for the life of the
  project — still passes.
- Do the **EDITS before the DELETIONS** (§7). Then a test failure is obviously an edit bug rather
  than a missing file.
- Pre-existing flake, not caused by any of this: `test_full_episode_contract_against_live_server`
  fails only when the whole suite runs with a server up. It is on the deletion list anyway.

## Operational

- Env: `/opt/anaconda3/envs/deep-rl` (Python 3.13). Never `base`.
- The Showdown server may still be running on :8000 from P6. Harmless; stop it when convenient.
- Six P6 run directories under `runs/` (~425–431k history rows each) are the evidence behind the
  numbers above. Copy them to the new repo if wanted before deleting.
- Git: commit only when asked; never commit and push in one command.

Fold anything durable into `STATUS.md` / `SESSION_LOGS.md`, then restore this file to the empty
stub (see git history for its wording).
