# Status

Current-state board. Rewritten in place (never appended) as work lands — update it in the
same commit that appends a `SESSION_LOGS.md` entry; hard cap ~80 lines. If this file
conflicts with the newest session-log entry, the log wins — say so and fix this file.

## Where we are (updated 2026-08-05)

**This repo is experimentally complete. No runs remain.** The from-scratch mandate is done:
DQN, PPO and SAC implemented, benchmarked and shipped on a shared harness. The Pokémon
Showdown capstone has **moved** to `/Users/nickgreenquist/Documents/Projects/pokemon-showdown-rl`,
which starts without the no-RL-libraries rule. The remaining work here is **cleanup only**,
specified in `CAPSTONE_REMOVAL.md`.

- **P6 COMPLETE + CREDITED (2026-08-05) — the last experiment run here.** Flat vs annealed LR
  at 12M on r512, 6/6 lanes, R0 passed. Annealed pooled **0.4607** (0.449/0.451/0.482) vs flat
  **0.4330** (0.425/0.424/0.450); delta +0.0277, z = 2.16 battle-level. **Credited, but it
  clears the pre-registered line by 0.003 where P5b cleared by double**; at seed level (n=3/arm)
  Welch t ≈ 2.03, p ≈ 0.12. **0.4607 is the first RL result past the BC clone (0.453)** and sits
  0.028 under the 0.489 SH-mirror ceiling. Flat 6M→12M is +0.0407, so budget bought as much as
  the anneal. **Full read: the P6 entry in `SESSION_LOGS.md`** — amended in place on the
  pre-registration entry (it has no `- 2026-08-05` bullet, so grep for `P6 RESULT`, not the date).
- Final ladder: M1 0.663 PASSED · M2 not passed · M3 shipped (`422f9ee`) · P3 complete (draw ≈4%
  of outcome variance) · P5 +0.037 · P5b +0.051 · P6 +0.028, all credited.
- **Stop rule RATIFIED (2026-08-02):** the 0.5 bar is not chased under this recipe class.

## BLOCKING GATE — no deletions yet

`CAPSTONE_REMOVAL.md` must not delete anything until **`pokemon-showdown-rl` passes its port
verification** (`MIGRATION.md` there, checks 1–8, each reproducing a recorded number).
**Verified 2026-08-05: only check 2 is done; the editable install still resolves to this repo**
(`import rl` → `deep-rl-from-scratch/rl/__init__.py`), so its precondition 3 has not run.
The request handed to that repo is `PORT_VERIFICATION_HANDOFF.md` (delete there when reported).

`runs/` (25 GB), `data/` (3.9 GB), `showdown/` and `logs/` are **gitignored — deletion is
unrecoverable**. `prior_work/`, `assets/`, `DESIGN_P7.md` and all tier-3 edits are git-recoverable.

## Open decisions — maintainer has not ruled

1. **`CAPSTONE_REMOVAL.md` §4, the written record.** Keep-and-reframe (the removal doc's own
   recommendation) vs the minimum-honesty strip vs deleting the `SESSION_LOGS.md` narrative.
   Do not silently pick. Both defensible options leave `SESSION_LOGS.md`/`PLAN_ARCHIVE.md` intact.
2. **The six P6 run dirs.** `runs/showdown_r512_12m_s{0,1,2}` + `runs/showdown_r512_lra12m_s{3,4,5}`
   (~6 GB) back the 0.4607 headline and are **not in the new repo** — `MIGRATION.md` handed over
   the numbers, not the artifacts. Copy before deleting, or decide explicitly to lose them.

## Next, in order

1. **Wait for the port-verification report** from `pokemon-showdown-rl` (§5 of the handoff file
   there): measured values per check, plus the P6-copy and `prior_work` diff results, plus one
   explicit line authorizing the strip.
2. **Resolve the two open decisions above.**
3. **Execute `CAPSTONE_REMOVAL.md` §7 order:** tier-3 EDITS first (shared files — `rl/envs/make.py`,
   `scripts/eval_checkpoint.py`, `scripts/watch.py`, `tests/test_selfplay_harness.py`,
   `pyproject.toml`), then tier-1, then tier-2 deletions, then the doc decisions, then §6 verify.
4. Delete `CAPSTONE_REMOVAL.md` and this gate section when done.

## Watch items

- **This tree must stay clean and unmodified in `rl/` until the gate clears** — `MIGRATION.md`'s
  failure playbook diffs suspect files against it. Doc-only edits are safe; `rl/` edits are not.
- **Keep the action-masking contract** through the removal. It is a genuine harness invariant,
  exercised by Connect 4 and MinAtar, pinned by `tests/test_masking.py`.
- **Do NOT delete** `rl/selfplay/`, `rl/common/masking.py`, `scripts/score_ladder.py`,
  `scripts/pons_*.py`, or the selfplay / `eval_win_rate` / loop-split wiring in `rl/train.py` —
  all Phase 4 Connect 4, all env-agnostic. `CAPSTONE_REMOVAL.md` §5.
- Test baseline before removal: **288 passed** with `--ignore=tests/test_showdown_env.py`. After
  removal the count drops; what matters is zero failures and that `test_harness.py` stays green.
- Pre-existing flake, unrelated: `test_full_episode_contract_against_live_server` fails only in a
  full-suite run with a server up; passes alone (2026-08-01). It is on the deletion list anyway.
- Precondition 3 in the new repo removes this repo's editable install. Harmless — from this root
  `./rl` wins on `sys.path`; re-confirm resolution before trusting a test result here.

## Operational

- Env: `/opt/anaconda3/envs/deep-rl` (Python 3.13). Never `base` or `pytorch_env`.
- Showdown server still up on :8000 (PID 33842, `simulator: 4`) from P6 — the new repo's checks
  4/6/7/8 need it, so leave it until they report. Stop it after.
- wandb offline since `e53323a`; `runs/*/history.csv` via `scripts/extract_history.py`.
- Git: commit only when asked; never commit and push in one command.
