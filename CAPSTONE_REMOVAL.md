# How to strip the Pokémon Showdown capstone out of this repo

Written 2026-08-05 during wrap-up. The capstone moved to
`/Users/nickgreenquist/Documents/Projects/pokemon-showdown-rl`. This repo reverts to what it was
built to be: **from-scratch DQN, PPO and SAC on a shared harness, benchmarked apples-to-apples.**

Delete this file when the removal is done.

---

## STOP — prerequisite

**Do not delete anything until the new repo has passed its port verification** (`start.md` §8b in
`pokemon-showdown-rl`: all 7 checks, each reproducing its recorded number). Several items below —
`data/bc_p4_*.npz`, the `runs/` artifacts, `prior_work/` — exist nowhere else. Deleting them before
the new repo is proven is unrecoverable without regeneration.

Once verified, work through the tiers in order and run the checks in §6 after each tier.

---

## 1. Delete outright — Showdown-only code

Verified by import graph on 2026-08-05: each of these imports `poke_env` or `rl.envs.showdown`, or
exists solely to serve them.

```
rl/envs/showdown.py            the env + encoder + adapter
rl/collect.py                  InferenceSeam / SeamPlayer — built for Showdown collection, imports poke_env
tests/test_showdown_env.py     env, encoder, outcome, live-server contract
tests/test_collect.py          imports poke_env
scripts/setup_showdown.sh      vendors the Showdown server
scripts/showdown_throughput.py imports poke_env
scripts/make_showdown_figure.py
scripts/make_bc_dataset.py     BC dataset generation (Showdown-only)
scripts/train_bc.py            the BC clone (Showdown-only)
scripts/p3_team_luck.py        team-luck variance decomposition (Showdown-only)
scripts/mutations/phase5_env.py
configs/showdown_*.yaml        all 15 of them
```

## 2. Delete outright — data and artifacts

```
showdown/                      vendored Node.js server (gitignored; ~large)
data/bc_p4_main.npz            ~1.1 GB
data/bc_p4_sub10k.npz          ~0.6 GB
data/bc_p4_40k.npz             ~2.2 GB
runs/showdown_*                every Showdown run directory
runs/bc_p4_*                   BC clone runs
assets/showdown_*.png          milestone figures (only if you also strip the README narrative — §4)
logs/                          launcher stdout, gitignored
prior_work/                    ENTIRELY capstone: papers, the ps-ppo analysis, the dataset index
DESIGN_P7.md                   capstone design proposal
P6_RESULTS.md                  if a copy ended up here
```

`prior_work/` is the one to think about — it is months of verification work, but it is 100%
Pokémon/Showdown research and belongs in the new repo. Confirm it copied across before deleting.

## 3. EDIT, do not delete — shared files with Showdown branches

**This is the part that can break the spine. Do not delete these files.**

| file | what to remove |
|---|---|
| `rl/envs/make.py` | the `elif env_id.startswith("Showdown")` branch (~line 34–35), `_ensure_showdown_registered()` (~line 143–147), and the Showdown mention in the eval-extras comment (~line 121) |
| `scripts/eval_checkpoint.py` | Showdown-specific branches; keep the generic re-evaluation tool |
| `scripts/watch.py` | Showdown replay-saving branch; keep the generic viewer |
| `tests/test_selfplay_harness.py` | Showdown references only. **The self-play harness itself is Connect 4's — keep the file and its tests.** |
| `pyproject.toml` | drop `poke-env` (and `pettingzoo`/`websockets` if they came in only as its transitive pins — check before removing) |

## 4. Docs — decide deliberately, this is not mechanical

The README's Phase 5 section, `PLAN.md`'s Phase 5 spec and the Showdown entries in
`SESSION_LOGS.md` are a substantial written record of real work.

**Recommendation: do NOT delete the narrative — reframe it.** Two reasons. Deleting log entries
rewrites a history that is otherwise honest and dated; and the Phase 5 write-up is some of the
strongest evidence in the repo of how this project actually works (pre-registered reads, credited
and uncredited levers, corrections to its own record). A reader who removes it is left with
benchmarks and no methodology.

The minimum-honesty version, if you strip the capstone anyway:
- Replace the README Phase 5 section with a short paragraph: what the capstone was, the final
  result (**PPO 12M annealed = 0.4607 vs SimpleHeuristics, above the 0.453 behavioral clone, with
  the SH mirror ceiling at 0.489**), and a link to `pokemon-showdown-rl`.
- Leave `SESSION_LOGS.md` and `PLAN_ARCHIVE.md` alone. They are dated archives; editing them to
  remove a phase makes every remaining entry less trustworthy.
- `PLAN.md`: the Phase 5 spec can move wholesale to `PLAN_ARCHIVE.md` — that is what the archive is
  for and it matches how Phases 0–4 were handled.
- `CLAUDE.md`: remove the capstone paragraph and the Showdown-specific guidance (server command,
  `simulator: 4`, the seed-collision and launcher-liveness watch items — all now capstone-only).
  **Keep** the action-masking contract: it is a genuine harness invariant, it is exercised by
  Connect 4 and MinAtar, and `tests/test_masking.py` pins it.

## 5. Do NOT touch — looks capstone-ish, isn't

```
rl/selfplay/           SnapshotPool etc. — Phase 4 Connect 4 uses this
rl/common/masking.py   harness contract, exercised by Connect 4 and MinAtar
scripts/score_ladder.py    written for Connect 4 ladders; the capstone borrowed it
scripts/pons_benchmark.py  Connect 4 solver validation
scripts/pons_agent_metrics.py
tests/test_harness.py      the CartPole sanity test — must stay green for the life of the project
tests/test_masking.py
tests/test_selfplay_pool.py
rl/train.py                selfplay/eval_win_rate wiring is shared with Connect 4
```

`rl/train.py` in particular: the `selfplay` block, `eval_win_rate` and the loop-split timers
(`time/collect_sec`, `time/update_sec`, `time/eval_sec`) are all env-agnostic and used by the
spine. Leave them.

## 6. Verify after each tier

```
/opt/anaconda3/envs/deep-rl/bin/pytest tests/ -q
```

Baseline before removal was **288 passed** with `--ignore=tests/test_showdown_env.py`. After
removal the count will be lower (the Showdown and collect tests are gone) — what matters is
**zero failures and zero errors**, and specifically that `test_harness.py`, `test_masking.py`,
`test_dqn.py`, `test_ppo.py`, `test_sac.py`, `test_minatar.py` and the Connect 4 suite all pass.

Then confirm no residue:

```
grep -rin 'showdown\|poke_env\|poke-env' --include=*.py --include=*.toml --include=*.yaml . | grep -v '^./SESSION_LOGS.md'
```

Expect zero hits outside docs you deliberately kept. Also confirm the package still imports and a
smoke run works:

```
/opt/anaconda3/envs/deep-rl/bin/python -m rl.train --config configs/cartpole_dqn.yaml
```

(substitute any surviving spine config).

## 7. Order of operations

1. Confirm the new repo passed all 7 port checks.
2. Tier 3 edits first (shared files) — then run tests. Doing this before the deletions means a
   failure is obviously an edit bug, not a missing file.
3. Tier 1 deletions — run tests.
4. Tier 2 deletions (data/artifacts) — run tests.
5. Tier 4 doc decisions.
6. Full verification (§6), then commit.
7. Delete this file.
