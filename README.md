# deep-rl-from-scratch

Deep reinforcement learning algorithms — DQN, PPO, SAC — implemented from scratch in PyTorch and benchmarked apples-to-apples on a shared training harness. No RL libraries.

The project has two parts:

- **Spine:** DQN → PPO → SAC, each a standalone milestone with a headline metric against a published baseline. Vanilla DQN is discrete-only and SAC is continuous-only, so the suite runs on two tracks with PPO as the bridge:
  - **Discrete** (DQN vs PPO): CartPole / LunarLander for sanity checks, then MinAtar.
  - **Continuous** (PPO vs SAC): MuJoCo locomotion (HalfCheetah, Hopper, Walker2d).
- **Capstone:** Pokémon Showdown Gen 1 singles (battle phase only) via poke-env against a local Showdown server, with PPO + self-play. A Connect 4 self-play on-ramp comes first, so the self-play loop, checkpoint pool and Elo harness are built and validated somewhere a run takes minutes and a solver gives ground truth.

## Layout

```
configs/          # one YAML per run: env id, seed, algorithm hyperparameters
rl/
├── agents/       # Agent interface + implementations (random, tabular Q; DQN/PPO/SAC later)
├── networks/     # MLP/CNN encoders (later phases)
├── buffers/      # replay (off-policy) and rollout (on-policy) buffers
├── envs/         # Gymnasium env factory + wrappers; vectorization seam
├── common/       # seeding, config, logger, evaluation, checkpointing
└── train.py      # unified entry point: config -> env + agent -> train/eval loop
scripts/          # run helpers
tests/            # harness sanity tests (must always stay green)
runs/             # run outputs: checkpoints, TensorBoard events (gitignored)
```

Every algorithm plugs into the same entry point, logger, and evaluation protocol:

```
python -m rl.train --config configs/<run>.yaml
```

## Design rules

- **From scratch.** No Stable-Baselines3, RLlib, Tianshou, or CleanRL as dependencies — owning the algorithm implementations is the point.
- **One harness.** Shared seeding, logging, evaluation, and checkpointing, with locked metric names (`rollout/episode_return`, `eval/return_mean`, …) so learning curves compare directly across algorithms.
- **Both action spaces are first-class.** Nothing in shared code assumes discrete actions.
- **Reproducible evaluation:** fixed eval seeds, deterministic policy, N episodes, mean ± std.
- **Minimal dependencies**, pinned. CPU by default; GPU only enters at the capstone.

## Status

| Phase | Deliverable | Status |
|------:|-------------|--------|
| 0 | Repo + shared harness; random-policy pipeline check on CartPole; tabular Q-learning on FrozenLake | done — Q-learning hits 0.67 success on slippery FrozenLake (random: 0.02, optimal: ~0.74) |
| 1 | DQN (replay buffer, target network, ε-greedy; Double/Dueling/n-step as toggles) | done — reproduces the MinAtar paper's DQN on all 5 games (see Results); solves CartPole/LunarLander at peak |
| 2 | PPO (GAE, clipped objective, entropy bonus, vectorized rollouts) | discrete track done — beats DQN on 3 of 5 MinAtar games (see Results); continuous track next |
| 3 | SAC (twin critics, reparameterized actor, auto-tuned entropy temperature) | planned |
| 4 | Connect 4 self-play on-ramp: opponent pool, checkpoint Elo harness | planned |
| 5 | Capstone: Pokémon Showdown Gen 1 via PPO + self-play | planned |

## Results — Phase 1: DQN on MinAtar

![DQN on MinAtar: five games vs the published baseline, plus Breakout ablations](assets/minatar_dqn_campaign.png)

From-scratch DQN reproduces the MinAtar paper's published DQN results (Young & Tian 2019; 5M frames, `-v0` envs — full 6-action set, sticky actions p=0.1) on **all five games**, 3 seeds per setting (paper: 30). Numbers are training return averaged over the final 500k steps, mean ± std across seeds — the paper's own ε-contaminated metric:

| Game | Best-matching setting | Ours | Paper ≈ |
|---|---|---|---|
| Breakout | Adam 2.5e-4 | 10.3 ± 0.4 | 10 |
| Freeway | either optimizer | 51.1 ± 0.1 (Adam) / 53.4 ± 0.2 (RMSprop) | 50.5 |
| Seaquest | centered RMSprop 2.5e-4 | 19.7 ± 4.1 | 20 |
| Space Invaders | centered RMSprop 2.5e-4 | 44.7 ± 0.5 | 45 |
| Asterix | centered RMSprop 1e-4 | 16.8 ± 1.2 | 16.5 |

Findings worth the compute:

- **Optimizer choice interacts per-game.** Adam matches or beats the paper's centered RMSprop on Breakout and Freeway, but loses badly on Seaquest (5.4 vs 19.7) and Space Invaders (32.3 vs 44.7) at the same learning rate. The published curves are RMSprop curves: replicating them took the paper's optimizer on three games and its per-game step-size tuning on Asterix.
- **Greedy policies score far above training return** — the ε=0.1 exploration floor is expensive here (Space Invaders: greedy ~91 vs ε-contaminated 45; one random action drops the ball / walks into a bullet).
- **Breakout ablations** (3 seeds each, de-biased 100-episode evals): n-step 3 yields the best final policy (25.1 vs vanilla's 23.3); Double DQN eliminates the best-vs-final churn gap (−0.1 vs vanilla's +2.5) at a training-return cost — textbook overestimation damping at the textbook price.
- **Training-time "best eval" checkpoints are winner's-cursed:** a 20-episode best overstates a fresh 100-episode re-eval by ~15% on every variant. Headline numbers use the de-biased protocol (`scripts/eval_checkpoint.py`, disjoint eval seeds).
- A strong Seaquest policy can survive **indefinitely** under greedy eval (oxygen is renewable and MinAtar registers no time limit) — two diagnostic runs spent 5+ hours inside a single eval episode before the eval protocol gained a 10k-step cap.

What that looks like in play — greedy rollouts from the best LunarLander checkpoint (seed 0), recorded with `scripts/record.py`:

![Trained DQN landing in LunarLander: three greedy episodes with a step/return HUD](assets/lunarlander_dqn_rollout.gif)

Three episodes scoring 254 / 273 / 266 — all three land on the pad rather than crashing or hovering out the clock, which is the behaviour the 200-point "solved" threshold is meant to capture.

Full experiment log in `PLAN.md`. Every run directory is self-describing — resolved config, git SHA, package versions, W&B history, best + final checkpoints — across the 63 five-million-step runs (~60 core-hours on a laptop CPU) behind these numbers.

## Results — Phase 2: PPO on MinAtar (and why the on-ramp mattered)

![PPO vs DQN on MinAtar: five training-return panels plus the de-biased re-eval comparison](assets/minatar_ppo_vs_dqn.png)

PPO runs the same harness, the same `-v0` envs, the same 5M-step budget and the
same conv trunk as the DQN campaign — Conv 16@3×3 → FC 128 — so the comparison
holds architecture fixed and varies only the algorithm. 30 runs, 5 games × 2
candidate learning rates × 3 seeds, ~55 minutes wall-clock.

**The headline is the de-biased 100-episode greedy re-eval**, not training
return. DQN's training return pays a constant ε=0.1 exploration tax while PPO's
pays a sampling tax that shrinks as entropy falls, so cross-algorithm gaps in
training return are exploration-mechanism artifacts. The re-eval taxes both
identically. Each algorithm is shown at its best-known configuration:

| Game | PPO (lr 1e-3) | DQN best-known | Verdict |
|---|---|---|---|
| Space Invaders | **276.9 ± 37.9** | 90.7 (RMSprop) | PPO **3×** |
| Asterix | **31.5 ± 1.2** | 25.6 (RMSprop lr 1e-4) | PPO +23% |
| Freeway | **61.3 ± 0.5** | 59.3 (RMSprop) | PPO modest |
| Breakout | 25.9 ± 2.6 | 25.1 (n-step 3) | tie |
| Seaquest | 24.5 ± **19.4** | **28.7** (RMSprop) | DQN — PPO unreliable |

Mean ± std across 3 seeds. **PPO wins decisively on two games, modestly on one,
ties one, and loses one** — a mixed result, which is what the literature would
predict; neither family dominates across environments.

<!-- TODO (in flight): fold in the DQN Breakout lr probe — configs/minatar_breakout_dqn_{lr5e4,lr1e3}.yaml,
     3 seeds each, seeds 0/1/2 paired against the vanilla runs. If DQN does not improve at higher lr,
     the Breakout tie stands with both algorithms swept and the "tuning budget" caveat below can be
     narrowed to Freeway only. If it does improve, the Breakout row flips to DQN and the summary line
     above becomes "decisively ahead on two, modestly on one, behind on two". -->
Greedy rollouts from the best PPO Breakout checkpoint (`scripts/record.py`):

![Trained PPO playing MinAtar Breakout: three greedy episodes with a step/return HUD](assets/minatar_breakout_ppo_rollout.gif)

Three episodes scoring 57 / 48 / 28. Watch the paddle track the ball's *trail*
rather than its current cell — the trail plane is what makes velocity
observable in a single frame, which is why one conv layer over 10×10 planes is
enough here and no frame stacking is needed.

Findings worth the compute:

- **Seaquest is a coin flip, not a loss.** Per-seed re-evals are 15.5 / 6.6 /
  **51.4** — one seed roughly doubles DQN's best, two don't. The ±19.4 is the
  point. (Checked: the outlier is genuinely stronger play at ~656-step episodes,
  not the degenerate immortal-but-scoreless policy that ate two Phase 1 runs.)
- **Throughput inverts the usual story.** PPO finishes 5M steps in ~9–12 min
  against DQN's ~55, because it takes 16 gradient steps per 1024 transitions
  where DQN takes ~1024. On a fixed *sample* budget replay is the more efficient
  learner; on fixed wall-clock, on-policy with vectorized envs wins outright.
- **PPO holds its final policy; DQN churns.** PPO's final checkpoint re-evals at
  or above its own best-selected checkpoint (Breakout: final 25.9 vs best 24.9 —
  the winner's curse on a 20-episode selection), where vanilla DQN gives back
  ~2.5 points between best and final. With the lr anneal off, that stability is
  a real property rather than a schedule artifact.
- **A mis-tuned learning rate is indistinguishable from a bug.** The originally
  locked config (lr 2.5e-4 annealed to zero over 5M) plateaued on Breakout at
  ~5.0 while *every health diagnostic read green* — clip_frac 0.087 inside its
  band, approx_kl 2e-3, entropy declining smoothly, value loss falling. It took
  a four-way disproof — CartPole reproducing the previous phase's eval sequence
  exactly, critic-vs-return correlation 1.000, a state-responsive actor, healthy
  unit statistics — to rule out a defect before sweeping the lr. The anneal was
  the trap: it put the *time-averaged* lr at 1.25e-4, and an independent
  implementation (gymnax) publishes a Breakout sweep that sits flat for a full
  10M steps at exactly that value. Raising lr to 1e-3 and turning the anneal off
  moved Breakout from 5.5 to 25.9.
- **clip_frac being in-band does not mean the lr is right.** It measures how far
  each update moves relative to the clip threshold, not whether learning is fast
  enough. It is a safety diagnostic, not a sufficiency one — a distinction that
  cost an afternoon.

Two caveats stated rather than buried:

- **Unmatched tuning budget on Freeway.** PPO was swept over five learning rates
  at 5M before its number was chosen; DQN's Freeway number comes from the paper's
  hyperparameters and was never lr-swept (Phase 1 swept lr only on the three games
  where DQN's curves trailed). PPO's 61.3 vs 59.3 margin there is small enough
  that a DQN sweep could plausibly close it. Breakout gets its own sweep (above);
  the Space Invaders and Asterix gaps are too large to be tuning artifacts.
- **Published PPO-on-MinAtar numbers are not comparable to these.** The JAX
  MinAtar ecosystem (gymnax, pgx) runs `use_minimal_action_set=True` — Breakout
  has 3 actions, not 6 — with **no sticky actions** and a capped episode length.
  That is a strictly easier MDP than the `-v0` envs used here. Their scores are
  directional context only. The DQN-vs-PPO comparison above is unaffected: both
  ran the identical environment.

### CartPole: PPO vs the REINFORCE on-ramp

![PPO vs REINFORCE on CartPole: eval return over 150k steps](assets/cartpole_ppo_vs_reinforce.png)

Before PPO, a ~80-line REINFORCE agent (per-episode updates, reward-to-go,
normalized returns, no critic) established the policy-gradient core in
isolation. On CartPole at a matched 150k-step budget it is the *better* agent —
de-biased 100-episode re-evals of the final checkpoints put REINFORCE at 500.0
and PPO at 475.4 ± 48.5.

That is the honest result, and the point: GAE, clipping, minibatch reuse and
vectorized collection buy nothing on a task REINFORCE already solves in 55k
steps. The case for the machinery starts where REINFORCE cannot go — MinAtar
above, and the continuous track next.

## Setup

Requires Python ≥ 3.10. The `box2d` extra (LunarLander) needs `swig` available at install time (`brew install swig` on macOS).

```
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Run

Train (W&B is the default logger — `wandb login` first, or prefix with
`WANDB_MODE=offline`, or set `logger: tensorboard` in the YAML):

```
python -m rl.train --config configs/frozenlake_q.yaml     # Q-learning on FrozenLake
python -m rl.train --config configs/cartpole_random.yaml  # random-policy pipeline check
```

Training is single-threaded torch by default (`torch_threads: 1` in the
config): per-step RL kernels are microseconds of math, so the default
intra-op thread pool costs more in fork/join than it buys (5x+ measured
slowdown on MinAtar), and one core per run is what lets multi-seed
benchmarks parallelize. For benchmark runs, also set the env var — the
OpenMP runtime sizes its pool at import time, before the config can act:

```
OMP_NUM_THREADS=1 python -m rl.train --config configs/minatar_breakout_dqn.yaml
```

Watch a trained checkpoint play in a render window, with a live step/return
line per episode (`--episodes N`; `--fps N` for slow motion — CartPole's
native 50 fps is over in a blink):

```
python scripts/watch.py runs/frozenlake_q/checkpoint.pt
python scripts/watch.py runs/cartpole_random/checkpoint.pt --fps 15   # random policy flailing
```

Record the same greedy rollouts as an annotated GIF — episode/step/return
stamped on every frame (`--seed` pins the episodes for a reproducible clip;
`--max-steps` caps recording of effectively-immortal policies):

```
python scripts/record.py runs/<run>/best_checkpoint.pt
```

Run outputs live under `runs/<run_name>/` (gitignored), so train before watching.
