"""Single entry point: `python -m rl.train --config configs/<run>.yaml`.
Every algorithm plugs in here; the loop stays algorithm-agnostic.
"""

import argparse
import importlib.metadata
import subprocess
import time
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import yaml

from rl.agents.base import Agent
from rl.agents.dqn import DQNAgent
from rl.agents.ppo import PPOAgent
from rl.agents.q_learning import QLearningAgent
from rl.agents.random_agent import RandomAgent
from rl.agents.reinforce import ReinforceAgent
from rl.agents.sac import SACAgent
from rl.common.checkpoint import save_checkpoint
from rl.common.config import Config, load_config, run_dir
from rl.common.evaluation import evaluate
from rl.common.logging import Logger, make_logger
from rl.common.seeding import set_seed
from rl.envs.make import make_env, make_vec_env
from rl.envs.normalize import (
    FrozenNormalizeObservation,
    NormalizeObservation,
    NormalizeReward,
    RunningMeanStd,
)


# Algo registry: make_agent constructs from it, and train() reads the class's
# `vectorized` flag to pick env construction and collection path up front.
ALGOS: dict[str, type[Agent]] = {
    "random": RandomAgent,
    "q_learning": QLearningAgent,
    "dqn": DQNAgent,
    "reinforce": ReinforceAgent,
    "ppo": PPOAgent,
    "sac": SACAgent,
}


def make_agent(cfg: Config, env: gym.Env) -> Agent:
    algo = cfg.agent.get("algo")
    cls = ALGOS.get(algo)
    if cls is None:
        raise ValueError(f"unknown algo {algo!r}")
    if cls is RandomAgent:
        return RandomAgent(env.action_space)
    hparams = {k: v for k, v in cfg.agent.items() if k != "algo"}
    if cls is QLearningAgent:
        return QLearningAgent(env.observation_space, env.action_space, **hparams)
    if cls.vectorized:
        # Vectorized agents build against one sub-env's spaces plus the batch
        # width. The getattr fallbacks cover the scalar-env rebuild in
        # watch/record/eval_checkpoint: plain spaces, width 1.
        return cls(
            getattr(env, "single_observation_space", env.observation_space),
            getattr(env, "single_action_space", env.action_space),
            num_envs=getattr(env, "num_envs", 1),
            device=cfg.device,
            **hparams,
        )
    # Torch agents share one constructor shape (DQN, REINFORCE, later SAC).
    return cls(env.observation_space, env.action_space, device=cfg.device, **hparams)


def _write_run_metadata(out_dir: Path, cfg: Config) -> None:
    """Stamp the run dir before training starts: the resolved config (CLI
    overrides baked in, reloadable via load_config) plus provenance — a
    benchmark campaign spans days of possible code drift, so every result
    must trace back to an exact tree."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(asdict(cfg), sort_keys=False))
    repo_root = Path(__file__).resolve().parents[1]
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, cwd=repo_root,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True, text=True, check=True, cwd=repo_root,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        sha, dirty = "unknown", False
    versions = {}
    for pkg in ("torch", "gymnasium", "numpy", "minatar", "wandb"):
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            pass
    meta = {
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": sha,
        "git_dirty": dirty,
        "versions": versions,
    }
    (out_dir / "meta.yaml").write_text(yaml.safe_dump(meta, sort_keys=False))


def train(cfg: Config) -> None:
    # First, before any tensor work (config.py explains the default of 1).
    # Belt-and-suspenders: OMP_NUM_THREADS=1 at launch also binds the OpenMP
    # runtime itself, which is sized before this call can run.
    torch.set_num_threads(cfg.torch_threads)
    set_seed(cfg.seed)
    # Collection path is a property of the algorithm class (Agent.vectorized),
    # known before construction; an unknown algo falls through to make_agent's
    # error. Eval always runs a single scalar env either way.
    agent_cls = ALGOS.get(cfg.agent.get("algo"))
    vectorized = agent_cls is not None and agent_cls.vectorized
    if (cfg.normalize_obs or cfg.normalize_reward) and not vectorized:
        # The normalizers are vector-level wrappers. Silently ignoring the
        # flags would still stamp them into the run's config snapshot, and
        # every checkpoint from that "successful" run would then refuse to
        # re-evaluate (frozen_obs_env raises on missing statistics).
        raise ValueError(
            f"normalize_obs/normalize_reward need a vectorized algorithm; "
            f"{cfg.agent.get('algo')!r} runs the scalar loop"
        )
    if vectorized:
        env = make_vec_env(cfg.env_id, cfg.seed, cfg.num_envs)
    else:
        env = make_env(cfg.env_id, cfg.seed)
    eval_env = make_env(cfg.env_id, cfg.seed)  # eval reseeds per episode
    # Normalization statistics are shared, not copied: the eval env reads the
    # training env's live RunningMeanStd but never updates it, so each eval
    # pass scores the policy under the statistics it is currently training
    # against. They are checkpointed at every save (see save_checkpoint).
    normalizers: dict[str, RunningMeanStd] = {}
    if cfg.normalize_obs:
        env = NormalizeObservation(env)
        normalizers["obs"] = env.rms
        eval_env = FrozenNormalizeObservation(eval_env, env.rms)
    if cfg.normalize_reward:
        # Reward scaling is a training-time device only — eval returns are
        # always reported in true env units.
        env = NormalizeReward(env, gamma=cfg.agent["gamma"])
        normalizers["reward"] = env.rms
    agent = make_agent(cfg, env)
    out_dir = run_dir(cfg)
    # Before the logger: even a run that dies in wandb.init leaves a stamped dir.
    _write_run_metadata(out_dir, cfg)
    logger = make_logger(cfg)

    if vectorized:
        _vector_loop(cfg, env, eval_env, agent, logger, out_dir, normalizers)
    else:
        _scalar_loop(cfg, env, eval_env, agent, logger, out_dir)

    logger.close()
    env.close()
    eval_env.close()


def _scalar_loop(
    cfg: Config, env: gym.Env, eval_env: gym.Env, agent: Agent, logger: Logger, out_dir: Path
) -> None:
    """One env, one transition per update() call — random/tabular/DQN/REINFORCE."""
    obs, info = env.reset(seed=cfg.seed)
    mask = info.get("action_mask")  # None only for continuous-action envs
    best_eval = float("-inf")
    ep_return, ep_length = 0.0, 0
    # Per-episode loss/* sums and per-key report counts: each key is averaged
    # over the steps that reported it (DQN reports every step, REINFORCE once
    # per episode), not over ep_length.
    ep_losses: dict[str, float] = defaultdict(float)
    ep_counts: dict[str, int] = defaultdict(int)
    last_step, last_time = 0, time.perf_counter()

    for step in range(1, cfg.total_steps + 1):
        action = agent.act(obs, mask)
        next_obs, reward, terminated, truncated, info = env.step(action)
        next_mask = info.get("action_mask")
        # Per-step update on the fresh transition (tabular Q; DQN keeps this
        # cadence but samples from replay instead). Both flags are passed:
        # only `terminated` stops bootstrapping (a time-limit cut still
        # bootstraps), but `truncated` still marks an episode boundary,
        # which n-step accumulation must not chain across. The mask pair
        # rides along: `mask` legalizes obs's actions, `next_mask` s''s (the
        # bootstrap max needs it).
        for name, value in agent.update(
            (obs, action, float(reward), next_obs, terminated, truncated, mask, next_mask)
        ).items():
            ep_losses[name] += value
            ep_counts[name] += 1
        obs = next_obs
        mask = next_mask
        ep_return += float(reward)
        ep_length += 1

        if terminated or truncated:
            now = time.perf_counter()
            logger.log(
                {
                    "rollout/episode_return": ep_return,
                    "rollout/episode_length": ep_length,
                    "time/steps_per_sec": (step - last_step) / (now - last_time),
                    **{name: total / ep_counts[name] for name, total in ep_losses.items()},
                },
                step,
            )
            last_step, last_time = step, now
            obs, info = env.reset()
            mask = info.get("action_mask")
            ep_return, ep_length = 0.0, 0
            ep_losses.clear()
            ep_counts.clear()

        if step % cfg.eval_every == 0:
            metrics = evaluate(agent, eval_env, cfg.eval_episodes)
            logger.log(metrics, step)
            # The final policy is an arbitrary sample of an oscillating
            # training trajectory (deep RL policies churn), so keep the
            # best-so-far policy too. Report final and best.
            if metrics["eval/return_mean"] > best_eval:
                best_eval = metrics["eval/return_mean"]
                save_checkpoint(out_dir / "best_checkpoint.pt", agent, step, cfg)
            # Latest-policy snapshot every eval: a run that dies mid-flight
            # still leaves best + latest + full metric history behind.
            save_checkpoint(out_dir / "checkpoint.pt", agent, step, cfg)

    save_checkpoint(out_dir / "checkpoint.pt", agent, cfg.total_steps, cfg)


def _vector_loop(
    cfg: Config,
    envs: gym.vector.VectorEnv,
    eval_env: gym.Env,
    agent: Agent,
    logger: Logger,
    out_dir: Path,
    normalizers: dict[str, RunningMeanStd] | None = None,
) -> None:
    """N lockstep envs, batched transitions — vectorized (on-policy) agents.

    Autoreset is disabled (see make_vec_env): finished sub-envs are reset
    manually right after their terminal step, so every transition handed to
    update() is a real env step and next_obs at a terminal row is the
    episode's true final observation. Loss metrics are logged whenever
    update() reports them (once per rollout batch for PPO) instead of
    averaged per episode — episodes end at different times across envs.
    """
    num_envs = envs.num_envs
    obs, infos = envs.reset(seed=cfg.seed)  # gymnasium seeds sub-env i with seed + i
    masks = infos.get("action_mask")  # (N, A); None only for continuous envs
    best_eval = float("-inf")
    ep_returns = np.zeros(num_envs)
    ep_lengths = np.zeros(num_envs, dtype=np.int64)
    step, next_eval = 0, cfg.eval_every
    last_step, last_time = 0, time.perf_counter()

    # Step advances num_envs at a time, so the run ends at the first multiple
    # of num_envs >= total_steps, and evals fire on crossing each threshold
    # (both overshoot by < num_envs steps).
    while step < cfg.total_steps:
        actions = agent.act(obs, masks)
        next_obs, rewards, terminated, truncated, infos = envs.step(actions)
        # Autoreset is disabled, so step infos always describe the true
        # successor states — at a truncated row next_masks is the final
        # state's real mask, exactly what a bootstrap consumer needs.
        next_masks = infos.get("action_mask")
        step += num_envs
        update_metrics = agent.update(
            (obs, actions, rewards, next_obs, terminated, truncated, masks, next_masks)
        )
        if update_metrics:
            logger.log(update_metrics, step)
        # Episode returns are always accumulated in TRUE env units. With
        # reward normalization on, `rewards` is scaled by a running statistic
        # that itself moves during training, so logging it would make
        # rollout/episode_return incomparable across runs and against every
        # DQN and discrete-PPO number in the repo. The wrapper republishes the
        # raw reward; the fallback covers every env without it.
        ep_returns += infos.get("raw_reward", rewards)
        ep_lengths += 1
        obs = next_obs
        masks = next_masks

        done = terminated | truncated
        if done.any():
            now = time.perf_counter()
            sps = (step - last_step) / (now - last_time)
            # One log per finished episode. Simultaneous finishes share a
            # step, which W&B merges (last write wins) — rare, accepted.
            for i in np.flatnonzero(done):
                logger.log(
                    {
                        "rollout/episode_return": float(ep_returns[i]),
                        "rollout/episode_length": int(ep_lengths[i]),
                        "time/steps_per_sec": sps,
                    },
                    step,
                )
            last_step, last_time = step, now
            ep_returns[done] = 0.0
            ep_lengths[done] = 0
            # Unfinished rows come back holding their current obs, so the
            # returned array replaces obs wholesale. The reset info's mask
            # array does NOT (non-reset rows are all-False placeholders in
            # gymnasium's aggregation) — merge on the done rows only.
            obs, reset_infos = envs.reset(options={"reset_mask": done})
            if masks is not None:
                masks = np.where(done[:, None], reset_infos["action_mask"], masks)

        if step >= next_eval:
            next_eval += cfg.eval_every
            metrics = evaluate(agent, eval_env, cfg.eval_episodes)
            logger.log(metrics, step)
            if metrics["eval/return_mean"] > best_eval:
                best_eval = metrics["eval/return_mean"]
                save_checkpoint(out_dir / "best_checkpoint.pt", agent, step, cfg, normalizers)
            save_checkpoint(out_dir / "checkpoint.pt", agent, step, cfg, normalizers)

    save_checkpoint(out_dir / "checkpoint.pt", agent, step, cfg, normalizers)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="path to a run YAML")
    # Overrides for the multi-seed benchmark protocol: same YAML, N seeds,
    # each under its own run name.
    parser.add_argument("--seed", type=int, default=None, help="override the config seed")
    parser.add_argument("--run-name", default=None, help="override the config run_name")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.seed is not None:
        cfg.seed = args.seed
    if args.run_name is not None:
        cfg.run_name = args.run_name
    train(cfg)


if __name__ == "__main__":
    main()
