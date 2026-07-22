"""Watch a checkpointed policy play its env in a render window.

    python scripts/watch.py runs/<run_name>/checkpoint.pt [--episodes N] [--fps N]

The checkpoint carries its own config, so the agent and env are rebuilt
from it directly — no YAML needed. Plays the deterministic (eval) policy.
"""

import argparse
import time

from rl.common.checkpoint import load_checkpoint
from rl.common.config import Config
from rl.envs.make import make_env
from rl.train import make_agent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="path to a checkpoint.pt")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="render speed; lower = slow motion (env defaults: CartPole 50, FrozenLake 4)",
    )
    args = parser.parse_args()

    ckpt = load_checkpoint(args.checkpoint)
    cfg = Config(**ckpt["config"])
    env = make_env(cfg.env_id, cfg.seed, render_mode="human")
    if args.fps:
        # The human renderer paces its clock off this metadata entry.
        env.unwrapped.metadata["render_fps"] = args.fps
    agent = make_agent(cfg, env)
    agent.load_state_dict(ckpt["agent"])

    for episode in range(args.episodes):
        obs, _ = env.reset()  # unseeded on purpose: fresh episodes every run
        ep_return, ep_length, done = 0.0, 0, False
        while not done:
            obs, reward, terminated, truncated, _ = env.step(agent.act(obs, deterministic=True))
            ep_return += float(reward)
            ep_length += 1
            print(f"\repisode {episode + 1}: step {ep_length}, return {ep_return:g} ", end="", flush=True)
            done = terminated or truncated
        outcome = "terminal state" if terminated else "time limit"
        print(f"— {outcome} after {ep_length} steps, return {ep_return:g}")
        time.sleep(1.0)  # visible episode boundary before the env resets
    env.close()


if __name__ == "__main__":
    main()
