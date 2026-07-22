"""Watch a checkpointed policy play its env in a render window.

    python scripts/watch.py runs/<run_name>/checkpoint.pt [--episodes N]

The checkpoint carries its own config, so the agent and env are rebuilt
from it directly — no YAML needed. Plays the deterministic (eval) policy.
"""

import argparse

from rl.common.checkpoint import load_checkpoint
from rl.common.config import Config
from rl.envs.make import make_env
from rl.train import make_agent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="path to a checkpoint.pt")
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()

    ckpt = load_checkpoint(args.checkpoint)
    cfg = Config(**ckpt["config"])
    env = make_env(cfg.env_id, cfg.seed, render_mode="human")
    agent = make_agent(cfg, env)
    agent.load_state_dict(ckpt["agent"])

    for episode in range(args.episodes):
        obs, _ = env.reset()  # unseeded on purpose: fresh episodes every run
        ep_return, done = 0.0, False
        while not done:
            obs, reward, terminated, truncated, _ = env.step(agent.act(obs, deterministic=True))
            ep_return += float(reward)
            done = terminated or truncated
        print(f"episode {episode + 1}: return {ep_return:g}")
    env.close()


if __name__ == "__main__":
    main()
