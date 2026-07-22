"""CartPole sanity test — the known-good path when a reward curve goes flat.

Runs the real train loop end-to-end (env -> rollout -> logging -> eval ->
checkpoint) with a random agent for a few hundred steps and asserts it
completes. Must stay green for the life of the project.
"""

from rl.common.checkpoint import load_checkpoint
from rl.common.config import Config
from rl.train import train


def test_cartpole_harness_completes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # runs/ output lands in the tmp dir
    cfg = Config(
        env_id="CartPole-v1",
        seed=0,
        total_steps=300,
        eval_every=150,
        eval_episodes=2,
        run_name="test_cartpole",
        logger="tensorboard",  # offline backend; keeps tests free of W&B auth
        agent={"algo": "random"},
    )
    train(cfg)
    ckpt = load_checkpoint(tmp_path / "runs" / "test_cartpole" / "checkpoint.pt")
    assert ckpt["step"] == 300
    assert ckpt["config"]["env_id"] == "CartPole-v1"
