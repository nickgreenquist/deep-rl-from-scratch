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


def test_cartpole_dqn_smoke(tmp_path, monkeypatch):
    """DQN through the same loop: warmup, gradient steps, a target sync, and
    a checkpoint that restores. Sized so all of those actually happen."""
    monkeypatch.chdir(tmp_path)
    cfg = Config(
        env_id="CartPole-v1",
        seed=0,
        total_steps=300,
        eval_every=150,
        eval_episodes=2,
        run_name="test_cartpole_dqn",
        logger="tensorboard",
        agent={
            "algo": "dqn",
            "hidden_sizes": [],
            "lr": 1.0e-3,
            "gamma": 0.99,
            "buffer_capacity": 1000,
            "batch_size": 32,
            "learning_starts": 100,  # < total_steps, so gradient steps run
            "target_update_every": 100,  # < steps after warmup, so a sync runs
            "epsilon_start": 1.0,
            "epsilon_end": 0.05,
            "epsilon_decay_steps": 200,
        },
    )
    train(cfg)
    ckpt = load_checkpoint(tmp_path / "runs" / "test_cartpole_dqn" / "checkpoint.pt")
    assert ckpt["step"] == 300
    assert ckpt["agent"]["transitions"] == 300
    # First gradient step lands on the transition that fills the buffer to
    # learning_starts, so steps 100..300 inclusive all train.
    assert ckpt["agent"]["grad_steps"] == 201
