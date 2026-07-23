"""MinAtar integration: explicit registration + ChannelFirst through the env
factory, ConvQNet shapes (plain and dueling), and conv-DQN through the real
train loop on Breakout.
"""

import numpy as np
import pytest
import torch

from rl.common.checkpoint import load_checkpoint
from rl.common.config import Config
from rl.envs.make import make_env
from rl.networks.conv import ConvQNet
from rl.train import train


def test_make_env_minatar_channel_first():
    env = make_env("MinAtar/Breakout-v0", seed=0)
    assert env.observation_space.shape == (4, 10, 10)
    assert env.action_space.n == 6  # v0: the full shared action set
    obs, _ = env.reset(seed=0)
    assert obs.shape == (4, 10, 10)
    assert obs.dtype == np.bool_  # 1 byte/entry into the replay buffer
    assert env.observation_space.contains(obs)
    obs, _, _, _, _ = env.step(1)
    assert obs.shape == (4, 10, 10)
    env.close()


def test_channel_first_is_a_transpose_not_a_reshape():
    env = make_env("MinAtar/Breakout-v0", seed=0)
    raw = np.arange(10 * 10 * 4).reshape(10, 10, 4)
    out = env.observation(raw)
    assert out.shape == (4, 10, 10)
    assert out[3, 2, 1] == raw[2, 1, 3]
    env.close()


def test_convqnet_shapes():
    for dueling in (False, True):
        net = ConvQNet((4, 10, 10), [128], 6, dueling=dueling)
        q = net(torch.zeros(7, 4, 10, 10))
        assert q.shape == (7, 6)


def test_convqnet_requires_fc_layer():
    with pytest.raises(ValueError):
        ConvQNet((4, 10, 10), [], 6)


def test_minatar_dqn_smoke(tmp_path, monkeypatch):
    """Conv-DQN through the real train loop: warmup, gradient steps on
    bool-plane batches, a target sync, and a checkpoint that restores.
    Runs the rmsprop optimizer knob — the Adam default is covered by the
    CartPole smokes and the real configs."""
    monkeypatch.chdir(tmp_path)
    cfg = Config(
        env_id="MinAtar/Breakout-v0",
        seed=0,
        total_steps=300,
        eval_every=150,
        eval_episodes=2,
        run_name="test_minatar_dqn",
        logger="tensorboard",
        agent={
            "algo": "dqn",
            "hidden_sizes": [32],
            "lr": 1.0e-3,
            "optimizer": "rmsprop",
            "gamma": 0.99,
            "buffer_capacity": 1000,
            "batch_size": 32,
            "learning_starts": 100,
            "target_update_every": 100,
            "epsilon_start": 1.0,
            "epsilon_end": 0.1,
            "epsilon_decay_steps": 200,
        },
    )
    train(cfg)
    ckpt = load_checkpoint(tmp_path / "runs" / "test_minatar_dqn" / "checkpoint.pt")
    assert ckpt["step"] == 300
    assert ckpt["agent"]["grad_steps"] > 0
