import gym
import pybullet_envs.gym_locomotion_envs
import pytest

import librl.task

from .bidirectional_halfcheetah import HalfCheetahDirecBulletEnv

@pytest.fixture()
def hypers(hypers):
    hypers['episode_length'] = 37
    return hypers

@pytest.fixture()
def AntEnv(hypers):
    # Ant environment misbehaves.
    # Must reduce logging level.
    # See:
    #   https://stackoverflow.com/questions/60149105/userwarning-warn-box-bound-precision-lowered-by-casting-to-float32
    gym.logger.set_level(40)
    env = gym.make("AntBulletEnv-v0")
    env._max_episode_steps = hypers['episode_length']
    return env

@pytest.fixture()
def HalfCheetahEnv(hypers):
    env = HalfCheetahDirecBulletEnv(True)
    env._max_episode_steps = hypers['episode_length']
    return env