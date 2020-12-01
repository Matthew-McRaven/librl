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
    env = gym.make("AntBulletEnv-v0")
    env._max_episode_steps = hypers['episode_length']
    return env

@pytest.fixture()
def HalfCheetahEnv(hypers):
    env = HalfCheetahDirecBulletEnv(True)
    env._max_episode_steps = hypers['episode_length']
    return env