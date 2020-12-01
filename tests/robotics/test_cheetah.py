import pytest

import librl.train.train_loop, librl.train.log
import librl.train.cc.pg, librl.train.cc.maml
import librl.nn.actor, librl.nn.core, librl.nn.critic, librl.nn.pg_loss
import librl.agent.pg

from . import *

@pytest.mark.parametrize("agent_type", [RandomAgent, VPGAgent, PGBAgent, PPOAgent])
@pytest.mark.parametrize('train_fn', [librl.train.cc.policy_gradient_step, librl.train.cc.maml_meta_step])
def test_pg_cheetah(HalfCheetahEnv, hypers, train_fn, agent_type):
    librl.train.train_loop.cc_episodic_trainer(hypers, cheetah_dist(HalfCheetahEnv, hypers, agent_type(HalfCheetahEnv, hypers)),
        train_fn, librl.train.log.cc_action_reward_logger)

@pytest.mark.parametrize('train_fn', [librl.train.cc.policy_gradient_step, librl.train.cc.maml_meta_step])
def test_cheetah_recurrent(HalfCheetahEnv, hypers, train_fn):
    x = functools.reduce(lambda x,y: x*y, HalfCheetahEnv.observation_space.shape, 1)
    policy_kernel = librl.nn.core.RecurrentKernel(x, 211, 3, recurrent_unit="LSTM")
    policy_net = librl.nn.actor.IndependentNormalActor(policy_kernel, HalfCheetahEnv.action_space, HalfCheetahEnv.observation_space)

    agent = librl.agent.pg.REINFORCEAgent(policy_net)
    agent.train()

    librl.train.train_loop.cc_episodic_trainer(hypers, cheetah_dist(HalfCheetahEnv, hypers, agent),
        train_fn, librl.train.log.cc_action_reward_logger)