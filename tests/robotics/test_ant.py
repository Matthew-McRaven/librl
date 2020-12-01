import pytest

import librl.train.train_loop, librl.train.log
import librl.train.cc.pg, librl.train.cc.maml
import librl.nn.actor, librl.nn.core, librl.nn.critic, librl.nn.pg_loss
import librl.agent.pg

from . import *

@pytest.mark.parametrize("agent_type", [RandomAgent, VPGAgent, PGBAgent, PPOAgent])
@pytest.mark.parametrize('train_fn', [librl.train.cc.policy_gradient_step, librl.train.cc.maml_meta_step])
def test_pg_ant(AntEnv, hypers, train_fn, agent_type):
    librl.train.train_loop.cc_episodic_trainer(hypers, ant_dist(AntEnv, hypers, agent_type(AntEnv, hypers)),
        train_fn, librl.train.log.cc_action_reward_logger)

@pytest.mark.parametrize("explore_bonus", [librl.reward.basic_entropy_bonus()])
@pytest.mark.parametrize("agent_type", [VPGAgent, PGBAgent, PPOAgent])
@pytest.mark.parametrize('train_fn', [librl.train.cc.policy_gradient_step])
def test_ant_bonus(AntEnv, hypers, train_fn, agent_type, explore_bonus):
    librl.train.train_loop.cc_episodic_trainer(hypers, ant_dist(AntEnv, hypers, agent_type(AntEnv, hypers, explore_bonus=explore_bonus)),
        train_fn, librl.train.log.cc_action_reward_logger)

@pytest.mark.parametrize('train_fn', [librl.train.cc.policy_gradient_step, librl.train.cc.maml_meta_step])
def test_ant_1d_convolution(AntEnv, hypers, train_fn):
    conv_list = [
            librl.nn.core.cnn.conv_def(4, 4, 1, 0, 1, False),
            librl.nn.core.cnn.conv_def(4, 4, 1, 0, 1, False),
            librl.nn.core.cnn.pool_def(1, 1, 0, 1, True, 'max'),
    ]
    policy_kernel = librl.nn.core.ConvolutionalKernel(conv_list, AntEnv.observation_space.shape, 1, dims=1)
    policy_net = librl.nn.actor.IndependentNormalActor(policy_kernel, AntEnv.action_space, AntEnv.observation_space)

    agent = librl.agent.pg.REINFORCEAgent(policy_net)
    agent.train()

    librl.train.train_loop.cc_episodic_trainer(hypers, ant_dist(AntEnv, hypers, agent),
        train_fn, librl.train.log.cc_action_reward_logger)

@pytest.mark.parametrize('train_fn', [librl.train.cc.policy_gradient_step, librl.train.cc.maml_meta_step])
def test_ant_recurrent(AntEnv, hypers, train_fn):
    x = functools.reduce(lambda x,y: x*y, AntEnv.observation_space.shape, 1)
    value_kernel = librl.nn.core.RecurrentKernel(x, 113, 1, recurrent_unit="GRU")
    value_net = librl.nn.critic.ValueCritic(value_kernel)
    policy_kernel = librl.nn.core.RecurrentKernel(x, 211, 3, recurrent_unit="RNN")
    policy_net = librl.nn.actor.IndependentNormalActor(policy_kernel, AntEnv.action_space, AntEnv.observation_space)
    policy_loss = librl.nn.pg_loss.PGB(value_net)

    agent = librl.agent.pg.ActorCriticAgent(value_net, policy_net, policy_loss)
    agent.train()
    
    librl.train.train_loop.cc_episodic_trainer(hypers, ant_dist(AntEnv, hypers, agent),
        train_fn, librl.train.log.cc_action_reward_logger)

@pytest.mark.parametrize('train_fn', [librl.train.cc.policy_gradient_step, librl.train.cc.maml_meta_step])
def test_ant_bilinear(AntEnv, hypers, train_fn):
    conv_list = [
        librl.nn.core.cnn.conv_def(4, 4, 1, 0, 1, False),
        librl.nn.core.cnn.conv_def(4, 4, 1, 0, 1, False),
        librl.nn.core.cnn.pool_def(1, 1, 0, 1, True, 'max'),
    ]
    k0 = librl.nn.core.ConvolutionalKernel(conv_list, AntEnv.observation_space.shape, 1, dims=1)
    k1 = librl.nn.core.MLPKernel(AntEnv.observation_space.shape, [200, 200])
    policy_kernel = librl.nn.core.JoinKernel(k0, k1, 20)
    policy_net = librl.nn.actor.IndependentNormalActor(policy_kernel, AntEnv.action_space, AntEnv.observation_space)

    agent = librl.agent.pg.REINFORCEAgent( policy_net,)
    agent.train()
    
    librl.train.train_loop.cc_episodic_trainer(hypers, ant_dist(AntEnv, hypers, agent),
        train_fn, librl.train.log.cc_action_reward_logger)