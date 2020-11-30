import functools
import unittest

import gym
import pybullet_envs.gym_locomotion_envs

import librl.agent.pg, librl.agent.mdp
import librl.nn.core, librl.nn.critic, librl.nn.actor
import librl.reward
import librl.task, librl.hypers
import librl.train.train_loop, librl.train.log
import librl.train.cc.pg, librl.train.cc.maml

from .antwrapper import AntWrapper

class ReinforceWithEntropyBonusTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env_wrapper = AntWrapper()
        
    def setUp(self):
        x = functools.reduce(lambda x,y: x*y, self.env_wrapper.env.observation_space.shape, 1)
        self.policy_kernel = librl.nn.core.RecurrentKernel(x, 200, 2, recurrent_unit="LSTM")
        self.policy_net = librl.nn.actor.IndependentNormalActor(self.policy_kernel, self.env_wrapper.env.action_space, self.env_wrapper.env.observation_space)

        self.agent = librl.agent.pg.REINFORCEAgent(self.policy_net, explore_bonus_fn=librl.reward.basic_entropy_bonus())
        self.agent.train()

        self.env_wrapper.setUp(self.agent)
        
    def tearDown(self):
        self.env_wrapper.tearDown()
        del self.policy_kernel, self.policy_net, self.agent

    def test_policy_updates(self):
        cc = librl.train.cc
        for idx, alg in enumerate([cc.policy_gradient_step, cc.maml_meta_step]):
            with self.subTest(i=idx):
                librl.train.train_loop.cc_episodic_trainer(self.env_wrapper.hypers, self.env_wrapper.dist,
                    alg, librl.train.log.cc_action_reward_logger)
class PGBTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env_wrapper = AntWrapper()
        
    def setUp(self):
        x = functools.reduce(lambda x,y: x*y, self.env_wrapper.env.observation_space.shape, 1)
        self.value_kernel = librl.nn.core.RecurrentKernel(x, 113, 1, recurrent_unit="GRU")
        self.value_net = librl.nn.critic.ValueCritic(self.value_kernel)
        self.policy_kernel = librl.nn.core.RecurrentKernel(x, 211, 3, recurrent_unit="RNN")
        self.policy_net = librl.nn.actor.IndependentNormalActor(self.policy_kernel, self.env_wrapper.env.action_space, self.env_wrapper.env.observation_space)
        self.policy_loss = librl.nn.pg_loss.PGB(self.value_net)

        self.agent = librl.agent.pg.ActorCriticAgent(self.value_net, self.policy_net, self.policy_loss)
        self.agent.train()

        self.env_wrapper.setUp(self.agent)
        
    def tearDown(self):
        self.env_wrapper.tearDown()
        del self.policy_kernel, self.policy_net, self.agent, self.value_kernel, self.value_net, self.policy_loss

    def test_policy_updates(self):
        cc = librl.train.cc
        for idx, alg in enumerate([cc.policy_gradient_step, cc.maml_meta_step]):
            with self.subTest(i=idx):
                librl.train.train_loop.cc_episodic_trainer(self.env_wrapper.hypers, self.env_wrapper.dist,
                    alg, librl.train.log.cc_action_reward_logger)

if __name__ == '__main__':
    unittest.main()