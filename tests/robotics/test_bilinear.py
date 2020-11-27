import functools
import unittest

import gym
import pybullet_envs.gym_locomotion_envs

import librl.agent.pg, librl.agent.mdp
import librl.nn.core, librl.nn.critic, librl.nn.actor
import librl.reward
import librl.task, librl.hypers
import librl.train.train_loop
import librl.train.cc.pg, librl.train.cc.maml

from .antwrapper import AntWrapper

class Reinforce(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env_wrapper = AntWrapper()
        
    def setUp(self):
        x = functools.reduce(lambda x,y: x*y, self.env_wrapper.env.observation_space.shape, 1)
        conv_list = [
            librl.nn.core.cnn.conv_def(4, 4, 1, 0, 1, False),
            librl.nn.core.cnn.conv_def(4, 4, 1, 0, 1, False),
            librl.nn.core.cnn.pool_def(1, 1, 0, 1, True, 'max'),
        ]
        k0 = librl.nn.core.ConvolutionalKernel(conv_list, self.env_wrapper.env.observation_space.shape, 1, dims=1)
        k1 = librl.nn.core.MLPKernel(self.env_wrapper.env.observation_space.shape, [200, 200])
        self.policy_kernel = librl.nn.core.JoinKernel(k0, k1, 20)
        self.policy_net = librl.nn.actor.IndependentNormalActor(self.policy_kernel, self.env_wrapper.env.action_space, self.env_wrapper.env.observation_space)

        self.agent = librl.agent.pg.REINFORCEAgent(self.policy_net, explore_bonus_fn=librl.reward.basic_entropy_bonus())
        self.agent.train()

        self.env_wrapper.setUp(self.agent)
        
    def tearDown(self):
        self.env_wrapper.tearDown()
        del self.policy_kernel, self.policy_net, self.agent

    def test_policy_grad(self):
        librl.train.train_loop.cc_episodic_trainer(self.env_wrapper.hypers, self.env_wrapper.dist, librl.train.cc.policy_gradient_step)

    def test_maml(self):
        librl.train.train_loop.cc_episodic_trainer(self.env_wrapper.hypers, self.env_wrapper.dist, librl.train.cc.maml_meta_step)

if __name__ == '__main__':
    unittest.main()