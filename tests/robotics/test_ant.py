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

class AntWrapper:
    def __init__(self):
        self.hypers = librl.hypers.get_default_hyperparams()
        self.hypers['device'] = 'cpu'
        self.hypers['epochs'] = 5
        self.hypers['task_count'] = 2
        self.env = gym.make("AntBulletEnv-v0")
        self.env._max_episode_steps = self.hypers['episode_length']
    def setUp(self, agent):
        self.dist = librl.task.TaskDistribution()
        self.dist.add_task(librl.task.Task.Definition(librl.task.ContinuousGymTask, env=self.env, agent=agent, episode_length=53))
    def tearDown(self):
        del self.dist

class RandomTest(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.env_wrapper = AntWrapper()
    def setUp(self):
        x = functools.reduce(lambda x,y: x*y, self.env_wrapper.env.observation_space.shape, 1)
        self.policy_kernel = librl.nn.core.MLPKernel(x)
        self.policy_net = librl.nn.actor.IndependentNormalActor(self.policy_kernel, self.env_wrapper.env.action_space, self.env_wrapper.env.observation_space)

        self.agent = librl.agent.mdp.RandomAgent(self.env_wrapper.env.observation_space, self.env_wrapper.env.action_space)
        self.agent.train()
        self.env_wrapper.setUp(self.agent)

    def tearDown(self):
        self.env_wrapper.tearDown()
        del self.policy_kernel, self.policy_net, self.agent


    def test_policy_grad(self):
        librl.train.train_loop.cc_episodic_trainer(self.env_wrapper.hypers, self.env_wrapper.dist, librl.train.cc.policy_gradient_step)

    def test_maml(self):
        librl.train.train_loop.cc_episodic_trainer(self.env_wrapper.hypers, self.env_wrapper.dist, librl.train.cc.maml_meta_step)

class ReinforceWithEntropyBonusTest(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.env_wrapper = AntWrapper()
        
    def setUp(self):
        x = functools.reduce(lambda x,y: x*y, self.env_wrapper.env.observation_space.shape, 1)
        self.policy_kernel = librl.nn.core.MLPKernel(x)
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

class PGBWithEntropyBonusTest(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.env_wrapper = AntWrapper()
        
    def setUp(self):
        x = functools.reduce(lambda x,y: x*y, self.env_wrapper.env.observation_space.shape, 1)
        self.value_kernel = librl.nn.core.MLPKernel(x)
        self.value_net = librl.nn.critic.ValueCritic(self.value_kernel)
        self.policy_kernel = librl.nn.core.MLPKernel(x)
        self.policy_net = librl.nn.actor.IndependentNormalActor(self.policy_kernel, self.env_wrapper.env.action_space, self.env_wrapper.env.observation_space)
        self.policy_loss = librl.nn.pg_loss.PGB(self.value_net, explore_bonus_fn=librl.reward.basic_entropy_bonus())

        self.agent = librl.agent.pg.ActorCriticAgent(self.value_net, self.policy_net, self.policy_loss)
        self.agent.train()

        self.env_wrapper.setUp(self.agent)
        
    def tearDown(self):
        self.env_wrapper.tearDown()
        del self.policy_kernel, self.policy_net, self.agent, self.value_kernel, self.value_net, self.policy_loss

    def test_policy_grad(self):
        librl.train.train_loop.cc_episodic_trainer(self.env_wrapper.hypers, self.env_wrapper.dist, librl.train.cc.policy_gradient_step)

    def test_maml(self):
        librl.train.train_loop.cc_episodic_trainer(self.env_wrapper.hypers, self.env_wrapper.dist, librl.train.cc.maml_meta_step)

class PPOWithEntropyBonusTest(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.env_wrapper = AntWrapper()
        
    def setUp(self):
        x = functools.reduce(lambda x,y: x*y, self.env_wrapper.env.observation_space.shape, 1)
        self.value_kernel = librl.nn.core.MLPKernel(x)
        self.value_net = librl.nn.critic.ValueCritic(self.value_kernel)
        self.policy_kernel = librl.nn.core.MLPKernel(x)
        self.policy_net = librl.nn.actor.IndependentNormalActor(self.policy_kernel, self.env_wrapper.env.action_space, self.env_wrapper.env.observation_space)
        self.policy_loss = librl.nn.pg_loss.PPO(self.value_net, explore_bonus_fn=librl.reward.basic_entropy_bonus())

        self.agent = librl.agent.pg.ActorCriticAgent(self.value_net, self.policy_net, self.policy_loss)
        self.agent.train()

        self.env_wrapper.setUp(self.agent)
        
    def tearDown(self):
        self.env_wrapper.tearDown()
        del self.policy_kernel, self.policy_net, self.agent, self.value_kernel, self.value_net, self.policy_loss

    def test_policy_grad(self):
        librl.train.train_loop.cc_episodic_trainer(self.env_wrapper.hypers, self.env_wrapper.dist, librl.train.cc.policy_gradient_step)

    def test_maml(self):
        librl.train.train_loop.cc_episodic_trainer(self.env_wrapper.hypers, self.env_wrapper.dist, librl.train.cc.maml_meta_step)

if __name__ == '__main__':
    unittest.main()