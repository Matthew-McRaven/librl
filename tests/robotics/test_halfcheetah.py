import functools
import unittest

from .bidirectional_halfcheetah import HalfCheetahDirecBulletEnv, HalfCheetahTask
import librl.agent.pg, librl.agent.mdp
import librl.nn.core, librl.nn.critic, librl.nn.actor
import librl.task, librl.hypers
import librl.train.train_loop
import librl.train.cc.pg, librl.train.cc.maml

class CheetahEnvWrapper:
    def __init__(self):
        self.hypers = librl.hypers.get_default_hyperparams()
        self.hypers['device'] = 'cpu'
        self.hypers['epochs'] = 5
        self.hypers['task_count'] = 2
        self.env = HalfCheetahDirecBulletEnv()
        self.env._max_episode_steps = self.hypers['episode_length']
    def setUp(self, agent):
        self.dist = librl.task.TaskDistribution()
        self.dist.add_task(librl.task.Task.Definition(HalfCheetahTask, forward=True,  env=self.env, agent=agent, episode_length=53))
        self.dist.add_task(librl.task.Task.Definition(HalfCheetahTask, forward=False, env=self.env, agent=agent, episode_length=53))
    def tearDown(self):
        del self.dist

class RandomTest(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.env_wrapper = CheetahEnvWrapper()
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

class ReinforceTest(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.env_wrapper = CheetahEnvWrapper()
        
    def setUp(self):
        x = functools.reduce(lambda x,y: x*y, self.env_wrapper.env.observation_space.shape, 1)
        self.policy_kernel = librl.nn.core.MLPKernel(x)
        self.policy_net = librl.nn.actor.IndependentNormalActor(self.policy_kernel, self.env_wrapper.env.action_space, self.env_wrapper.env.observation_space)

        self.agent = librl.agent.pg.REINFORCEAgent(self.policy_net)
        self.agent.train()

        self.env_wrapper.setUp(self.agent)
        
    def tearDown(self):
        self.env_wrapper.tearDown()
        del self.policy_kernel, self.policy_net, self.agent

    def test_policy_grad(self):
        librl.train.train_loop.cc_episodic_trainer(self.env_wrapper.hypers, self.env_wrapper.dist, librl.train.cc.policy_gradient_step)

    def test_maml(self):
        librl.train.train_loop.cc_episodic_trainer(self.env_wrapper.hypers, self.env_wrapper.dist, librl.train.cc.maml_meta_step)

class PGBTest(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.env_wrapper = CheetahEnvWrapper()
        
    def setUp(self):
        x = functools.reduce(lambda x,y: x*y, self.env_wrapper.env.observation_space.shape, 1)
        self.value_kernel = librl.nn.core.MLPKernel(x)
        self.value_net = librl.nn.critic.ValueCritic(self.value_kernel)
        self.policy_kernel = librl.nn.core.MLPKernel(x)
        self.policy_net = librl.nn.actor.IndependentNormalActor(self.policy_kernel, self.env_wrapper.env.action_space, self.env_wrapper.env.observation_space)
        self.policy_loss = librl.nn.pg_loss.PGB(self.value_net)

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

class PPOTest(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.env_wrapper = CheetahEnvWrapper()
        
    def setUp(self):
        x = functools.reduce(lambda x,y: x*y, self.env_wrapper.env.observation_space.shape, 1)
        self.value_kernel = librl.nn.core.MLPKernel(x)
        self.value_net = librl.nn.critic.ValueCritic(self.value_kernel)
        self.policy_kernel = librl.nn.core.MLPKernel(x)
        self.policy_net = librl.nn.actor.IndependentNormalActor(self.policy_kernel, self.env_wrapper.env.action_space, self.env_wrapper.env.observation_space)
        self.policy_loss = librl.nn.pg_loss.PPO(self.value_net)

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