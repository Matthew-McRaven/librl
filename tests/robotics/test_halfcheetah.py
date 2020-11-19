import functools
import unittest

from .maml_env import HalfCheetahDirecBulletEnv, HalfCheetahTask
import librl.agent.pg
import librl.nn.core, librl.nn.critic, librl.nn.actor
import librl.task, librl.hypers
import librl.train.train_loop
import librl.train.cc.pg, librl.train.cc.maml

class ReinforceTest(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.hypers = librl.hypers.get_default_hyperparams()
        self.hypers['device'] = 'cpu'
        self.hypers['epochs'] = 5
        self.hypers['episode_length'] = 53
        self.hypers['episode_count']  = 3
        self.hypers['adapt_steps'] = 1
        self.hypers['actor_loss_mul'] = -1
        self.env = HalfCheetahDirecBulletEnv()
        self.env._max_episode_steps = self.hypers['episode_length']
        
    def setUp(self):
        x = functools.reduce(lambda x,y: x*y, self.env.observation_space.shape, 1)
        self.policy_kernel = librl.nn.core.MLPKernel(x)
        self.policy_net = librl.nn.actor.IndependentNormalActor(self.policy_kernel, self.env.action_space, self.env.observation_space)

        self.agent = librl.agent.pg.REINFORCEAgent(self.hypers, self.policy_net)
        self.agent.train()
        self.dist = librl.task.TaskDistribution()
        self.dist.add_task(librl.task.Task.Definition(HalfCheetahTask, forward=True,  env=self.env, agent=self.agent))
        self.dist.add_task(librl.task.Task.Definition(HalfCheetahTask, forward=False, env=self.env, agent=self.agent))
        
    def tearDown(self):
        del self.policy_kernel
        del self.policy_net
        del self.agent
        del self.dist

    def test_policy_grad(self):
        librl.train.train_loop.cc_episodic_trainer(self.hypers, self.dist, librl.train.cc.policy_gradient_step)

    def test_maml(self):
        librl.train.train_loop.cc_episodic_trainer(self.hypers, self.dist, librl.train.cc.maml_meta_step)

class PGBTest(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.hypers = librl.hypers.get_default_hyperparams()
        self.hypers['device'] = 'cpu'
        self.hypers['epochs'] = 5
        self.hypers['episode_length'] = 53
        self.hypers['episode_count']  = 3
        self.hypers['adapt_steps'] = 1
        self.hypers['actor_loss_mul'] = -1
        self.env = HalfCheetahDirecBulletEnv()
        self.env._max_episode_steps = self.hypers['episode_length']
        
    def setUp(self):
        x = functools.reduce(lambda x,y: x*y, self.env.observation_space.shape, 1)
        self.value_kernel = librl.nn.core.MLPKernel(x)
        self.value_net = librl.nn.critic.ValueCritic(self.value_kernel, self.hypers)
        self.policy_kernel = librl.nn.core.MLPKernel(x)
        self.policy_net = librl.nn.actor.IndependentNormalActor(self.policy_kernel, self.env.action_space, self.env.observation_space)
        self.policy_loss = librl.nn.pg_loss.PGB(self.value_net, self.hypers['gamma'])

        self.agent = librl.agent.pg.ActorCriticAgent(self.hypers, self.value_net, self.policy_net, self.policy_loss)
        self.agent.train()
        self.dist = librl.task.TaskDistribution()

        self.dist.add_task(librl.task.Task.Definition(HalfCheetahTask, forward=True,  env=self.env, agent=self.agent))
        self.dist.add_task(librl.task.Task.Definition(HalfCheetahTask, forward=False, env=self.env, agent=self.agent))
        
    def tearDown(self):
        del self.policy_kernel
        del self.policy_net
        del self.agent
        del self.dist

    def test_policy_grad(self):
        librl.train.train_loop.cc_episodic_trainer(self.hypers, self.dist, librl.train.cc.policy_gradient_step)

    def test_maml(self):
        librl.train.train_loop.cc_episodic_trainer(self.hypers, self.dist, librl.train.cc.maml_meta_step)

class PPOTest(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.hypers = librl.hypers.get_default_hyperparams()
        self.hypers['device'] = 'cpu'
        self.hypers['epochs'] = 5
        self.hypers['episode_length'] = 53
        self.hypers['episode_count']  = 3
        self.hypers['adapt_steps'] = 1
        self.hypers['actor_loss_mul'] = -1
        self.env = HalfCheetahDirecBulletEnv()
        self.env._max_episode_steps = self.hypers['episode_length']
        
    def setUp(self):
        x = functools.reduce(lambda x,y: x*y, self.env.observation_space.shape, 1)
        self.value_kernel = librl.nn.core.MLPKernel(x)
        self.value_net = librl.nn.critic.ValueCritic(self.value_kernel, self.hypers)
        self.policy_kernel = librl.nn.core.MLPKernel(x)
        self.policy_net = librl.nn.actor.IndependentNormalActor(self.policy_kernel, self.env.action_space, self.env.observation_space)
        self.policy_loss = librl.nn.pg_loss.PPO(self.value_net, self.hypers['gamma'])

        self.agent = librl.agent.pg.ActorCriticAgent(self.hypers, self.value_net, self.policy_net, self.policy_loss)
        self.agent.train()
        self.dist = librl.task.TaskDistribution()
        
        self.dist.add_task(librl.task.Task.Definition(HalfCheetahTask, forward=True,  env=self.env, agent=self.agent))
        self.dist.add_task(librl.task.Task.Definition(HalfCheetahTask, forward=False, env=self.env, agent=self.agent))
        
    def tearDown(self):
        del self.policy_kernel
        del self.policy_net
        del self.agent
        del self.dist

    def test_policy_grad(self):
        librl.train.train_loop.cc_episodic_trainer(self.hypers, self.dist, librl.train.cc.policy_gradient_step)

    def test_maml(self):
        librl.train.train_loop.cc_episodic_trainer(self.hypers, self.dist, librl.train.cc.maml_meta_step)


if __name__ == '__main__':
    unittest.main()