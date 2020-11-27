import gym

import librl.agent.pg, librl.agent.mdp
import librl.nn.core, librl.nn.critic, librl.nn.actor
import librl.task, librl.hypers

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