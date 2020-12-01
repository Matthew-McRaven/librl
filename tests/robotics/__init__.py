import functools
from librl import agent
import librl.agent.pg, librl.agent.mdp
import librl.nn.core, librl.nn.critic, librl.nn.actor
import librl.task

def RandomAgent(env, hypers, explore_bonus=None):
    return librl.agent.mdp.RandomAgent(env.observation_space, env.action_space)

def VPGAgent(env, hypers, explore_bonus=lambda x:0):
    x = functools.reduce(lambda x,y: x*y, env.observation_space.shape, 1)
    policy_kernel = librl.nn.core.MLPKernel(x)
    policy_net = librl.nn.actor.IndependentNormalActor(policy_kernel, env.action_space, env.observation_space)

    agent = librl.agent.pg.REINFORCEAgent(policy_net, explore_bonus_fn=explore_bonus)
    agent.train()
    return agent

def PGBAgent(env, hypers, explore_bonus=lambda x:0):
    x = functools.reduce(lambda x,y: x*y, env.observation_space.shape, 1)
    value_kernel = librl.nn.core.MLPKernel(x)
    value_net = librl.nn.critic.ValueCritic(value_kernel)
    policy_kernel = librl.nn.core.MLPKernel(x)
    policy_net = librl.nn.actor.IndependentNormalActor(policy_kernel, env.action_space, env.observation_space)
    policy_loss = librl.nn.pg_loss.PGB(value_net, explore_bonus_fn=explore_bonus)

    agent = librl.agent.pg.ActorCriticAgent(value_net, policy_net, policy_loss)
    agent.train()
    return agent

def PPOAgent(env, hypers, explore_bonus=lambda x:0):
    x = functools.reduce(lambda x,y: x*y, env.observation_space.shape, 1)
    value_kernel = librl.nn.core.MLPKernel(x)
    value_net = librl.nn.critic.ValueCritic(value_kernel)
    policy_kernel = librl.nn.core.MLPKernel(x)
    policy_net = librl.nn.actor.IndependentNormalActor(policy_kernel, env.action_space, env.observation_space)
    policy_loss = librl.nn.pg_loss.PPO(value_net, explore_bonus_fn=explore_bonus)

    agent = librl.agent.pg.ActorCriticAgent(value_net, policy_net, policy_loss)
    agent.train()

    return agent    
def ant_dist(env, hypers, agent):
    dist = librl.task.distribution.TaskDistribution()
    dist.add_task(librl.task.Task.Definition(librl.task.ContinuousGymTask, env=env, agent=agent, episode_length=hypers['episode_length']))
    return dist

class HalfCheetahTask(librl.task.ContinuousControlTask):
    def __init__(self, forward=True, **kwargs):
        super(HalfCheetahTask, self).__init__(**kwargs)
        self.forward = forward
    def init_env(self):
        self.env.set_direction(self.forward)

def cheetah_dist(env, hypers, agent):
    dist = librl.task.distribution.TaskDistribution()
    dist.add_task(librl.task.Task.Definition(HalfCheetahTask, forward=True,  env=env, agent=agent, episode_length=hypers['episode_length']))
    dist.add_task(librl.task.Task.Definition(HalfCheetahTask, forward=False, env=env, agent=agent, episode_length=hypers['episode_length']))
    return dist