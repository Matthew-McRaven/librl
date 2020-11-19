import functools

import librl.agent.pg
import librl.task
import librl.train.cc

@functools.singledispatch
def policy_gradient_update(agent, episode_iterable):
    # If you introduce a new agent type, and you wish to allow this agent to be updated via
    # policy gradient, you must register an override for this function.
    raise NotImplemented(f"Policy gradient update not implemented for this agent type.")

@policy_gradient_update.register(librl.agent.pg.REINFORCEAgent)
def _(agent, tasks_iterable):
    librl.train.cc.update_unshared_cc_agent(agent, tasks_iterable, 1, 'actor_loss', -1, 'actor_optimizer')

@policy_gradient_update.register(librl.agent.pg.ActorCriticAgent)
def _(agent, tasks_iterable):
    librl.train.cc.update_unshared_cc_agent(agent, tasks_iterable, agent.hypers['critic_steps'], 'critic_loss', 1, 'critic_optimizer')
    librl.train.cc.update_unshared_cc_agent(agent, tasks_iterable, 1, 'actor_loss', -1, 'actor_optimizer')

# Assumes there is no interaction between different agents.
# If components of an agent are shared between tasks, gradient updates are no longer independent.
def policy_gradient_step(task_samples):
    # Determine which agents are present in this training run.
    agents = { (id(task.agent),task.agent) for task in task_samples}
    for task in task_samples:
        assert task.problem_type == librl.task.ProblemTypes.ContinuousControl
        task.sample(task)

    # Collate tasks by the agent running them.
    collated_tasks = {id_agent:[] for id_agent, _ in agents}
    for task in task_samples: collated_tasks[id(task.agent)].append(task)
    for id_agent, agent in agents: policy_gradient_update(agent, collated_tasks[id_agent])