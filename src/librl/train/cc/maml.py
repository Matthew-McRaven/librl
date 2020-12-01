import functools

import librl.agent.pg, librl.agent.mdp
import librl.task
import librl.train.cc

@functools.singledispatch
def compute_meta_loss(agent, task):
    # If you introduce a new agent type, and you wish to allow this agent to be updated via
    # MAML, you must register an override for this function.
    # Must return an iterable of losses.
    raise NotImplemented(f"Meta loss computation not implemented for this agent type.")

@compute_meta_loss.register(librl.agent.mdp.RandomAgent)
def _(agent, task):
    return []

@compute_meta_loss.register(librl.agent.pg.REINFORCEAgent)
def _(agent, task):
    return (getattr(agent, 'actor_loss_mult', 1) * agent.actor_loss(task),)

@compute_meta_loss.register(librl.agent.pg.ActorCriticAgent)
def _(agent, task):
    return (agent.critic_loss(task), getattr(agent,'actor_loss_mult', 1) * agent.actor_loss(task))



@functools.singledispatch
def step_meta_optimizer(agent):
        # If you introduce a new agent type, and you wish to allow this agent to be updated via
    # MAML, you must register an override for this function.
    raise NotImplemented(f"Meta optimizer step not implemented for this agent type.")

@step_meta_optimizer.register(librl.agent.mdp.RandomAgent)
def _(agent):
    pass
@step_meta_optimizer.register(librl.agent.pg.REINFORCEAgent)
def _(agent):
    # TODO: Grab grad clip from agent's hypers.
    #torch.nn.utils.clip_grad_norm(agent.parameters(), 40)
    agent.actor_optimizer.step()
@step_meta_optimizer.register(librl.agent.pg.ActorCriticAgent)
def _(agent):
    # TODO: Grab grad clip from agent's hypers.
    #torch.nn.utils.clip_grad_norm(agent.parameters(), 40)
    agent.actor_optimizer.step()
    agent.critic_optimizer.step()


# Implement MAML algorithm for meta-RL tasks
# Assumes there is no interaction between different agents.
# If components of an agent are shared between tasks, gradient updates are no longer independent.
def maml_meta_step(task_samples, adapt_steps = 1):
    for task in task_samples: assert task.problem_type == librl.task.ProblemTypes.ContinuousControl
    # Collect all unique agents in our task list.
    agents = { (id(task.agent),task.agent) for task in task_samples}
    # Make sure we do not persist grads between training epochs.
    for _, agent in agents: agent.zero_grad()
    # Keep track of our starting parmeters.
    # Since we have potential redundant agents, only keep one copy of our params per agent.
    slow_parameters = {id_agent : agent.steal() for id_agent, agent in agents}
    # However, each task will have unique parameters after being adapted.
    size = len(task_samples)
    adapted_parameters, adapted_grads, adapted_rewards = size*[None], size*[None], size*[None]
    
    ####################
    # Task Adaptation  #
    ####################
    for idx, task in enumerate(task_samples):
        agent = task.agent
        agent.stuff(slow_parameters[id(agent)])
        # Perform an arbitrary number of adaptation steps
        for _ in range(adapt_steps): librl.train.cc.policy_gradient_step([task])
        
        # Sample a task for the meta-adaptation step and compute the loss based on agent type.
        task.sample(task)
        for loss in compute_meta_loss(agent, task): loss.backward()

        # All pytorch optimizers perform updates using grads.
        # If we copy out the grads and apply them later, pytorch won't be any wiser
        # as to how those grads got there. Use "clone" to prevent grads from being deleted.
        adapted_grads[idx] = [(p.grad.clone().detach() if p.grad != None else None)for p in agent.parameters()]
        adapted_rewards[idx] = functools.reduce(lambda x, y: x+sum(y.reward_buffer.view(-1)), task.trajectories, 0)

        # Prevent meta-grads from polluting the next task's update.
        agent.zero_grad()
        adapted_parameters[idx] = agent.steal()

    ####################
    # Meta-update step #
    ####################
    # We want to apply gradient updates to our starting weights,
    # and we want to discard any existing gradient updates
    for id_agent, agent in agents: agent.stuff(slow_parameters[id_agent]), agent.zero_grad()

    # Apply the i'th task's gradient update to the i'th task's agent.
    for idx, task in enumerate(task_samples):
        agent = task.agent
        for param, grad in zip(agent.parameters(), adapted_grads[idx]):
            # Some parts of a model may be frozen, so we can skip those params.
            if not param.requires_grad or grad is None: continue
            # If our grad object doesn't exist, we must create one in place.
            if param.grad is None: param.grad = grad
            # Otherwise accumulate grads in place.
            else: param.grad.add_(grad)
    
    # And apply those using our existing optimizer.
    for _, agent in agents: step_meta_optimizer(agent)
