import functools

import torch

import librl.replay
from .maml import maml_meta_step, compute_meta_loss, step_meta_optimizer
from .pg import policy_gradient_step, policy_gradient_update
def update_unshared_cc_agent(agent, tasks, it, loss_fn, loss_mul, optim):
    for i in range(it):
        # Iterate across all tasks, accumulating losses using the current task's agent's loss function.
        # Must init reduce with 0 or else first x is an task, not a scalar.
        # Provide lambda returning 0 in case the specified loss function DNE.
        losses = functools.reduce(lambda x,y: x + getattr(agent, loss_fn, lambda _: 0)(y), tasks, 0)
        # Compute mean and apply optional loss multiplier.
        losses *= (loss_mul / functools.reduce(lambda x,y: x + y.trajectory_count, tasks, 0))
        losses.backward()

        torch.nn.utils.clip_grad_norm_(agent.parameters(), 5)
        getattr(agent, optim).step(), agent.zero_grad()