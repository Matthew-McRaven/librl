from os import stat
import types

import torch
import torch.distributions, torch.nn.init
import torch.optim
from overrides import overrides, EnforceOverrides

import librl.calc
import librl.task
##########################
# Policy Gradient Losses #
##########################

# Implements a framework for policy gradient losses that use  entropy bonuses when computing discounted rewards.
class PolicyLossWithEntropyBonus(EnforceOverrides):

    def __init__(self, gamma, beta):
        self.gamma = gamma
        self.beta = beta

    # Compute discounted rewards with entropy bonus
    def get_discounted_rewards(self, trajectory):
        entropy = -self.beta * (trajectory.logprob_buffer[:trajectory.done]) * (trajectory.logprob_buffer[:trajectory.done].exp())
        bonus_rewards = trajectory.reward_buffer[:trajectory.done] + entropy
        return librl.calc.discounted_returns(bonus_rewards, gamma=self.gamma)

    # Override this method to compute the loss of a single trajectory.
    # Should return a single value.
    def compute_trajectory_loss(self, trajectory):
        raise NotImplemented("Must implement this in subclass")

    # When called with a CC task, run the loss algorithm over each trajectory in the task.
    # This will implicitly perform the outer mean over the the number of trajectories,
    # and defer the computation of per-trajectory loss to a subclass via compute_trajectory_loss.
    def __call__(self, task):
        assert isinstance(task, librl.task.ContinuousControlTask)
        losses = []
        for trajectory in task.trajectories: losses.append(self.compute_trajectory_loss(trajectory))
        return sum(losses) / len(losses)

# Vanilla policy gradient update / loss function.
class VPG(PolicyLossWithEntropyBonus):
    def __init__(self, gamma=.95, beta=0.01):
        super(VPG, self).__init__(gamma, beta)
        
    @overrides
    def compute_trajectory_loss(self, trajectory):
        return sum(trajectory.logprob_buffer[:trajectory.done] * self.get_discounted_rewards(trajectory))
       
# Policy gradient with baseline update / loss function.
class PGB(PolicyLossWithEntropyBonus):
    def __init__(self, critic_net,  gamma=.95, beta=.01):
        super(PGB, self).__init__(gamma, beta)
        self.critic_net = critic_net

    @overrides
    def compute_trajectory_loss(self, trajectory):
        # Don't propogate gradients into critic when updating actor.
        with torch.no_grad(): estimated_values = self.critic_net(trajectory.state_buffer).view(-1)[:trajectory.done]
        return sum(trajectory.logprob_buffer[:trajectory.done] * (self.get_discounted_rewards(trajectory)-estimated_values))

# Proximal policy optimization update / loss function.
class PPO(PolicyLossWithEntropyBonus):

    def __init__(self, critic_net, gamma=0.975, beta=0.01, lambd=.99, epsilon=.5, c_1=1):
        super(PPO, self).__init__(gamma, beta)
        self.critic_net = critic_net
        self.lambd = lambd
        self.epsilon = epsilon
        self.c_1 = c_1

    @overrides
    def compute_trajectory_loss(self, trajectory):
        # Don't propogate gradients into critic when updating actor.
        with torch.no_grad(): estimated_values = self.critic_net(trajectory.state_buffer).view(-1)[:trajectory.done]
        discounted = self.get_discounted_rewards(trajectory)
        A =  librl.calc.gae(trajectory.reward_buffer[:trajectory.done], estimated_values, self.gamma)
        log_prob_old = librl.calc.old_log_probs(trajectory.action_buffer[:trajectory.done], trajectory.policy_buffer[:trajectory.done])
        # Compute indiviudal terms of the PPO algorithm.
        ratio = trajectory.logprob_buffer[:trajectory.done].exp() / (log_prob_old.exp() + 1e-6)
        lhs, rhs = ratio * A, torch.clamp(ratio, 1-self.epsilon, 1+self.epsilon) * A
        minterm = torch.min(lhs, rhs)
        subterm = self.c_1 * (discounted-estimated_values.view(-1)).pow(2)
        return torch.sum(minterm - subterm)