import torch
import torch.distributions, torch.nn.init
import torch.optim

import librl.calc
##########################
# Policy Gradient Losses #
##########################

# Vanilla policy gradient update / loss function.
class VPG:
    def __init__(self, gamma, beta=0.01):
        self.gamma = gamma
        self.beta = beta

    def __call__(self, task):
        losses = []
        for trajectory in task.trajectories:
            entropy = -self.beta * (trajectory.logprob_buffer[:trajectory.done]) * (trajectory.logprob_buffer[:trajectory.done].exp())
            bonus_rewards = trajectory.reward_buffer[:trajectory.done] + entropy
            discounted = librl.calc.discounted_returns(bonus_rewards, gamma=self.gamma)
            losses.append(sum(trajectory.logprob_buffer[:trajectory.done] * discounted))
        # Perform outer mean.
        return sum(losses) / len(losses)
        
# Policy gradient with baseline update / loss function.
class PGB:
    def __init__(self, critic_net,  gamma, beta=0.01):
        self.critic_net = critic_net
        self.gamma = gamma
        self.beta = beta

    def __call__(self, task):
        losses = []
        for trajectory in task.trajectories:
            # Don't propogate gradients into critic when updating actor.
            with torch.no_grad(): estimated_values = self.critic_net(trajectory.state_buffer).view(-1)[:trajectory.done]
            entropy = -self.beta * (trajectory.logprob_buffer[:trajectory.done]) * (trajectory.logprob_buffer[:trajectory.done].exp())
            bonus_rewards = trajectory.reward_buffer[:trajectory.done] + entropy
            discounted = librl.calc.discounted_returns(bonus_rewards, gamma=self.gamma)
            losses.append(torch.sum(trajectory.logprob_buffer[:trajectory.done] * (discounted - estimated_values)))
        # Perform outer mean.
        return sum(losses) / len(losses)

# Proximal policy optimization update / loss function.
class PPO:
    def __init__(self, critic_net, gamma, lambd=.99, epsilon=.5, c_1=1, beta=0.01):
        self.critic_net = critic_net
        self.gamma = gamma
        self.lambd = lambd
        self.epsilon = epsilon
        self.c = c_1
        self.beta = beta

    def __call__(self, task):
        losses = []
        for trajectory in task.trajectories:
            # Don't propogate gradients into critic when updating actor.
            with torch.no_grad(): estimated_values = self.critic_net(trajectory.state_buffer[:trajectory.done]).view(-1)

            entropy = -self.beta * (trajectory.logprob_buffer[:trajectory.done]) * (trajectory.logprob_buffer[:trajectory.done].exp())
            bonus_rewards = trajectory.reward_buffer[:trajectory.done] + entropy
            # Use helper methods to vectorize dicount, GAE, and log_prob computations.
            discounted = librl.calc.discounted_returns(bonus_rewards, gamma=self.gamma)
            A =  librl.calc.gae(trajectory.reward_buffer[:trajectory.done], estimated_values, self.gamma)
            log_prob_old = librl.calc.old_log_probs(trajectory.action_buffer[:trajectory.done], trajectory.policy_buffer[:trajectory.done])

            # Compute indiviudal terms of the PPO algorithm.
            ratio = trajectory.logprob_buffer[:trajectory.done].exp() / (log_prob_old.exp() + 1e-6)
            lhs, rhs = ratio * A, torch.clamp(ratio, 1-self.epsilon, 1+self.epsilon) * A
            minterm = torch.min(lhs, rhs)
            subterm = self.c * (discounted-estimated_values.view(-1)).pow(2)
            losses.append(torch.sum(minterm - subterm))
        # Perform outer mean.
        return sum(losses) / len(losses)