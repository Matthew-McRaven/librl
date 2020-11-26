import torch

import librl.calc
###################################
# Exploration Bonuses Computation #
###################################

# Compute the entropy of a trajectory, which is just p(x)log(p(x))
class basic_entropy_bonus:
    def __init__(self, beta=0.01):
        self.beta = beta
    def __call__(self, trajectory):
        return -self.beta * trajectory.logprob_buffer[:trajectory.done] * trajectory.logprob_buffer[:trajectory.done].exp()

##############################
# Reward Discounting Methods #
##############################
# Different reward functions taken from:
#   https://danieltakeshi.github.io/2017/04/02/notes-on-the-generalized-advantage-estimation-paper/

# The reward at each timestep is the sum of all rewards over the epoch
def total_reward(trajectory):
    return torch.full(*trajectory.reward_buffer.shape, torch.sum(trajectory.reward_buffer)) # type: ignore

# At each timestep, compute the infinite-horizon discounted reward.
class to_go_reward:
    def __init__(self, gamma=.975):
        self.gamma = gamma
    def __call__ (self, trajectory):
        return librl.calc.discounted_returns(trajectory.reward_buffer[:trajectory.done], gamma=self.gamma)

# At each timestep, compute the infinite-horizon discounted reward and subtract a baseline generate by a critic nn.
class baseline_to_go(to_go_reward):
    def __init__(self, critic_fn, gamma=.975):
        super(baseline_to_go, self).__init__(gamma=gamma)
        self.critic_fn = critic_fn
    def __call__ (self, trajectory):
        with torch.no_grad(): estimated_values = self.critic_fn(trajectory.state_buffer).view(-1)[:trajectory.done]
        return super().__call__(trajectory) - estimated_values

# Compute the temporal difference residual of the rewards.
class td_residual:
    def __init__(self, critic_fn, gamma=.975):
        self.critic_fn = critic_fn
        self.gamma = gamma
    def __call__(self, trajectory):
        with torch.no_grad(): estimated_values = self.critic_fn(trajectory.state_buffer).view(-1)[:trajectory.done]
        return librl.calc.td_residual(trajectory.reward_buffer[:trajectory.done], estimated_values, gamma=self.gamma)