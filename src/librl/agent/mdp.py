
import functools

import more_itertools
import torch
import torch.nn as nn

import librl.agent
# The random agent random selects one edge pair to toggle per timestep.
@librl.agent.add_agent_attr()
class RandomAgent(nn.Module):
    def __init__(self, observation_space, action_space):
        # Must initialize torch.nn.Module
        super(RandomAgent, self).__init__()
        self.action_space = action_space
        self.input_dimension = list(more_itertools.always_iterable(observation_space.shape))
        self.__input_size = functools.reduce(lambda x,y: x*y, self.input_dimension, 1)

    # Our action is just asking the pytorch implementation for a random set of nodes.
    def act(self, inputs):
        return self.forward(inputs)

    # Implement required pytorch interface
    def forward(self, adj):
        count = adj.view(-1, self.__input_size).shape[0]
        actions = [self.action_space.sample() for _ in range(count)]
        randoms = torch.tensor(actions, device=adj.device) # type: ignore
        # Currently makes little sense to ask "what was the probability of drawing this random action"
        # with gym spaces. TODO: Make log prob a real, useful number.
        return randoms, torch.full((count,), float('nan')) # type: ignore

    # Random agent has no parameters.
    def steal(self):
        return {}
    def stuff(self, vals):
        pass