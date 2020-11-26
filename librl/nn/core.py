import functools
import types

import more_itertools
import torch
import torch.nn as nn
import torch.optim

# Agent network based on a submission to my COSC689 class
# It is a stochastic policy network. It will return the policy from forward,
# and you can use this policy to generate further samples
# The current policy is sampling random values from a torch Categorical distrubtion
# conditioned on the output of a linear network.
# The Categorical distribution is non-differentiable, so this may cause
# problems for future programmers.
class MLPKernel(nn.Module):
    def __init__(self, input_dimensions, layer_list=None, dropout=None):
        
        super(MLPKernel, self).__init__()
        dropout = dropout if dropout else self.get_default_hyperparameters().dropout
        layer_list = layer_list if layer_list else self.get_default_hyperparameters().layer_list
        self.input_dimensions = list(more_itertools.always_iterable(input_dimensions))
        self.__input__size = functools.reduce(lambda x, y: x*y, self.input_dimensions, 1)

        # Build linear layers from input defnition.
        linear_layers = []
        previous = self.__input__size
        for index,layer in enumerate(layer_list):
            linear_layers.append(nn.Linear(previous, layer))
            linear_layers.append(nn.LeakyReLU())
            # We have an extra component at the end, so we can dropout after every layer.
            linear_layers.append(nn.Dropout(dropout))
            previous = layer
        self.output_dimension = (previous, )
        self.linear_layers = nn.Sequential(*linear_layers)

        # Initialize NN
        for x in self.parameters():
            if x.dim() > 1:
                nn.init.kaiming_normal_(x)
    def recurrent(self):
        return False
    @staticmethod
    def get_default_hyperparameters():
        ret = {}
        ret['dropout'] = .1
        ret['layer_list'] = [200, 100]
        return types.SimpleNamespace(**ret)

    def forward(self, input):
        input = input.view(-1, self.__input__size)
        # Push observations through feed forward layers.
        output = self.linear_layers(input.float())

        assert not torch.isnan(output).any()

        return output

class LSTMKernel(nn.Module):
    def __init__(self, input_dimensions, hidden_size, num_layers, bidirectional=False, dropout=None):
        super(LSTMKernel, self).__init__()
        dropout = dropout if dropout else self.get_default_hyperparameters().dropout
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.input_dimensions = list(more_itertools.always_iterable(input_dimensions))
        self.__input__size = functools.reduce(lambda x, y: x*y, self.input_dimensions, 1)
        self.output_dimension = (hidden_size, )


        self.recurrent_layer = nn.LSTM(self.__input__size, hidden_size, num_layers = num_layers, bidirectional=bidirectional)
        self.init_hidden()
        # Initialize NN
        for x in self.parameters():
            if x.dim() > 1:
                nn.init.kaiming_normal_(x)

    def init_hidden(self):
        self.hidden_state = torch.zeros(self.num_layers, 1, self.hidden_size)
        self.cell_state = torch.zeros(self.num_layers, 1, self.hidden_size)
        nn.init.kaiming_normal_(self.hidden_state)
        nn.init.kaiming_normal_(self.cell_state)

    def recurrent(self):
        return True

    def save_hidden(self):
        return self.hidden_state

    def restore_hidden(self, state=None):
        if state == None:
            self.init_hidden()
        else:
            assert 0
            hidden_state, cell_state = state
            # Assert that state be correct shape for calling forward().
            # Assert here rather than forward(), since it may not be obvious why forward fails
            # when cell state is of wrong shape.
            hidden_state = hidden_state.view(self.num_layers, -1, self.hidden_size)
            cell_state = cell_state.view(self.num_layers, -1, self.hidden_size)

            self.hidden_state, self.cell_state = hidden_state, cell_state

    @staticmethod
    def get_default_hyperparameters():
        ret = {}
        ret['dropout'] = .1
        ret['layer_list'] = [200, 100]

        return types.SimpleNamespace(**ret)
    def forward(self, input):
        # TODO: Figure out how to batch hidden states
        input = input.view(1, -1, self.__input__size)
        # Push observations through feed forward layers.
        h0 = self.hidden_state, self.cell_state
        output, h1 = self.recurrent_layer(input.float(), h0)
        self.hidden_state, self.cell_state = h1
        # We really dont care about the progress / history of our state data.
        self.hidden_state, self.cell_state = self.hidden_state.detach(), self.cell_state.detach()
        assert not torch.isnan(output).any()

        return output