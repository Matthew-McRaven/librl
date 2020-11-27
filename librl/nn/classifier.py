import functools

import more_itertools
import torch
import torch.nn as nn
import torch.optim


# A simple NN designed for classification.
# It performs not output normalization. 
class Classifier(nn.Module):
    def __init__(self, neural_module, output_dimension):
        super(Classifier, self).__init__()
        self.neural_module = neural_module
        self.input_dimension = list(more_itertools.always_iterable(neural_module.output_dimension))

        self.output_dimension = output_dimension
        self.__input_size = functools.reduce(lambda x,y: x*y, self.input_dimension, 1)
        self.output_layer = nn.Linear(self.__input_size, self.output_dimension)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.003, weight_decay=0.1)
    
    def recurrent(self):
        return self.neural_module.recurrent()

    def save_hidden(self):
        assert self.recurrent()
        return self.neural_module.save_hidden()

    def restore_hidden(self, state=None):
        assert self.recurrent()
        self.neural_module.restore_hidden(state)

    def forward(self, inputs):
        # Require that outputs be of the shape specified by our neural module.
        outputs = self.neural_module(inputs).view(-1, *self.input_dimension)
        # Then cast it to something usable by a MLP.
        outputs = outputs.view(-1, self.__input_size)
        outputs = self.output_layer(outputs)

        return outputs
