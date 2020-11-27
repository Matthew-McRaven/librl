import collections
import functools
import types

import more_itertools
import torch
import torch.nn as nn
import torch.optim

# Determine the size of a dimension after applying a pool / convolutional layer.
def resize_convolution(x, kernel_size, dilation, stride, padding):
    x = int(1 + (x + 2*padding - dilation * (kernel_size - 1) - 1)/stride)
    return x

# Determine the size of a dimension after applying a transposed convolution layer.
def resize_transpose_convolution(x, kernel_size, dilation, stride, padding, output_padding):
    t1 = (x-1)*stride
    t2 = 2*padding
    t3 = dilation*(kernel_size-1)
    t4 = output_padding
    return t1 - t2 + t3 + t4 + 1
    
# Class containing overlapping parameters been convolutional, pooling layers
class conpool_core:
    def __init__(self, kernel, stride, padding, dilation, non_linear_after):
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.non_linear_after = non_linear_after

# Class describing a 2D convolutional layer.
class conv_def(conpool_core):
    def __init__(self, kernel, out_channels, stride=1, padding=0, dilation=1, non_linear_after=True, padding_type='zeros'):
        super(conv_def, self).__init__(kernel, stride, padding, dilation, non_linear_after)
        self.out_channels = out_channels
        self.padding_type = padding_type

# Class describing a transposed 2d convolutional layer.
class conv_transpose_def(conpool_core):
    def __init__(self, kernel, out_channels, stride=1, padding=0, dilation=1, output_padding=0, non_linear_after=True):
        super(conv_transpose_def, self).__init__(kernel, stride, padding, dilation, non_linear_after)
        self.out_channels = out_channels
        self.output_padding = output_padding
        self.padding_type = 'zeros'

# Class describing a 2D pooling layer.
class pool_def(conpool_core):
    def __init__(self, kernel, stride=None, padding=0, dilation=1, non_linear_after=False, pool_type='avg'):
        super(pool_def, self).__init__(kernel, stride, padding, dilation, non_linear_after)
        if stride is None:
            self.stride = self.kernel
        self.pool_type = pool_type

def construct_layers(conv_layers, input_dimensions, input_channels, dropout):
        # Construct convolutional layers.
        H = input_dimensions[0]
        W = input_dimensions[1]
        conv_list = []
        non_linear = torch.nn.LeakyReLU
        nlo_name = "ReLU"
        output_channels = input_channels
        output_dimension = (*input_dimensions, input_channels)

        # Iterate over all pooling/convolutional layer configurations.
        # Construct all items as a (name, layer) tuple so that the layers may be loaded into
        # an ordered dictionary. Ordered dictionaries respect the order in which items were inserted,
        # and are the least painful way to construct a nn.Sequential object.
        for index, item in enumerate(conv_layers):
            # Next item is a convolutional layer, so construct one and re-compute H,W, channels.
            if isinstance(item, conv_def):
                conv_list.append((f'conv{index}', nn.Conv2d(input_channels, item.out_channels, item.kernel,
                    stride=item.stride, padding=item.padding, dilation=item.dilation, padding_mode=item.padding_type)))
                H = resize_convolution(H, item.kernel, item.dilation, item.stride, item.padding)
                W = resize_convolution(W, item.kernel, item.dilation, item.stride, item.padding)
                output_channels = item.out_channels
            # Next item is a transposed convolutional layer.
            if isinstance(item, conv_transpose_def):
                conv_list.append((f'conv{index}', nn.ConvTranspose2d(input_channels, item.out_channels, item.kernel,
                    stride=item.stride, padding=item.padding, dilation=item.dilation, padding_mode=item.padding_type, output_padding=item.output_padding)))
                H = resize_transpose_convolution(H, item.kernel, item.dilation, item.stride, item.padding, item.output_padding)
                W = resize_transpose_convolution(W, item.kernel, item.dilation, item.stride, item.padding, item.output_padding)
                output_channels = item.out_channels
            # Next item is a pooling layer, so construct one and re-compute H,W.
            elif isinstance(item, pool_def):
                if item.pool_type.lower() == 'avg':
                    conv_list.append((f'avgpool{index}',nn.AvgPool2d(item.kernel, stride=item.stride, padding=item.padding)))
                    H = resize_convolution(H, item.kernel, 1, item.stride, item.padding)
                    W = resize_convolution(W, item.kernel, 1, item.stride, item.padding)
                elif item.pool_type.lower() == 'max':
                    conv_list.append((f'maxpool{index}', nn.MaxPool2d(item.kernel, stride=item.stride, padding=item.padding, dilation=item.dilation)))
                    H = resize_convolution(H, item.kernel, item.dilation, item.stride, item.padding)
                    W = resize_convolution(W, item.kernel, item.dilation, item.stride, item.padding)
                else:
                    raise NotImplementedError(f"{item.pool_type.lower()} is not an implemented form of pooling.")
            output_dimension = (output_channels, H, W)
            # Add a non-linear operator if specified by item. Non linear operators also pair with dropout
            # in all the examples I've seen
            if item.non_linear_after:
                conv_list.append((f"{nlo_name}{index}", non_linear))
                conv_list.append((f"dropout{index}", nn.Dropout(dropout)))
            #if print_sizes: print(f"Layer {index} is ({H} x {W})")
        return conv_list, output_dimension

class ConvolutionalKernel(nn.Module):
    def __init__(self, conv_layers, input_dimensions, input_channels, dropout=None):
        conv_list, out_dims = construct_layers(conv_layers, input_dimensions, input_channels, dropout)

        self.conv_layers = nn.Sequential(collections.OrderedDict(conv_list))
        self.input_dimensions = (*input_dimensions, input_channels)
        self.__input__size = functools.reduce(lambda x, y: x*y, self.input_dimensions, 1)
        self.output_dimension = out_dims

        # Randomize initial parameters
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.kaiming_normal_(p)

    def forward(self, input):

        H,W, input_channels = self.input_dimensions

        # Magic line of code needed to cast image vector into correct dimensions?
        input = input.view(-1, input_channels, H, W)
        outputs = self.conv_layers(input)
        
        assert not torch.isnan(output).any() # type: ignore
        return outputs