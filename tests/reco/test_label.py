import unittest

import pytest
import torch.utils
import torchvision.datasets, torchvision.transforms


import librl.nn.core
import librl.nn.core.cnn
import librl.nn.classifier
import librl.task
from librl.train.classification import label
import librl.train.train_loop, librl.train.classification
import librl.utils


def build_class_net(kind, dims, labels):
    class_kernel = None
    if kind == "mlp": 
        class_kernel = librl.nn.core.MLPKernel(dims, (200, 100))
    elif kind == "cnn":
        conv_list = [
        librl.nn.core.cnn.conv_def(4, 4, 1, 0, 1, False),
        librl.nn.core.cnn.conv_def(4, 4, 1, 0, 1, False),
        librl.nn.core.cnn.pool_def(1, 1, 0, 1, True, 'max'),
        ]
        class_kernel = librl.nn.core.ConvolutionalKernel(conv_list, dims[1:], dims[0])
    else: raise NotImplementedError("I don't understand the network type.")
    return librl.nn.classifier.Classifier(class_kernel, labels)

@pytest.mark.parametrize('nn_kind', ['cnn', 'mlp'])
def test_label_images(image_dataset, nn_kind, hypers):
    dset, dims, labels = image_dataset
    class_net = build_class_net(nn_kind, dims, labels)
    t,v = dset.t_loaders, dset.v_loaders
    
    # Construct a labelling task.
    dist = librl.task.distribution.TaskDistribution()
    dist.add_task(librl.task.Task.Definition(librl.task.ClassificationTask, classifier=class_net, criterion=torch.nn.CrossEntropyLoss(), train_data_iter=t, validation_data_iter=v))
    librl.train.train_loop.cls_trainer(hypers, dist, librl.train.classification.train_single_label_classifier)
