import pytest
import torch.nn

import librl.task
import librl.train.train_loop, librl.train.classification

from . import *

@pytest.mark.parametrize('nn_kind', ['cnn', 'mlp'])
def test_label_images(image_dataset, nn_kind, hypers):
    dset, dims, labels = image_dataset
    class_net = build_class_net(nn_kind, dims, labels)
    t,v = dset.t_loaders, dset.v_loaders
    
    # Construct a labelling task.
    dist = librl.task.distribution.TaskDistribution()
    dist.add_task(librl.task.Task.Definition(librl.task.ClassificationTask, 
        classifier=class_net, criterion=torch.nn.CrossEntropyLoss(),
        train_data_iter=t, validation_data_iter=v,train_percent=.2, validation_percent=.1)
    )
    librl.train.train_loop.cls_trainer(hypers, dist, librl.train.classification.train_single_label_classifier)
