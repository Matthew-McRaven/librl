import more_itertools
import torch

import librl.task
# The current module will be imported before librl.task is finished, so we
# will need to refer to our parent class by relative path.
from .base import Task as _Task

# Given a classifier and a dataset, the goal is to train the classifier to predict dataset labels
# with high portability to the testing data.
# train_data_iter, validation_data_iter should be an iterable of DataLoaders rather than bare DataLoaders.
class ClassificationTask(_Task):
    def __init__(self, classifier=None, criterion=None, train_data_iter=None, validation_data_iter=None, train=True,**kwargs):
        super(ClassificationTask, self).__init__(librl.task.ProblemTypes.Classification, **kwargs)
        assert classifier is not None and criterion is not None
        assert (train_data_iter is not None) or (validation_data_iter is not None)
        # Train / validation data should be an iterable of data loaders rather than a single data loader.
        assert not isinstance(train_data_iter, torch.utils.data.DataLoader) # type: ignore
        assert not isinstance(validation_data_iter, torch.utils.data.DataLoader) # type: ignore

        self.classifier = classifier
        self.criterion = criterion
        self.train = train

        self.train_batch = train_one_batch
        self.train_data_iter = train_data_iter
        self.validate_batch = test_one_batch
        self.validation_data_iter = validation_data_iter

# Default implementation that trains a classifier on a single batch of data
def train_one_batch(classifier, criterion, data, target):
    output = classifier(data)
    loss = criterion(output, target)
    loss.backward()
    classifier.optimizer.step()
    classifier.optimizer.zero_grad()
    return loss.item(), output.argmax(dim=1)

# Default implementation that tests a classifier on a single batch of data.
def test_one_batch(classifier, criterion, data, target):
    output = classifier(data)
    loss = criterion(output, target)
    return loss.item(), output.argmax(dim=1)