import unittest

import torch.utils
import torchvision.datasets, torchvision.transforms

import librl.nn.core
import librl.nn.classifier
import librl.task
import librl.train.train_loop, librl.train.classification
import librl.utils

class TestClassification(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dist = librl.task.TaskDistribution()

    def tearDown(self):
        self.dist.clear_tasks()

    def test_label_mnist(self):
        hypers = {'epochs':2, 'task_count':1}
        transformation = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()]) 
        class_kernel = librl.nn.core.MLPKernel((1, 28, 28), (200, 100))
        class_net = librl.nn.classifier.Classifier(class_kernel, 10)

        # Load the MNIST training / validation datasets
        train_dset = torchvision.datasets.MNIST("__pycache__/MNIST", transform=transformation, download=True)
        validation_dset = torchvision.datasets.MNIST("__pycache__/MNIST", transform=transformation, train=False)
        # Construct dataloaders from datasets
        t_loaders, v_loaders = librl.utils.load_split_data(train_dset, 100, 3), librl.utils.load_split_data(validation_dset, 1000, 1)
        # Construct a labelling task.
        self.dist.add_task(librl.task.Task.Definition(librl.task.ClassificationTask, classifier=class_net, criterion=torch.nn.CrossEntropyLoss(), train_data_iter=t_loaders, validation_data_iter=v_loaders))
        librl.train.train_loop.cls_trainer(hypers, self.dist, librl.train.classification.train_single_label_classifier)
        
    def test_label_cifar10(self):
        hypers = {'epochs':2, 'task_count':1}
        mean, stdev = .5, .25
        transformation = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((mean, mean, mean), (stdev, stdev, stdev))]) 

        class_kernel = librl.nn.core.MLPKernel((3, 32, 32), (200, 100))
        class_net = librl.nn.classifier.Classifier(class_kernel, 10)
        
        # Load the CIFAR10 training / validation datasets
        train_dset = torchvision.datasets.CIFAR10("__pycache__/CIFAR10", transform=transformation, download=True)
        validation_dset = torchvision.datasets.CIFAR10("__pycache__/CIFAR10", transform=transformation, train=False)
        # Construct dataloaders from datasets
        t_loaders, v_loaders = librl.utils.load_split_data(train_dset, 100, 3), librl.utils.load_split_data(validation_dset, 1000, 1)
        # Construct a labelling task.
        self.dist.add_task(librl.task.Task.Definition(librl.task.ClassificationTask, classifier=class_net, criterion=torch.nn.CrossEntropyLoss(), train_data_iter=t_loaders, validation_data_iter=v_loaders))
        librl.train.train_loop.cls_trainer(hypers, self.dist, librl.train.classification.train_single_label_classifier)


if __name__ == "__main__":
    unittest.main()