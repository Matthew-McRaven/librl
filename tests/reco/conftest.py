import pytest
def mnist_dataset():
    import torchvision.datasets, torchvision.transforms
    from librl.utils import load_split_data
    class helper:
        def __init__(self):
            transformation = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),]) 
            # Construct dataloaders from datasets
            train_dset = torchvision.datasets.MNIST("__pycache__/MNIST", transform=transformation, download=True)
            validation_dset = torchvision.datasets.MNIST("__pycache__/MNIST", transform=transformation, train=False)
            self.t_loaders, self.v_loaders = load_split_data(train_dset, 100, 3), load_split_data(validation_dset, 1000, 1)
    return helper()

def cifar10_dataset():
    import torchvision.datasets, torchvision.transforms
    from librl.utils import load_split_data
    class helper:
        def __init__(self):
            mean, stdev = .5, .25
            transformation = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((mean, mean, mean), (stdev, stdev, stdev))]) 

            # Construct dataloaders from datasets
            train_dset = torchvision.datasets.CIFAR10("__pycache__/CIFAR10", transform=transformation, download=True)
            validation_dset = torchvision.datasets.CIFAR10("__pycache__/CIFAR10", transform=transformation, train=False)
            self.t_loaders, self.v_loaders = load_split_data(train_dset, 100, 3), load_split_data(validation_dset, 1000, 1)
    return helper()

@pytest.fixture(params=['MNIST', 'CIFAR10'])
def image_dataset(request):
    # Return loaders, image dims, # labels.
    if request.param == "CIFAR10":
        return cifar10_dataset(), (3, 32, 32), 10
    elif request.param == "MNIST":
        return mnist_dataset(), (1, 28, 28), 10