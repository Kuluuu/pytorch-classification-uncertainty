import torch
from torchvision.datasets.mnist import MNIST
from torchvision.datasets.cifar import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# data_train = MNIST("./data/mnist",
#                    download=True,
#                    train=True,
#                    transform=transforms.Compose([transforms.ToTensor()]))

# data_val = MNIST("./data/mnist",
#                  train=False,
#                  download=True,
#                  transform=transforms.Compose([transforms.ToTensor()]))

# dataloader_train = DataLoader(
#     data_train, batch_size=1000, shuffle=True, num_workers=8)
# dataloader_val = DataLoader(data_val, batch_size=1000, num_workers=8)

# dataloaders = {
#     "train": dataloader_train,
#     "val": dataloader_val,
# }

# digit_one, _ = data_val[5]

data_train = CIFAR10("./data/cifar10",
                   download=False,
                   train=True,
                   transform=transforms.Compose([transforms.ToTensor()]))

data_val = CIFAR10("./data/cifar10",
                 train=False,
                 download=False,
                 transform=transforms.Compose([transforms.ToTensor()]))

dataloader_train = DataLoader(
    data_train, batch_size=1000, shuffle=True, num_workers=8)
dataloader_val = DataLoader(data_val, batch_size=1000, num_workers=8)

dataloaders = {
    "train": dataloader_train,
    "val": dataloader_val,
}

digit_one, _ = data_val[5]
