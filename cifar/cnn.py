import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_epochs = 100
batch_size = 128
weight_decay_lambda = 1e-4
learning_rage = 0.01
momentum = 0.9

transforms_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4882, 0.4465), (0.2023, 0.1994, 0.2010))
])

transforms_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4882, 0.4465), (0.2023, 0.1994, 0.2010))
])

train_dev_dataset = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    transform=transforms_train,
    download=True)

test_dataset = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,
    transform=transforms_test,
    download=True)

train_dataset, dev_dataset = torch.utils.data.random_split(train_dev_dataset, [45000, 5000])


class Lenet(nn.Module):
    def __init__(self):
        super(Lenet, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.fc_layer = nn.Sequential(
            nn.Linear(16*5*5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )
    
    def forward(self, x):
        out = self.layer(x)
        out = out.view(out.size(0), -1)
        out = self.fc_layer(out)
        return out