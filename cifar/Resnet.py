import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample):
        super().__init__()
        self.downsample = downsample
        if self.downsample:
            stride = 2
            self.down_skip_net = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                          kernel_size=1, stride=2),
                nn.BatchNorm2d(num_features=out_channels)
            )
        else:
            stride = 1
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        
    def forward(self, x):
        if self.downsample:
            skip = self.down_skip_net(x)
        else:
            skip = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += skip
        out = self.relu(out)
        
        return out


class Resnet18(nn.Module):
    def __init__(self, num_classes=10):
        super.__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(in_channels=64, out_channels=64, num_blocks=2)
        self.layer2 = self._make_layer(in_channels=64, out_channels=128, num_blocks=2,
                                       downsample=True)
        self.layer3 = self._make_layer(in_channels=128, out_channels=256, num_blocks=2,
                                       downsample=True)        
        self.layer4 = self._make_layer(in_channels=256, out_channels=512, num_blocks=2,
                                       downsample=True)
        self.adapool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_features=512, out_features=num_classes)
        
    def _make_layer(self, in_channels, out_channels, num_blocks, downsample=False):
        layer = []
        layer.append(BasicBlock(in_channels, out_channels, downsample))
        for _ in range(1, num_blocks):
            layer.append(BasicBlock(in_channels, out_channels, downsample=False))
        return nn.Sequential(*layer)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = self.adapool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        
        return out
    
class BottleNeckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample):
        super().__init__()
        
        self.downsample = downsample
        if self.downsample:
            stride = 2
            self.down_skip_net = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                          kernel_size=1, stride=2),
                nn.BatchNorm2d(num_features=out_channels)
            )
        else:
            stride = 1
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels//4,
                               kernel_size=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(num_features=out_channels//4)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channels//4, out_channels=out_channels//4,
                               kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels//4)
        self.conv3 = nn.Conv2d(in_channels=out_channels//4, out_channels=out_channels,
                               kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm2d(num_features=out_channels)
        
    def forward(self, x):
        if self.downsample:
            skip = self.down_skip_net(x)
        else:
            skip = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out += skip
        out = self.relu(out)
        
        return out
    
class Resnet101(nn.Module):
    def __init__(self, num_classes = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=256, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=256)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(in_channels=256, out_channels=256, num_blocks=3)
        self.layer2 = self._make_layer(in_channels=256, out_channels=512, num_blocks=4,
                                  downsample=True)
        self.layer3 = self._make_layer(in_channels=512, out_channels=1024, num_blocks=23,
                                  downsample=True)
        self.layer4 = self._make_layer(in_channels=1024, out_channels=2048, num_blocks=3,
                                  downsample=True)
        
        self.adapool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_features=2048, out_features=num_classes)
        
    def _make_layer(self, in_channels, out_channels, num_blocks, downsample=False):
        layer = []
        layer.append(BottleNeckBlock(in_channels, out_channels, downsample))
        for _ in range(1, num_blocks):
            layer.append(BottleNeckBlock(out_channels, out_channels, downsample=False))
        return nn.Sequential(*layer)

    def forward(self, x):
        out=self.conv1(x)
        out=self.bn1(out)
        out=self.relu(out)
        out=self.maxpool(out)
        out=self.layer1(out)
        out=self.layer2(out)
        out=self.layer3(out)
        out=self.layer4(out)
        out = self.adapool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
    
    
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_epochs = 100
batch_size = 128
weight_decay_lambda = 1e-4
learning_rate = 0.1
momentum = 0.9


transforms_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010))
])

transforms_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010))
])
train_dev_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                 transform=transforms_train,
                                                 download=True)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            transform=transforms_test)
train_dataset, dev_dataset = torch.utils.data.random_split(train_dev_dataset,
                                                          [45000, 5000])
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
dev_loader = torch.utils.data.DataLoader(dataset=dev_dataset,
                                           batch_size=batch_size,
                                           shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                           batch_size=batch_size,
                                           shuffle=False)

model = Resnet101().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),
            lr=learning_rate, momentum=momentum,
            weight_decay=weight_decay_lambda)
scheduler = MultiStepLR(optimizer, milestones=[50,75], gamma=0.1)

def evaluation(data_loader):
    correct = 0
    total = 0
    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    return correct/total

total_step = len(train_loader)
max = 0.0

for epoch in range(num_epochs):
    i = 0
    for images, labels in train_loader:
        model.train()
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
            .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
            
            with torch.no_grad():
                model.eval()
                acc = evaluation(dev_loader)
                if max < acc:
                    max = acc
                    print("max dev accuracy: ", max)
                    torch.save(model.state_dict(), "model.ckpt")
        i = i + 1
    scheduler.step()

with torch.no_grad():
    last_acc = evaluation(test_loader)
    print("Last Accuracy of the network on the 10000 test images: {}%".format(last_acc*100))

    model.load_state_dict(torch.load('model.ckpt'))
    best_acc = evaluation(test_loader)
    print("Best Accuracy of the network on the 10000 test images: {}%".format(best_acc*100))