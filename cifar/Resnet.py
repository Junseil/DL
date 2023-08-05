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
        super.__init__()
        
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