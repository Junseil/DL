import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_size = 784
hidden_size = 600
num_classes = 10
num_epochs = 10
batch_size = 100
drop_prob = 0.2
weight_decay_lambda = 0.01
learning_rate = 0.001

train_dev_dataset = torchvision.datasets.MNIST(root = './data', train=True, transform=transforms.ToTensor(), download=True)
train_dataset, dev_dataset = torch.utils.data.random_split(train_dev_dataset, [50000, 10000])
test_dataset = torchvision.datasets.MNIST(root = './data', train = False, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size=batch_size, shuffle = True)
dev_loader = torch.utils.data.DataLoader(dataset = dev_dataset, batch_size=batch_size, shuffle = False)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size=batch_size, shuffle = False)


class NeuralNet(nn.Module):
    
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(drop_prob)
        
        torch.nn.init.kaiming_normal_(self.fc1.weight)
        torch.nn.init.zeros_(self.fc1.bias)
        torch.nn.init.kaiming_normal_(self.fc2.weight)
        torch.nn.init.zeros_(self.fc2.bias)
        torch.nn.init.kaiming_normal_(self.fc3.weight)
        torch.nn.init.zeros_(self.fc3.bias)
        
    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = self.dropout(out)
        out = F.relu(self.fc2(out))
        out = self.dropout(out)
        out = self.fc3(out)
        return out

class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
            )
        self.fc_layer = nn.Sequential(
            nn.Linear(64*5*5, num_classes)
        )
    def forward(self, x):
        out = self.layer(x)
        out = out.view(batch_size, -1)
        out = self.fc_layer(out)
        return out

model = CNN(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay=weight_decay_lambda)

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
    return correct / total

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
            print(f'Epochs {epoch+1} / {num_epochs}, Steps {i+1} / {total_step}, Loss {loss.item():.4f}')
            
            with torch.no_grad():
                model.eval()
                acc = evaluation(dev_loader)
                if max < acc:
                    max = acc
                    print(f'max dev accurancy : {max}')
                    torch.save(model.state_dict(), 'model.ckpt')
        i += 1
        
with torch.no_grad():
    last_acc = evaluation(test_loader)
    print(f'Last accurancy of the network on the 10000 test images : {last_acc * 100:.4f} %')
    model.load_state_dict(torch.load('model.ckpt'))
    best_acc = evaluation(test_loader)
    print(f'Last accurancy of the network on the 10000 test images : {best_acc * 100:.4f} %')
