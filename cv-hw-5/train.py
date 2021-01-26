import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.autograd import Variable

import matplotlib.pyplot as plt
import numpy as np
import cv2

from torchvision import datasets, transforms

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Sequential(  # input_size=(1*28*28)
            nn.Conv2d(1, 6, 5, 1, 2),  # padding=2
            nn.ReLU(),  # input_size=(6*28*28)
            nn.MaxPool2d(kernel_size=2, stride=2),  # output_size=(6*14*14)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),  # input_size=(16*10*10)
            nn.MaxPool2d(2, 2)  # output_size=(16*5*5)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(84, 10)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
        # return F.log_softmax(x, dim=1)

BATCH_SIZE = 512
EPOCHS = 25 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net().to(DEVICE)
optimizer = optim.SGD(model.parameters(), lr=0.01)

train_dataset = torchvision.datasets.MNIST(root='data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root='data', train=False, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])),
    batch_size = BATCH_SIZE, shuffle = True)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False, transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])),
    batch_size = BATCH_SIZE, shuffle = True)

def train(model, device, train_loader, optimizer, epoch):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            data, target = Variable(data), Variable(target)
            
            '''
            image_array,_=train_dataset[batch_idx]
            image_array=image_array.reshape(28,28)
            plt.imshow(image_array)
            plt.show()
            '''

            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            # loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if(batch_idx+1)%30 ==0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))

def test(epoch_num, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)

            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

for epoch in range(1, EPOCHS + 1):
    train(model, DEVICE, train_loader, optimizer, epoch)
    test(epoch, model, DEVICE, test_loader)

torch.save(model, 'model.pth')
