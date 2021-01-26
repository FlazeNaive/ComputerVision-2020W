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
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 6, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), 
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),
            nn.ReLU(), 
            nn.MaxPool2d(2, 2) 
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load('model.pth') 

model = model.to(device)
model.eval()

PATH = "3.bmp"

img = cv2.imread(PATH)

trans = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = trans(img)
img = img.to(device)
img = img.unsqueeze(0)

output = model(img)

prob = F.softmax(output, dim=1)
prob = Variable(prob)
prob = prob.cpu().numpy()
prob = prob[0]
print(prob)
pred_out = np.argmax(prob)

plt.title('pred is {}'.format(pred_out))
img = cv2.imread(PATH)
plt.imshow(img)

plt.show()
