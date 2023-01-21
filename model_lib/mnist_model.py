import torch.nn as nn
from torch.nn import ReLu, Linear, Softmax, MaxPool2d


"""
Formula to compute shape as going through conv layer = [(X-F+2P) / S] + 1
X = Width / Height
F = Kernel size
P = Padding
S = Strideds (default=1)
""" 

class Mnist_Net(nn.Module):
    def __init__(self, num_class) -> None:
        super(Mnist_Net, self).__init__()

        # first set of CONV => RELU => POOL layers
        # [Batch size, 1, 28, 28] --> [Batch size, 32, 24, 24]
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=32, 
            kernel_size=5, stride=1, padding="same"
        )

        self.relu1 = ReLu()
        # [Batch size, 32, 24, 24] --> [Batch size, 32, 12, 12]
        self.maxpool1 = MaxPool2d(kernel_size=2)


        # second set of CONV => RELU => POOL layers
        # [Batch size, 32, 12, 12] --> [Batch size, 64, 8, 8]
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, 
            kernel_size=5, stride=1, padding="same"
        )
        self.relu2 = ReLu()      
        # [Batch size, 64, 8, 8] --> [Batch size, 64, 4, 4]
        self.maxpool2 = MaxPool2d(kernel_size=2)

        # Input shape --> 100 outputs
        self.fc1 = Linear(64*4*4, 100)
        self.relu3 = ReLu()
        # 100 outputs --> Num-Class outputs
        self.fc2 = Linear(num_class)
        self.softmax = Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(1)

        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu3(x)

        x = self.fc2(x)
        x = self.softmax(x)

        return x






