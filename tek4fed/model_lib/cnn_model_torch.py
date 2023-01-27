from torch.nn import Module
from torch.nn import Embedding
from torch.nn import Conv1d
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import MaxPool1d
from torch.nn import Dropout
from torch.nn import Softmax
from torch import flatten


def compute_output_size(input_length, window, stride):
    return int((input_length-window) / stride) + 1


class CNN(Module):
    def __init__(self, vocab_size, embed_dim, input_length, num_class):
        super(CNN, self).__init__()
        self.embedding = Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim, padding_idx=0)

        self.conv1 = Conv1d(in_channels=embed_dim, out_channels=64, kernel_size=3, padding='valid')
        self.relu1 = ReLU()
        self.maxpool1 = MaxPool1d(kernel_size=3)
        output_size = compute_output_size(input_length, 3, 1) // 4

        self.conv2 = Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding='valid')
        self.relu2 = ReLU()
        output_size = compute_output_size(output_size, 3, 1)
        self.maxpool2 = MaxPool1d(kernel_size=output_size)

        self.fc1 = Linear(in_features=64, out_features=64)
        self.relu3 = ReLU()
        self.dropout = Dropout(p=0.5)

        self.fc2 = Linear(in_features=64, out_features=num_class)
        self.softmax = Softmax(dim=1)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)

        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.softmax(x)

        return x



