from tek4fed.decorator import timer
from tek4fed import model_lib
from typing import Callable
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam
from torch import nn
import gc


class Client:
    def __init__(self, index: int):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.index = index
        self.x_train = None
        self.y_train = None
        self.data_loader = None

        self.local_epochs = None
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.lr = None

    def setup(self, **client_config):
        """Set up the configuration of each client"""
        batch_size = client_config['training_config']['batch_size']
        x_train = torch.from_numpy(self.x_train)
        y_train = torch.from_numpy(self.y_train)

        train_dataset = TensorDataset(x_train, y_train)
        self.data_loader = DataLoader(train_dataset,
                                      shuffle=True,
                                      batch_size=batch_size)

        self.local_epochs = client_config['training_config']['local_epochs']
        self.lr = 0.001
        self.optimizer = Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss()

    def receive_data(self, x, y):
        self.x_train = x
        self.y_train = y

    def receive_and_init_model(self, model_fn: Callable, model_weights):
        temp_model = model_fn().to(self.device)
        model_lib.set_model_weights(temp_model, model_weights, used_device=self.device)
        self.model = temp_model

    @timer
    def edge_train(self):
        if self.model is None:
            raise ValueError('Model is not created for client: {0}'.format(self.index))
        self.model.to(self.device)
        # set the model_lib in training mode
        self.model.train()

        losses = []

        for e in range(0, self.local_epochs):
            # loop over the training set
            for (x_batch, y_batch) in self.data_loader:
                # send the input to the device
                (x_batch, y_batch) = (x_batch.to(self.device),
                                      y_batch.long().to(self.device))
                
                # perform a forward pass and calculate the training loss
                pred = self.model(x_batch)
                loss = self.criterion(pred, y_batch)

                # zero out the gradients, perform the backpropagation step,
                # and update the weights
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                losses.append(loss.item())
                torch.cuda.empty_cache()

        print("\t\t Loss value: {:.6f}".format(sum(losses) / len(losses)))

        gc.collect()

        return losses

