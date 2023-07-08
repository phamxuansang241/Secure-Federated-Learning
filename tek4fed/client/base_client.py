from tek4fed import model_lib, data_lib
from typing import Callable
import torch
from torch.utils.data import TensorDataset
from torch.optim import Adam
from torch import nn


class BaseClient:
    def __init__(self, index: int):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.index = index

        self.x_train, self.y_train, self.dataset = None, None, None
        self.training_config = None
        self.model, self.optimizer, self.criterion, self.lr = None, None, None, None

    def setup(self, **client_config):
        """
        Set up the configuration of each client.

        Args:
            client_config (dict): Configuration details for the client.
        """
        self._configure_training(client_config)
        self.optimizer = Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss()

    def _configure_training(self, client_config):
        """
        Configure the training settings for the client.

        Args:
            client_config (dict): Configuration details for the client.
        """
        self.training_config = client_config['training_config']
        dataset_name = self.training_config['dataset_name']
        self.lr = 0.001

        if dataset_name == 'covid':
            self.dataset = data_lib.ChestXRayDataset(self.x_train, self.y_train, 'train')
        else:
            x_train = torch.from_numpy(self.x_train)
            y_train = torch.from_numpy(self.y_train)
            self.dataset = TensorDataset(x_train, y_train)

    def receive_data(self, x, y):
        self.x_train = x
        self.y_train = y

    def receive_and_init_model(self, model_fn: Callable, model_weights):
        temp_model = model_fn().to(self.device)
        model_lib.set_model_weights(temp_model, model_weights, used_device=self.device)
        self.model = temp_model
