import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
from math import *
from typing import Callable
from tek4fed.server import ServerInterface
from tek4fed.client import NormalClient, PriClient
from tek4fed.fed_learn.weight_summarizer import WeightSummarizer
from tek4fed.model_lib import (
    get_model_weights,
    get_model_infor,
    get_rid_of_models,
    set_model_weights,
)


class BaseServer(ServerInterface):
    """
    The Server class represents a generic server in a federated learning setting.
    It is responsible for managing the global model, clients, and federated learning process.

    Attributes:
        device (torch.device): The device to be used for computations (GPU or CPU).
        x_test (numpy.ndarray): The test input data.
        y_test (numpy.ndarray): The test output data.
        data_loader (torch.utils.data.DataLoader): The DataLoader for test data.
        nb_clients (int): The number of clients participating in federated learning.
        client_fraction (float): The fraction of clients to be selected for each communication round.
        weight_summarizer (WeightSummarizer): An object that handles weight summarization for federated learning.
        training_config (dict): The configuration parameters for the client training process.
        model_fn (Callable): A function that creates and returns the model to be used in federated learning.
        model_infor (dict): A dictionary containing information about the model.
        dp_config (dict): Configuration parameters specific to differential privacy.
        dp_mode (bool): The differential privacy mode to be used during training.
        global_test_metrics (dict): A dictionary storing the global test loss and accuracy.
        global_model_weights (list): The weights of the global model.
        ClientClass (type): The client class to be used (either federated learning client or privacy-preserving client).
        global_train_losses (list): A list of global train losses.
        epoch_losses (list): A list of epoch losses.
        clients (list): A list of clients participating in federated learning.
        client_model_weights (list): A list of client model weights.
        sum_client_weight (list): A list of summed client weights.
        max_acc (float): The maximum accuracy achieved by the global model on the test dataset.
        global_weight_path (str): The path to the file where the global model's weights are stored.
        """
    def __init__(
            self,
            model_fn: Callable,
            weight_summarizer: WeightSummarizer,
            dp_mode,
            fed_config=None,
            dp_config=None
    ):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.x_test = None
        self.y_test = None
        self.data_loader = None

        self.nb_clients = fed_config['nb_clients']
        self.client_fraction = fed_config['fraction']
        self.weight_summarizer = weight_summarizer

        # Training config used by the clients
        self.training_config = {}

        # Initialize the global model's weights
        self.model_fn = model_fn
        temp_model = self.model_fn()

        # Get model information
        self.model_infor = {'weights_shape': (get_model_infor(temp_model))[0],
                            'total_params': (get_model_infor(temp_model))[1]}

        print('-' * 100)
        print("[INFO] MODEL INFORMATION ...")
        print("\t Model Weight Shape: ", self.model_infor['weights_shape'])
        print("\t Total Params of model: ", self.model_infor['total_params'])
        print()

        self.dp_config = dp_config
        self.dp_mode = dp_mode

        self.global_test_metrics = {
            'loss': [], 'accuracy': []
        }

        self.global_model_weights = get_model_weights(temp_model)
        get_rid_of_models(temp_model)

        # Initialize the client with differential privacy or not
        if not self.dp_mode:
            self.ClientClass = NormalClient
        else:
            self.ClientClass = PriClient

        # Initialize the losses
        self.global_train_losses = []
        self.epoch_losses = []

        # Initialize clients and clients' weights
        self.clients = []
        self.client_model_weights = []
        self.sum_client_weight = []

        self.max_acc = 0
        self.global_weight_path = None

    def setup(self):
        # Setup server's dataloader
        batch_size = self.training_config['batch_size']
        x_test = torch.from_numpy(self.x_test)
        y_test = torch.from_numpy(self.y_test)

        test_dataset = TensorDataset(x_test, y_test)
        self.data_loader = DataLoader(test_dataset, shuffle=True, batch_size=batch_size)

        # Set up each client
        client_config = {
            'training_config': self.training_config.copy(),
            'dp_config': self.dp_config.copy()
        }

        for client in self.clients:
            self.send_model(client)
            client.setup(**client_config)

    def create_clients(self):
        for i in range(self.nb_clients):
            client = self.ClientClass(i)
            self.clients.append(client)

    def select_clients(self):
        nb_clients_to_use = max(int(self.nb_clients * self.client_fraction), 1)
        client_indices = np.arange(self.nb_clients)
        np.random.shuffle(client_indices)
        selected_client_indices = client_indices[:nb_clients_to_use]
        print('\t Selected clients for epoch: {0}'.format('| '.join(map(str, selected_client_indices))))

        return np.asarray(self.clients)[selected_client_indices]

    def create_model_with_updated_weights(self):
        temp_model = self.model_fn()
        set_model_weights(temp_model, self.global_model_weights, used_device=self.device)
        return temp_model

    def init_for_new_epoch(self):
        # Reset the collected weights
        self.client_model_weights.clear()
        # Reset epoch losses
        self.epoch_losses.clear()

    def send_model(self, client):
        client.receive_and_init_model(self.model_fn, self.global_model_weights)

    def receive_results(self, client):
        client_weights = get_model_weights(client.model)

        self.client_model_weights.append(client_weights)
        client.reset_model()

    def summarize_weights(self, encrypt_mode=False):
        if not encrypt_mode:
            new_weights = self.weight_summarizer.process(self.client_model_weights)
        else:
            new_weights = self.weight_summarizer.process_encryption(self.nb_clients, self.sum_client_weight)
        self.global_model_weights = new_weights

    def test_global_model(self):
        temp_model = self.create_model_with_updated_weights()

        loss_fn = nn.CrossEntropyLoss()
        total_test_loss = 0
        test_correct = 0

        with torch.no_grad():
            temp_model.eval()

            for (x_batch, y_batch) in self.data_loader:
                (x_batch, y_batch) = (x_batch.to(self.device),
                                      y_batch.long().to(self.device))

                pred = temp_model(x_batch)
                total_test_loss = total_test_loss + loss_fn(pred, y_batch)
                test_correct = test_correct + (pred.argmax(1) == y_batch).type(
                    torch.float
                ).sum().item()

        avg_test_loss = (total_test_loss / len(self.data_loader)).cpu().detach().item()
        test_correct = test_correct / len(self.x_test)

        self.global_test_metrics['loss'].append(avg_test_loss)
        self.global_test_metrics['accuracy'].append(test_correct)

        if isclose(self.max_acc, test_correct) or (test_correct > self.max_acc):
            self.max_acc = test_correct
            self.save_model_weights(self.global_weight_path)

        get_rid_of_models(temp_model)
        print("\t----- Evaluating on server's test dataset -----")
        print('{0}: {1}'.format('\t\tLoss', avg_test_loss))
        print('{0}: {1}'.format('\t\tAccuracy', test_correct))
        print('-' * 100)

    def save_model_weights(self, path):
        temp_model = self.create_model_with_updated_weights()
        torch.save(temp_model, str(path))
        get_rid_of_models(temp_model)

    def load_model_weights(self, path, by_name: bool = False):
        temp_model = self.create_model_with_updated_weights()
        temp_model.load_weights(str(path), by_name=by_name)
        self.global_model_weights = temp_model.get_weights()
        get_rid_of_models(temp_model)

    def receive_data(self, x, y):
        self.x_test = x
        self.y_test = y
