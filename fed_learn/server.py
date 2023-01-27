import model_lib
from typing import Callable
import numpy as np
import fed_learn
from fed_learn.weight_summarizer import WeightSummarizer
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
from opacus.validators import ModuleValidator


class Server:
    def __init__(self, model_fn: Callable,
                 weight_summarizer: WeightSummarizer,
                 global_config=None, fed_config=None, dp_config=None):

        if fed_config is None:
            fed_config = {}
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.x_test = None
        self.y_test = None
        self.data_loader = None

        self.nb_clients = fed_config['nb_clients']
        self.client_fraction = fed_config['fraction']
        self.weight_summarizer = weight_summarizer

        # Training config used by the clients
        self.training_config = {
            'batch_size': 32,
            'global_epochs': 5,
            'local_epochs': 1,
            'lr': 0.001,
            'verbose': 1,
            'shuffle': True
        }

        # Initialize the global model's weights
        self.model_fn = model_fn
        temp_model = self.model_fn()

        """
            GET MODEL INFORMATION
        """
        weights_shape, total_params = model_lib.get_model_infor(temp_model)

        print('-' * 100)
        print("[INFO] MODEL INFORMATION ...")
        print("\t Model Weight Shape: ", weights_shape)
        print("\t Total Params of model: ", total_params)
        print()

        self.dp_config = dp_config
        if global_config['dp_mode'] and not ModuleValidator.is_valid(temp_model):
            temp_model = ModuleValidator.fix(temp_model)

        self.global_test_metrics = {
            'loss': [], 'accuracy': []
        }

        self.global_model_weights = model_lib.get_model_weights(temp_model)
        model_lib.get_rid_of_models(temp_model)

        # Initialize the client with differential privacy or not
        if not global_config['dp_mode']:
            self.ClientClass = fed_learn.Client
        else:
            self.ClientClass = fed_learn.PriClient

        # Initialize the losses
        self.global_train_losses = []
        self.epoch_losses = []

        # Initialize clients and clients' weights
        self.clients = []
        self.client_model_weights = []

    def setup(self):
        """Setup all configuration for federated learning"""
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

    def create_model_with_updated_weights(self):
        temp_model = self.model_fn()
        model_lib.set_model_weights(temp_model, self.global_model_weights,
                                    used_device=self.device)
        return temp_model

    def send_model(self, client):
        client.receive_and_init_model(self.model_fn, self.global_model_weights)

    def init_for_new_epoch(self):
        # Reset the collected weights
        self.client_model_weights.clear()
        # Reset epoch losses
        self.epoch_losses.clear()

    def receive_results(self, client):
        client_weights = model_lib.get_model_weights(client.model)

        self.client_model_weights.append(client_weights)
        client.reset_model()

    def create_clients(self):
        for i in range(self.nb_clients):
            client = self.ClientClass(i)
            self.clients.append(client)

    def summarize_weights(self):
        new_weights = self.weight_summarizer.process(self.client_model_weights)
        self.global_model_weights = new_weights

    def update_training_config(self, config: dict):
        self.training_config.update(config)

    def test_global_model(self):
        temp_model = self.create_model_with_updated_weights()

        loss_fn = nn.CrossEntropyLoss()
        total_test_loss = 0
        test_correct = 0

        with torch.no_grad():
            temp_model.eval()

            for (x_batch, y_batch) in self.data_loader:
                (x_batch, y_batch) = (x_batch.float().to(self.device),
                                      y_batch.long().to(self.device))
    
                pred = temp_model(x_batch)
                total_test_loss = total_test_loss + loss_fn(pred, y_batch)
                test_correct = test_correct + (pred.argmax(1) == y_batch).type(
                    torch.float
                ).sum().item()

        avg_test_loss = total_test_loss / len(self.data_loader)
        test_correct = test_correct / len(self.x_test)

        results_dict = {
            'loss': avg_test_loss.cpu().detach().item(),
            'accuracy': test_correct
        }

        for metric_name, value in results_dict.items():
            self.global_test_metrics[metric_name].append(value)

        model_lib.get_rid_of_models(temp_model)
        return results_dict

    def select_clients(self):
        nb_clients_to_use = max(int(self.nb_clients * self.client_fraction), 1)
        client_indices = np.arange(self.nb_clients)
        np.random.shuffle(client_indices)
        selected_client_indices = client_indices[:nb_clients_to_use]
        return np.asarray(self.clients)[selected_client_indices]

    def save_model_weights(self, path):
        temp_model = self.create_model_with_updated_weights()
        torch.save(temp_model, str(path))
        model_lib.get_rid_of_models(temp_model)

    def load_model_weights(self, path, by_name: bool = False):
        temp_model = self.create_model_with_updated_weights()
        temp_model.load_weights(str(path), by_name=by_name)
        self.global_model_weights = temp_model.get_weights()
        model_lib.get_rid_of_models(temp_model)

    def receive_data(self, x, y):
        self.x_test = x
        self.y_test = y


