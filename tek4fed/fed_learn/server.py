from tek4fed.model_lib import get_model_weights, get_model_infor, get_rid_of_models,\
     set_model_weights
from tek4fed.model_lib import get_model_weights, get_model_infor, get_rid_of_models, set_model_weights, \
    get_dssgd_update
from tek4fed.fed_learn.weight_summarizer import WeightSummarizer
from tek4fed import fed_learn
from tek4fed.compress_params_lib import CompressParams
from tek4fed.encryption_lib import EccEncryption, ElGamalEncryption
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
from opacus.validators import ModuleValidator
from math import *
from typing import Callable
import numpy as np


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
            'lr': 0.001
        }

        # Initialize the global model's weights
        self.model_fn = model_fn
        temp_model = self.model_fn()

        """
            GET MODEL INFORMATION
        """
        self.model_infor = {'weights_shape': (get_model_infor(temp_model))[0],
                            'total_params': (get_model_infor(temp_model))[1]}

        print('-' * 100)
        print("[INFO] MODEL INFORMATION ...")
        print("\t Model Weight Shape: ", self.model_infor['weights_shape'])
        print("\t Total Params of model: ", self.model_infor['total_params'])
        print()

        self.dp_config = dp_config
        if global_config['dp_mode'] and not ModuleValidator.is_valid(temp_model):
            temp_model = ModuleValidator.fix(temp_model)

        self.global_test_metrics = {
            'loss': [], 'accuracy': []
        }

        self.global_model_weights = get_model_weights(temp_model)
        get_rid_of_models(temp_model)

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
        self.sum_client_weight = []

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
        set_model_weights(temp_model, self.global_model_weights, used_device=self.device)
        return temp_model

    def send_model(self, client):
        client.receive_and_init_model(self.model_fn, self.global_model_weights)

    def init_for_new_epoch(self):
        # Reset the collected weights
        self.client_model_weights.clear()
        # Reset epoch losses
        self.epoch_losses.clear()

    def select_clients(self):
        nb_clients_to_use = max(int(self.nb_clients * self.client_fraction), 1)
        client_indices = np.arange(self.nb_clients)
        np.random.shuffle(client_indices)
        selected_client_indices = client_indices[:nb_clients_to_use]
        print('Selected clients for epoch: {0}'.format('| '.join(map(str, selected_client_indices))))

        return np.asarray(self.clients)[selected_client_indices]

    def receive_results(self, client):
        client_weights = get_model_weights(client.model)

        self.client_model_weights.append(client_weights)
        client.reset_model()

    def create_clients(self):
        for i in range(self.nb_clients):
            client = self.ClientClass(i)
            self.clients.append(client)

    def summarize_weights(self, encrypt_mode=False):

        if not encrypt_mode:
            new_weights = self.weight_summarizer.process(self.client_model_weights)
        else:
            new_weights = self.weight_summarizer.process_encryption(self.nb_clients, self.sum_client_weight)
        self.global_model_weights = new_weights

    def update_training_config(self, config: dict):
        self.training_config.update(config)

    def train_fed_compress(self):
        compress_params = CompressParams(self.training_config['compress_digit'])
        print('\t Compress number:', compress_params.compress_number)

        for epoch in range(self.training_config['global_epochs']):
            print('[TRAINING] Global Epoch {0} starts ...'.format(epoch))

            self.init_for_new_epoch()
            selected_clients = self.select_clients()
            
            for client in selected_clients:
                print('\t Client {} starts training'.format(client.index))

                if self.training_config['dp_mode']:
                    if client.current_iter > client.max_allow_iter:
                        break

                set_model_weights(client.model, self.global_model_weights, client.device)
                client_losses = client.edge_train()

                print('\t\t Encoding parameters ...')
                compress_params.encode_model(client=client)

                self.epoch_losses.append(client_losses[-1])

            decoded_weights = compress_params.decode_model(selected_clients)
            self.client_model_weights = decoded_weights.copy()
            self.summarize_weights()

            epoch_mean_loss = np.mean(self.epoch_losses)
            self.global_train_losses.append(epoch_mean_loss)
            print('\tLoss (client mean): {0}'.format(self.global_train_losses[-1]))

            # testing current model_lib
            self.test_global_model()

    def train_fed_ecc_encryption(self, short_ver=False):
        mtx_size = ceil(sqrt(self.model_infor['total_params']))

        encrypt = EccEncryption(self.nb_clients, mtx_size)
        encrypt.client_setup_private_params()

        if not short_ver:
            encrypt.client_setup_keys()
            encrypt.calculate_server_public_key()

        for epoch in range(self.training_config['global_epochs']):
            print('[TRAINING] Global Epoch {0} starts ...'.format(epoch))

            self.init_for_new_epoch()
            selected_clients = self.select_clients()

            # perform phase on of the encryption
            encrypt.perform_phase_one(short_ver)
            for client in selected_clients:
                print('\t Client {} starts training'.format(client.index))
                set_model_weights(client.model, self.global_model_weights, client.device)
                client_losses = client.edge_train()
                self.epoch_losses.append(client_losses[-1])

                print('\t\t [ECC] Encoding phase two')
                encrypt.encoded_message_phase_two(client)

            print('\t [ECC] Server decoding phase two')
            self.sum_client_weight = encrypt.decoded_message_phase_two(selected_clients)
            self.summarize_weights(encrypt_mode=True)

            self.global_train_losses.append(np.mean(self.epoch_losses))
            print('\t Loss (client mean): {0}'.format(self.global_train_losses[-1]))

            # testing current model_lib
            self.test_global_model()

    def train_fed_elgamal_encryption(self, short_ver=False):
        mtx_size = ceil(sqrt(self.model_infor['total_params']))

        encrypt = ElGamalEncryption(self.nb_clients, mtx_size)
        encrypt.generate_client_noise_mtx()
        encrypt.generate_client_key()


        for epoch in range(self.training_config['global_epochs']):
            print('[TRAINING] Global Epoch {0} starts ...'.format(epoch))

            self.init_for_new_epoch()
            selected_clients = self.select_clients()

            encrypt.calculate_server_public_key(selected_clients)

            for client in selected_clients:
                print('\t Client {} starts training'.format(client.index))
                set_model_weights(client.model, self.global_model_weights, client.device)
                client_losses = client.edge_train()
                self.epoch_losses.append(client_losses[-1])
                print('\t\t [ELGAMAL] Encoding ...')
                encrypt.encoded_message(client)

            print('\t [ELGAMAL] Server decoding ...')
            self.sum_client_weight = encrypt.decoded_message(selected_clients)
            self.summarize_weights(encrypt_mode=True)

            epoch_mean_loss = np.mean(self.epoch_losses)
            self.global_train_losses.append(epoch_mean_loss)
            print('\t Loss (client mean): {}'.format(self.global_train_losses[-1]))

            # testing current model_lib
            self.test_global_model()

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

        avg_test_loss = total_test_loss / len(self.data_loader)
        test_correct = test_correct / len(self.x_test)

        results_dict = {
            'loss': avg_test_loss.cpu().detach().item(),
            'accuracy': test_correct
        }

        for metric_name, value in results_dict.items():
            self.global_test_metrics[metric_name].append(value)

        get_rid_of_models(temp_model)
        print("\t----- Evaluating on server's test dataset -----")
        print('{0}: {1}'.format('\tLoss', results_dict['loss']))
        print('{0}: {1}'.format('\tAccuracy', results_dict['accuracy']))
        print('-' * 100)

        return results_dict

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

    def train_dssgd(self):
        for epoch in range(self.training_config['global_epochs']):
            print('[TRAINING] Global Epoch {0} starts ...'.format(epoch))

            selected_clients = self.select_clients()
            clients_ids = [c.index for c in selected_clients]
            print('Selected clients for epoch: {0}'.format('| '.join(map(str, clients_ids))))

            for client in selected_clients:
                print('\t Client {} starts training'.format(client.index))
                set_model_weights(client.model, self.global_model_weights, client.device)
                client.edge_train()
                self.global_model_weights = get_dssgd_update(client.model, self.global_model_weights, self.model_infor['weights_shape'], theta_upload=0.9)

            self.test_global_model()
                






