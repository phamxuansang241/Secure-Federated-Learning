import torch
from tek4fed import dp_lib
import model_lib
from typing import Callable
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.optim import Adam
from torch import nn
import gc
import time
from opacus import PrivacyEngine


class PriClient:
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

        self.epsilon = None
        self.delta = None
        self.clipping_norm = None
        self.sigma = None
        self.privacy_engine = None
        self.used_eps = 0

        self.max_allow_iter = None
        self.current_iter = 0

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
        self.epsilon = float(client_config['dp_config']['epsilon'])
        self.delta = float(client_config['dp_config']['delta'])
        self.clipping_norm = float(client_config['dp_config']['clipping_norm'])

        # if 'is_fixed_client_iter' is true, the iteration number of participants is indicated by using 'client_iter'
        # and 'sigma' is calculated by the private budget and 'client_iter'
        if client_config['dp_config']['is_fixed_client_iter']:
            self.max_allow_iter = client_config['dp_config']['client_iter']
            sample_rate = 1 / len(self.data_loader)
            self.sigma = dp_lib.get_min_sigma(
                sample_rate=sample_rate, num_steps=self.max_allow_iter, delta=self.delta, require_eps=self.epsilon
            )
            print(f'Under {self.max_allow_iter} iterations and the sample_rate = {sample_rate}, '
                  f'the sigma of client {self.index} is {self.sigma}')
        # if 'is_fixed_client_iter' is false, use 'sigma' to indicate the amount of noise added by the participant,
        # and 'client_iter' is calculated based on the privacy budget and 'sigma'
        else:
            self.sigma = float(client_config['dp_config']['sigma'])
            sample_rate = 1 / len(self.data_loader)
            self.max_allow_iter = dp_lib.get_client_iter(
                sample_rate=sample_rate, max_eps=self.sigma, delta=self.delta, sigma=self.sigma
            )
            print(f'Under sigma = {self.sigma} and sample_rate = {sample_rate}, '
                  f'the maximum iterations of client {self.index} is {self.max_allow_iter}')

        self.optimizer = Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss()
        self.privacy_engine = PrivacyEngine(secure_mode=False)
        self.model, self.optimizer, self.data_loader = self.privacy_engine.make_private(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=self.data_loader,
            noise_multiplier=self.sigma,
            max_grad_norm=self.clipping_norm
        )

    def edge_train(self):
        """
        Update local model using local dataset
        """
        if self.model is None:
            raise ValueError('Model is not created for client: {0}'.format(self.index))
        self.model.to(self.device)
        self.model.train()
        # print(self.model)
        losses = []

        if self.current_iter + self.local_epochs*len(self.data_loader) <= self.max_allow_iter:
            local_allow_iter = self.local_epochs*len(self.data_loader)
        else:
            local_allow_iter = self.max_allow_iter - self.current_iter
        self.current_iter = self.current_iter + self.local_epochs*len(self.data_loader)

        _batch_idx = 0

        # measure how long training is going to take
        start_time = time.time()

        while True:
            for x_batch, y_batch in self.data_loader:
                if _batch_idx == local_allow_iter:
                    break

                # print(_batch_idx)
                _batch_idx = _batch_idx + 1
                # send the input to the device
                (x_batch, y_batch) = (x_batch.long().to(self.device),
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

            if _batch_idx == local_allow_iter:
                break

        self.used_eps = self.privacy_engine.accountant.get_epsilon(
            delta=self.delta
        )

        # finish measuring how long training took
        end_time = time.time()
        print("\t\tLoss value: {:.6f}".format(sum(losses) / len(losses)))
        print("\t\tEpsilon is used: {:.6f}".format(self.used_eps))
        print("\t\t Total time taken to train: {:.2f}s".format(end_time - start_time))

        gc.collect()

        return losses

    def init_model(self, model_fn: Callable, model_weights):
        temp_model = model_fn().to(self.device)
        model_lib.set_model_weights(temp_model, model_weights, used_device=self.device)
        self.model = temp_model

    def receive_data(self, x, y):
        self.x_train = x
        self.y_train = y

    def receive_and_init_model(self, model_fn: Callable, model_weights):
        self.init_model(model_fn, model_weights)

    def reset_model(self):
        model_lib.get_rid_of_models(self.model)
