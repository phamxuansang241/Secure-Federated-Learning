from tek4fed.decorator import timer
from tek4fed import dp_lib, model_lib
from tek4fed.client import BaseClient
from opacus import PrivacyEngine
from typing import Callable
import torch
import gc


class PriClient(BaseClient):
    """
    A class representing a client for federated learning with differential privacy.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epsilon, self.delta, self.clipping_norm, self.sigma, self.privacy_engine = None, None, None, None, None

        self.used_eps = 0
        self.max_allow_iter = None
        self.current_iter = 0

    def setup(self, **client_config):
        super().setup(**client_config)
        self._configure_privacy(client_config)
        self.privacy_engine = PrivacyEngine(secure_mode=False)

        self.model, self.optimizer, self.data_loader = self.privacy_engine.make_private(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=self.data_loader,
            noise_multiplier=self.sigma,
            max_grad_norm=self.clipping_norm
        )

    def _configure_privacy(self, client_config):
        """
        Configure the privacy settings for the client.

        Args:
            client_config (dict): Configuration details for the client.
        """
        dp_config = client_config['dp_config']
        self.epsilon = float(dp_config['epsilon'])
        self.delta = float(dp_config['delta'])
        self.clipping_norm = float(dp_config['clipping_norm'])

        # if 'is_fixed_client_iter' is true, the iteration number of participants is indicated by using 'client_iter'
        # and 'sigma' is calculated by the private budget and 'client_iter'
        if dp_config['is_fixed_client_iter']:
            self.max_allow_iter = dp_config['client_iter']
            sample_rate = 1 / len(self.data_loader)
            self.sigma = dp_lib.get_min_sigma(
                sample_rate=sample_rate, num_steps=self.max_allow_iter, delta=self.delta, require_eps=self.epsilon
            )
            print(f'Under {self.max_allow_iter} iterations and the sample_rate = {sample_rate}, '
                  f'the sigma of client {self.index} is {self.sigma}')
        # if 'is_fixed_client_iter' is false, use 'sigma' to indicate the amount of noise added by the participant,
        # and 'client_iter' is calculated based on the privacy budget and 'sigma'
        else:
            self.sigma = float(dp_config['sigma'])
            sample_rate = 1 / len(self.data_loader)
            self.max_allow_iter = dp_lib.get_client_iter(
                sample_rate=sample_rate, max_eps=self.epsilon, delta=self.delta, sigma=self.sigma
            )
            print(f'Under sigma = {self.sigma} and sample_rate = {sample_rate}, '
                  f'the maximum iterations of client {self.index} is {self.max_allow_iter}')

    @timer
    def edge_train(self):
        """
        Update local model using local dataset
        """
        if self.model is None:
            raise ValueError('Model is not created for client: {0}'.format(self.index))

        self.model.to(self.device)
        self.model.train()

        losses = []

        if self.current_iter + self.local_epochs * len(self.data_loader) <= self.max_allow_iter:
            local_allow_iter = self.local_epochs * len(self.data_loader)
        else:
            local_allow_iter = self.max_allow_iter - self.current_iter
        self.current_iter = self.current_iter + self.local_epochs * len(self.data_loader)

        _batch_idx = 0

        while True:
            for x_batch, y_batch in self.data_loader:
                if _batch_idx == local_allow_iter:
                    break

                # print(_batch_idx)
                _batch_idx = _batch_idx + 1
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

            if _batch_idx == local_allow_iter:
                break

        # self.used_eps = self.privacy_engine.accountant.get_epsilon(
        #     delta=self.delta
        # )

        # print("\t\tLoss value: {:.6f}".format(sum(losses) / len(losses)))
        # print("\t\tEpsilon is used: {:.6f}".format(self.used_eps))
        gc.collect()

        return losses

    def receive_data(self, x, y):
        self.x_train = x
        self.y_train = y

    def receive_and_init_model(self, model_fn: Callable, model_weights):
        temp_model = model_fn().to(self.device)
        model_lib.set_model_weights(temp_model, model_weights, used_device=self.device)
        self.model = temp_model
