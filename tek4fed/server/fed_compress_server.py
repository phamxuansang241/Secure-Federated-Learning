from tek4fed.server import BaseServer
from tek4fed.compress_params_lib import CompressParams
from tek4fed.decorator import print_decorator
from tek4fed.model_lib import set_model_weights
import numpy as np


class FedCompressServer(BaseServer):
    """
    A subclass of BaseServer that implements federated learning with model compression.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train(self):
        """
        Trains the model using the federated learning method with model compression.
        """
        self.train_fed_compress()

    def train_fed_compress(self):
        """
        Implementation of the federated learning with model compression training method.
        """
        compress_params = CompressParams(self.training_config['compress_digit'])

        def epoch_train():
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

            return self.global_train_losses[-1]

        for epoch in range(self.training_config['global_epochs']):
            print_decorator(epoch)(epoch_train)()
            # testing current model_lib
            self.test_global_model()
