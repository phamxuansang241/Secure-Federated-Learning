from tek4fed.server_lib import BaseServer
from tek4fed.model_lib import get_model_weights, set_model_weights
from tek4fed.decorator import print_decorator
import numpy as np


class FedServer(BaseServer):
    """
    A subclass of BaseServer that implements federated learning with average aggregation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train(self):
        """
        Trains the model using the federated learning method.
        """
        self.train_fed()

    def train_fed(self):
        """
        Implementation of the federated learning training method.
        """

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

                self.epoch_losses.append(client_losses[-1])

                self.client_model_weights.append(get_model_weights(client.model))
            self.summarize_weights()

            epoch_mean_loss = np.mean(self.epoch_losses)
            self.global_train_losses.append(epoch_mean_loss)

            return self.global_train_losses[-1]

        for epoch in range(self.training_config['global_epochs']):
            print_decorator(epoch)(epoch_train)()
            # testing current model_lib
            self.test_global_model()
