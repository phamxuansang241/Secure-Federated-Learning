from tek4fed.server_lib import BaseServer
from tek4fed.model_lib import get_model_weights, set_model_weights
from tek4fed.decorator import print_decorator
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


class StandaloneServer(BaseServer):
    """
    A subclass of BaseServer that implements federated learning with average aggregation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.global_test_metrics['loss'] = {}
        self.global_test_metrics['accuracy'] = {}

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

                if self.dp_mode:
                    if client.current_iter > client.max_allow_iter:
                        break

                client_losses = client.edge_train()

                self.test_client_model(client)

            return self.global_train_losses[-1]

        for epoch in range(self.training_config['global_epochs']):
            print_decorator(epoch)(epoch_train)()
            # testing current model_lib
            self.test_global_model()

    def test_client_model(self, client):
        client_model = client.model
        client_model.to(client.device)

        batch_size = self.training_config['batch_size']
        data_loader = DataLoader(self.dataset, batch_size=batch_size)
        loss_fn = nn.CrossEntropyLoss()
        total_test_loss = 0
        test_correct = 0

        with torch.no_grad():
            client_model.eval()

            for (x_batch, y_batch) in data_loader:
                (x_batch, y_batch) = (x_batch.to(self.device),
                                      y_batch.long().to(self.device))

                pred = client_model(x_batch)
                total_test_loss = total_test_loss + loss_fn(pred, y_batch)
                test_correct = test_correct + (pred.argmax(1) == y_batch).type(
                    torch.float
                ).sum().item()

        avg_test_loss = (total_test_loss / len(data_loader)).cpu().detach().item()
        test_correct = test_correct / len(self.x_test)

        