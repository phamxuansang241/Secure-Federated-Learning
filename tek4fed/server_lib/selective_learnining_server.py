from tek4fed.server_lib import BaseServer
from tek4fed.model_lib import set_model_weights, get_dssgd_update


class DssgdServer(BaseServer):
    """
    A subclass of BaseServer that implements
    decentralized stochastic gradient descent (DSSGD) - selective learning of Reza Shokri.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train(self):
        """
        Trains the model using the decentralized stochastic gradient descent (DSSGD)  method.
        """
        self.train_dssgd()

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
                self.global_model_weights = get_dssgd_update(client.model, self.global_model_weights,
                                                             self.model_infor['weights_shape'], theta_upload=0.9)

            self.test_global_model()
