from tek4fed.server import BaseServer
from tek4fed.model_lib import set_model_weights
from tek4fed.encryption_lib import ElGamalEncryption
from math import *
import numpy as np


class ElGamalEncryptionServer(BaseServer):
    """
    A subclass of BaseServer that implements federated learning
    with ElGamal encryption for multi-party secure computations.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train(self):
        """
        Trains the model using the federated learning method with ElGamal encryption.
        """
        self.train_fed_elgamal_encryption()

    def train_fed_elgamal_encryption(self):
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
            