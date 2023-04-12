from tek4fed.server_lib import BaseServer
from tek4fed.model_lib import set_model_weights
from tek4fed.encryption_lib import EccEncryption
from math import *
import numpy as np


class EccEncryptionServer(BaseServer):
    """
    A subclass of BaseServer that implements federated learning
    with ECC encryption for multi-party secure computations.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train(self):
        """
        Trains the model using the federated learning method with ECC encryption.
        """
        self.train_fed_ecc_encryption(short_ver=False)

    def train_fed_ecc_encryption(self, short_ver=True):
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
