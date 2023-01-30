from tek4fed.model_lib import get_model_weights, weight_to_mtx, split_weight, get_model_infor
from encryption_lib.elgamal_encrypt import *
import numpy as np
import random as rd


MIN_VAL = 1
MAX_VAL = 10


class ElGamalEncryption:
    def __init__(self, nb_client, mtx_size) -> None:
        self.g = 2
        self.p = gen_prime(bit=128)
        self.mtx_size = mtx_size
        self.nb_client = nb_client
        self.K_mtx = generate_invertible_matrix(self.mtx_size)

        # initialize noise matrix for clients
        self.client_noise_mtx = {
            'r_i': {}
        }

        # initialize private keys for clients
        self.client_private_key = {
            'x_i': {}, 'y_i': {}
        }

        # initialize public keys for clients
        self.client_public_key = {
            'X_i': {}, 'Y_i': {}
        }

        # initialize public keys for server
        self.server_public_key= {
            'X': None, 'Y': None
        }

        self.client_encoded_message = {
            'M_i': {}, 'H_i': {}, 'S_i': {}
        }

        self.server_decoded_message = {
            'S': np.zeros((self.mtx_size, self.mtx_size)), 
            'Q': np.zeros((self.mtx_size, self.mtx_size)), 
            'R': [],
            'M': np.ones((self.mtx_size, self.mtx_size)),
            'H': np.ones((self.mtx_size, self.mtx_size))
        }

    def generate_client_noise_mtx(self):
        for client_index in range(self.nb_client):
            self.client_noise_mtx['r_i'][client_index] = np.random.randint(
                low=MIN_VAL, high=MAX_VAL, size=(self.mtx_size, self.mtx_size))
    
    def generate_client_key(self):
        for client_index in range(self.nb_client):
            self.client_private_key['x_i'][client_index] = rd.randint(1, self.p - 2)
            self.client_private_key['y_i'][client_index] = rd.randint(1, self.p - 2)
            
            self.client_public_key['X_i'][client_index] = power_mode(
                self.g, self.client_private_key['x_i'][client_index], self.p
                )
            self.client_public_key['Y_i'][client_index] = power_mode(
                self.g, self.client_private_key['y_i'][client_index], self.p
                )

    def calculate_server_public_key(self, selected_clients):
        self.server_public_key['X'] = 1
        self.server_public_key['Y'] = 1

        for client in selected_clients:
            self.server_public_key['X'] = (self.server_public_key['X']*self.client_public_key['X_i'][client.index]) % self.p
            self.server_public_key['Y'] = (self.server_public_key['X']*self.client_public_key['Y_i'][client.index]) % self.p
    
    def encoded_message(self, client):
        """
        Encoded messages include: M_i, H_i, S_i 
        for each client
        """
        # get noise matrix, private keys of client
        r_i = self.client_noise_mtx['r_i'][client.index]
        x_i = self.client_private_key['x_i'][client.index]
        y_i = self.client_private_key['y_i'][client.index]

        # calculate M_i and H_i
        mi_row = []
        hi_row = []

        for row in range(self.mtx_size):
            mi_col = []
            hi_col = []
            for col in range(self.mtx_size):
                g_row_col = power_mode(self.g, r_i[row][col], self.p)
                Xy_row_col = power_mode(self.server_public_key['X'], y_i, self.p)
                mi_row_col = (g_row_col * Xy_row_col) % self.p

                hi_row_col = power_mode(self.server_public_key['Y'], x_i, self.p)
                hi_row_col = inverse_power_mode(hi_row_col, self.p)

                mi_col.append(mi_row_col)
                hi_col.append(hi_row_col)

            mi_row.append(mi_col)
            hi_row.append(hi_col)

        self.client_encoded_message['M_i'][client.index] = mi_row
        self.client_encoded_message['H_i'][client.index] = hi_row

        # calculate S_i
        client_weights = get_model_weights(client.model)
        client_weight_mtx = weight_to_mtx(client_weights, self.mtx_size)

        self.client_encoded_message['S_i'][client.index] = np.dot(client_weight_mtx, self.K_mtx) + r_i 

    def decoded_message(self, selected_clients):
        self.server_decoded_message['S'] = np.zeros((self.mtx_size, self.mtx_size))
        self.server_decoded_message['M'] = np.ones((self.mtx_size, self.mtx_size))
        self.server_decoded_message['H'] = np.ones((self.mtx_size, self.mtx_size))
        for client in selected_clients:
            self.server_decoded_message['S'] = self.server_decoded_message['S'] + self.client_encoded_message['S_i'][client.index]
            self.server_decoded_message['M'] = np.multiply(self.server_decoded_message['M'], self.client_encoded_message['M_i'][client.index]) % self.p
            self.server_decoded_message['H'] = np.multiply(self.server_decoded_message['H'], self.client_encoded_message['H_i'][client.index]) % self.p


        self.server_decoded_message['R'] = []
        for row in range(self.mtx_size):
            for col in range(self.mtx_size):
                for d in range(0, MAX_VAL*self.nb_client + 1):
                    g_d = power_mode(self.g, d, self.p)

                    check_val = (self.server_decoded_message['M'][row][col]*self.server_decoded_message['H'][row][col]) % self.p
                    if g_d == check_val:
                        self.server_decoded_message['R'].append(check_val)

        self.server_decoded_message['R'] = np.array(self.server_decoded_message['R']).reshape(
            self.mtx_size, self.mtx_size
        )

        return np.dot(self.server_decoded_message['S']-self.server_decoded_message['R'], 
        self.K_mtx)

    
