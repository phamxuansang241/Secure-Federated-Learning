import numpy as np

from encryption_lib import *
from fastecdsa import keys, curve

MIN_VAL = 1
MAX_VAL = 18


class EccEncryption:
    def __init__(self, nb_client, mtx_size) -> None:
        self.curve = curve.P192
        self.mtx_size = mtx_size
        self.nb_client = nb_client

        # initialize private parameters for clients
        self.client_private_parameter = {
            'm_i': {}, 'n_i': {},
            'r_i': {}, 's_i': {}
        }
        self.initialize_client_private_parameter()
        
        # initialize private keys for clients
        self.client_private_key = {
            'p_i': {}, 'q_i': {}, 
            'c_i': {}, 'd_i': {}
        }
        self.initialize_client_private_key()

        # initialize public keys for clients
        self.client_public_key = {
            'P_i': {}, 'Q_i': {}, 
            'C_i': {}, 'D_i': {}
        }
        self.initialize_client_public_key()

        self.server_public_key = {
            'P': None, 'Q':  None, 
            'C': None, 'D': None
        }

        self.client_encoded_message = {
            'A_i': {}, 'B_i': {}, 
            'R_i': {}, 'S_i': {}, 
            'T_i': {}
        }

        self.server_decoded_message = {
            'RR': None, 'SS': None,
            'M': np.zeros((self.mtx_size, self.mtx_size)),
            'N': np.zeros((self.mtx_size, self.mtx_size))
        }

        self.server_sum_weight = None

    def initialize_client_private_parameter(self):
        S_mtx, S_invert_mtx = generate_invertible_matrix(self.mtx_size)

        for client_index in range(self.nb_client):
            self.client_private_parameter['m_i'][client_index], self.client_private_parameter['n_i'][client_index] = \
                generate_mi_ni_matrix_client(S_mtx, S_invert_mtx, self.mtx_size, MIN_VAL, MAX_VAL)
            self.client_private_parameter['r_i'][client_index] = generate_matrix(self.mtx_size, MIN_VAL, MAX_VAL)
            self.client_private_parameter['s_i'][client_index] = generate_matrix(self.mtx_size, MIN_VAL, MAX_VAL)

    def initialize_client_private_key(self):
        list_type_key = ['p_i', 'q_i', 'c_i', 'd_i']
        for client_index in range(self.nb_client):
            for type_key in list_type_key:
                self.client_private_key[type_key][client_index] = generate_matrix(self.mtx_size, MIN_VAL, MAX_VAL)

    def initialize_client_public_key(self):
        list_type_public_key = ['P_i', 'Q_i', 'C_i', 'D_i']
        list_type_private_key = ['p_i', 'q_i', 'c_i', 'd_i']
        for client_index in range(self.nb_client):
            for type_public_key, type_private_key in zip(list_type_public_key, list_type_private_key):
                self.client_public_key[type_public_key][client_index] = ecc_multiply_matrix_with_point(
                    self.client_private_key[type_private_key][client_index], self.mtx_size, self.curve
                )

    def calculate_server_public_key(self):
        list_type_server_key = ['P', 'Q', 'C', 'D']
        list_type_client_key = ['P_i', 'Q_i', 'C_i', 'D_i']

        for client_index in range(self.nb_client):
            for (type_server_key, type_client_key) in zip(list_type_server_key, list_type_client_key):
                if client_index == 0:
                    self.server_public_key[type_server_key] = self.client_public_key[type_client_key][client_index]
                else:
                    self.server_public_key[type_server_key] = ecc_add_pointmatrix_with_pointmatrix(
                        self.server_public_key[type_server_key], self.client_public_key[type_client_key][client_index], 
                        self.mtx_size, self.curve
                    )

    def calculate_encoded_message_phase_one(self):
        """
        Encoded messages for phase one include: A_i, B_i, R_i, S_i 
        for each client
        """
        for client_index in range(self.nb_client):
            # calculate A_i
            self.client_encoded_message['A_i'][client_index] = \
                self.client_private_parameter['m_i'][client_index] + self.client_private_parameter['r_i'][client_index]
            
            # calculate B_i
            self.client_encoded_message['B_i'][client_index] = \
                self.client_private_parameter['n_i'][client_index] + self.client_private_parameter['s_i'][client_index]
            
            # calculate R_i
            ri_G = ecc_multiply_matrix_with_point(
                self.client_private_parameter['r_i'][client_index], 
                self.mtx_size, self.curve
            )
            qi_P = ecc_multiply_matrix_with_pointmatrix(
                self.client_private_key['q_i'][client_index], self.server_public_key['P'], 
                self.mtx_size, self.curve
            )
            pi_Q = ecc_multiply_matrix_with_pointmatrix(
                self.client_private_key['p_i'][client_index], self.server_public_key['Q'], 
                self.mtx_size, self.curve
            )
            temp = ecc_add_pointmatrix_with_pointmatrix(ri_G, qi_P, self.mtx_size, self.curve)
            self.client_encoded_message['R_i'][client_index] = ecc_subtract_pointmatrix_with_pointmatrix(
                temp, pi_Q,
                self.mtx_size, self.curve
            )

            # calculate S_i
            si_G = ecc_multiply_matrix_with_point(
                self.client_private_parameter['s_i'][client_index], 
                self.mtx_size, self.curve
            )
            ci_D = ecc_multiply_matrix_with_pointmatrix(
                self.client_private_key['c_i'][client_index], self.server_public_key['D'], 
                self.mtx_size, self.curve
            )
            di_C = ecc_multiply_matrix_with_pointmatrix(
                self.client_private_key['d_i'][client_index], self.server_public_key['C'], 
                self.mtx_size, self.curve
            )
            temp = ecc_add_pointmatrix_with_pointmatrix(si_G, ci_D, self.mtx_size, self.curve)
            self.client_encoded_message['S_i'][client_index] = ecc_subtract_pointmatrix_with_pointmatrix(
                temp, di_C,
                self.mtx_size, self.curve
            )

    def calculate_decoded_message_phase_one(self):
        """
        Decoded messages for phase one include: M, N
        """

        # calculate R and S
        for client_index in range(self.nb_client):
            print(client_index)
            if client_index == 0:
                print('ds')
                print(self.client_encoded_message['R_i'][client_index])
                print(type(self.client_encoded_message['R_i'][client_index]))
                self.server_decoded_message['RR'] = self.client_encoded_message['R_i'][client_index]
                self.server_decoded_message['SS'] = self.client_encoded_message['S_i'][client_index]
            else:
                self.server_decoded_message['RR'] = ecc_add_pointmatrix_with_pointmatrix(
                    self.server_decoded_message['RR'], self.client_encoded_message['R_i'][client_index],
                    self.mtx_size, self.curve
                )
                self.server_decoded_message['SS'] = ecc_add_pointmatrix_with_pointmatrix(
                    self.server_decoded_message['SS'], self.client_encoded_message['S_i'][client_index],
                    self.mtx_size, self.curve
                )
        
        # predict r and s such that rG=R and sG=S
        r_pred, s_pred = self.predict_r_s()

        # calculate M and N
        for client_index in range(self.nb_client):
            self.server_decoded_message['M'] = self.server_decoded_message['M'] + self.client_encoded_message['A_i'][client_index]
            self.server_decoded_message['N'] = self.server_decoded_message['N'] + self.client_encoded_message['B_i'][client_index]

        self.server_decoded_message['M'] = self.server_decoded_message['M'] - r_pred
        self.server_decoded_message['N'] = self.server_decoded_message['N'] - s_pred

    def predict_r_s(self):
        """
        predicting r and s for phase one
        """
        r_pred = []
        s_pred = []

        for i in range(self.mtx_size):
            for j in range(self.mtx_size):
                point_r = Point(
                    self.curve, 
                    self.server_decoded_message['RR'][i, j, 0], self.server_decoded_message['RR'][i, j, 1]
                )
                point_s = Point(
                    self.curve, 
                    self.server_decoded_message['SS'][i, j, 0], self.server_decoded_message['SS'][i, j, 1]
                )

                for di in range(1, MAX_VAL*self.nb_client):
                    if di * self.curve.g == point_r:
                        r_pred.append(di)
                    if di * self.curve.g == point_s:
                        s_pred.append(di)
        
        r_pred = np.array(r_pred).reshape(self.mtx_size, self.mtx_size)
        s_pred = np.array(s_pred).reshape(self.mtx_size, self.mtx_size)

        return r_pred, s_pred

    def calculate_encoded_message_phase_two(self):
        """
        Encoded messages for phase one include: T_i 
        for each client
        """
        for client_index in range(self.nb_client):
            self.client_encoded_message['T_i'][client_index] = \
                self.weight_mtx[client_index] + self.client_private_parameter['m_i'][client_index]*self.server_decoded_message['N'] \
                    - self.client_private_parameter['n_i'][client_index]*self.server_decoded_message['M']

    def calculate_decoded_message_phase_two(self):
        """
        Decoded messages for phase two include: sum of weights of clients
        """
        for client_index in range(self.nb_client):
            if client_index == 0:
                self.server_sum_weight = self.client_encoded_message['T_i'][client_index]
            else:
                self.server_sum_weight = self.server_sum_weight + self.client_encoded_message['T_i'][client_index]

    
