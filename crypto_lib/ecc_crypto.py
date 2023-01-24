from crypto_lib import generate_invertible_matrix, generate_mi_ni_matrix_client, generate_matrix
from tinyec import registry
from tinyec.ec import SubGroup, Curve, Point

MIN_VAL = 1
MAX_MAL = 18


class EccCrypto:
    def __init__(self, nb_client, mtx_size) -> None:
        self.curve = registry.get_curve('secp192r1')
        self.mtx_size = mtx_size
        self.nb_client = nb_client

        self.S_mtx, self.S_invert_mtx = generate_invertible_matrix(mtx_size)
        
        # initialize private parameters for clients
        self.client_private_parameter = {
            'm_i': {}, 'n_i': {},
            'r_i': {}, 's_i': {}
        }
        self.initialize_private_parameters()
        
        # initialize private keys for clients
        self.client_private_key = {
            'p_i': {}, 'q_i': {}, 
            'c_i': {}, 'd_i': {}
        }

        self.client_public_key = {
            'Q_i': {}, 'P_i': {}, 
            'C_i': {}, 'D_i': {}
        }

    def initialize_private_parameters(self):
        S_mtx, S_invert_mtx = generate_invertible_matrix(self.mtx_size)

        for client_index in range(self.nb_client):
            self.client_private_parameter['m_i'][client_index], self.private_parameter['n_i'][client_index] = \
            generate_mi_ni_matrix_client(S_mtx, S_invert_mtx, self.mtx_size, MIN_VAL, MAX_MAL)
            self.client_private_parameter['r_i'][client_index] = generate_matrix(self.mtx_size, MIN_VAL, MAX_MAL)
            self.client_private_parameter['s_i'][client_index] = generate_matrix(self.mtx_size, MIN_VAL, MAX_MAL)

    def initialize_private_keys(self):
        for client_index in range(self.nb_client):
            self.client_private_key['p_i'][client_index] = generate_matrix(self.mtx_size, MIN_VAL, MAX_MAL)
            self.client_private_key['q_i'][client_index] = generate_matrix(self.mtx_size, MIN_VAL, MAX_MAL)
            self.client_private_key['c_i'][client_index] = generate_matrix(self.mtx_size, MIN_VAL, MAX_MAL)
            self.client_private_key['d_i'][client_index] = generate_matrix(self.mtx_size, MIN_VAL, MAX_MAL)
    
    


