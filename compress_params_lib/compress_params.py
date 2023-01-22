import model_lib
import math
import numpy as np


class CompressParams:
    def __init__(self, compress_digit) -> None:
        self.compress_digit = compress_digit
        self.compress_number = math.pow(10, self.compress_digit)

        self.client_compress_config = {
            'ceil_max_weight': {}, 
            'encoded_weights': {}
        }

    def encode_model(self, client):
        client_weights = model_lib.get_model_weights(client.model)
        weights_shape, _ = model_lib.get_model_infor(client.model)

        flatten_weights = model_lib.flatten_weight(client_weights)
        self.client_compress_config['ceil_max_weight'][client.index] = math.ceil(np.max(np.abs(flatten_weights)))
        flatten_encoded_weights = np.ceil((flatten_weights*self.compress_number) / self.client_compress_config['ceil_max_weight'][client.index])
        self.client_compress_config['encoded_weights'][client.index] = model_lib.split_weight(flatten_encoded_weights, weights_shape)

    def decode_model(self, selected_clients):
        decoded_weights = []

        weights_shape, _ = model_lib.get_model_infor(selected_clients[0].model)

        for client in selected_clients:
            client_encoded_weights = self.client_compress_config['encoded_weights'][client.index]
            client_ceil_max_weight = self.client_compress_config['ceil_max_weight'][client.index]
            client_flatten_encoded_weights = model_lib.flatten_weight(client_encoded_weights)
            client_flatten_encoded_weights = client_flatten_encoded_weights * client_ceil_max_weight / self.compress_number
            client_decoded_weights = model_lib.split_weight(client_flatten_encoded_weights, weights_shape)

            decoded_weights.append(client_decoded_weights)

        return decoded_weights
