import numpy as np


class WeightSummarizer:
    def __init__(self):
        pass

    def process(self, client_weight_list, global_weights=None):
        raise NotImplementedError()

    def process_encryption(self, nb_clients, sum_client_weight):
        raise NotImplementedError()


class FedAvg(WeightSummarizer):
    def __init__(self, nu: float = 1.0):
        super().__init__()
        self.nu = nu

    def process(self, client_weight_list, global_weight_per_layer=None):
        nb_clients = len(client_weight_list)
        weight_average = [np.zeros_like(w) for w in client_weight_list[0]]

        for layer_index in range(len(weight_average)):
            w = weight_average[layer_index]

            if global_weight_per_layer is not None:
                global_weight_mtx = global_weight_per_layer[layer_index]
            else:
                global_weight_mtx = np.zeros_like(w)

            for client_weight_index in range(nb_clients):
                client_weight_mtx = client_weight_list[client_weight_index][layer_index]
                client_weight_diff_mtx = client_weight_mtx - global_weight_mtx
                w = w + client_weight_diff_mtx

            weight_average[layer_index] = (self.nu * w / nb_clients) + global_weight_mtx

        return weight_average

    def process_encryption(self, nb_clients, sum_client_weight):
        new_weight = [np.zeros_like(w, dtype=np.float32) for w in sum_client_weight]

        for layer_index in range(len(sum_client_weight)):
            new_weight[layer_index] = (self.nu * sum_client_weight[layer_index]) / nb_clients
            new_weight[layer_index] = np.float32(new_weight[layer_index])

        return new_weight




