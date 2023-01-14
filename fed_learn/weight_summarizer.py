import numpy as np


class WeightSummarizer:
    def __init__(self):
        pass

    def process(self, client_weight_list, global_weights=None):
        raise NotImplementedError()


class FedAvg(WeightSummarizer):
    def __init__(self, nu: float = 1.0):
        super().__init__()
        self.nu = nu

    def process(self, client_weight_list, global_weights_per_layer=None):
        nb_clients = len(client_weight_list)
        weights_average = [np.zeros_like(w) for w in client_weight_list[0]]

        for layer_index in range(len(weights_average)):
            w = weights_average[layer_index]

            if global_weights_per_layer is not None:
                global_weight_mtx = global_weights_per_layer[layer_index]
            else:
                global_weight_mtx = np.zeros_like(w)

            for client_weight_index in range(nb_clients):
                client_weight_mtx = client_weight_list[client_weight_index][layer_index]
                client_weight_diff_mtx = client_weight_mtx - global_weight_mtx
                w = w + client_weight_diff_mtx

            weights_average[layer_index] = (self.nu * w / nb_clients) + global_weight_mtx

        return weights_average



