import random
import numpy as np


def iid_data_indices(nb_clients: int, labels: np.ndarray):
    labels = labels.flatten()
    data_len = len(labels)
    indices = np.arange(data_len)
    np.random.shuffle(indices)
    chunks = np.array_split(indices, nb_clients)

    return chunks


def non_iid_data_indices(nb_clients: int, labels: np.ndarray, nb_shards: int = 200):
    labels = labels.flatten()
    data_len = len(labels)

    indices = np.arange(data_len)
    indices = indices[labels.argsort()]

    shards = np.array_split(indices, nb_shards)
    random.shuffle(shards)
    shards_for_users = np.array_split(shards, nb_clients)
    indices_for_users = [np.hstack(x) for x in shards_for_users]

    return indices_for_users


class DataHandler:
    def __init__(self, x_train, y_train, x_test, y_test):
        self.nb_classes = len(np.unique(y_train))
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def sampling(self, sampling_technique, nb_clients):
        if sampling_technique.lower() == 'iid':
            sampler_fn = iid_data_indices
        else:
            sampler_fn = non_iid_data_indices
        client_data_indices = sampler_fn(nb_clients, self.y_train)

        return client_data_indices

    def assign_data_to_clients(self, clients, sampling_technique):
        sampled_data_indices = self.sampling(sampling_technique, len(clients))
        for client, data_indices in zip(clients, sampled_data_indices):
            x = self.x_train[data_indices]
            y = self.y_train[data_indices]
            client.receive_data(x, y)
