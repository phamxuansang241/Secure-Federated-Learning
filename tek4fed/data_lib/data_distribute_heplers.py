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


def non_iid_label_dir_data_indices(nb_clients: int, labels: np.ndarray, beta: float=0.8):
    nb_class = len(np.unique(labels))
    data_len = len(labels)
    min_size = 0
    min_require_size = 10

    net_dataidx_map = []

    while min_size < min_require_size:
        idx_batch = [[] for _ in range(nb_clients)]
        for k in range(nb_class):
            idx_k = np.where(labels==k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(beta, nb_clients))

            proportions = np.array([p * (len(idx_j) < data_len / nb_clients) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])
    
    for j in range(nb_clients):
            np.random.shuffle(idx_batch[j])
            print('Client {} with data len {}'.format(j, len(idx_batch[j])))
            # print(idx_batch[j])
            net_dataidx_map.append(idx_batch[j])

    return net_dataidx_map


class DataHandler:
    def __init__(self, x_train, y_train):
        self.nb_classes = len(np.unique(y_train))
        self.x_train = x_train
        self.y_train = y_train

    def sampling(self, sampling_technique, nb_clients):
        if sampling_technique.lower() == 'iid':
            sampler_fn = iid_data_indices
        elif sampling_technique.lower() == 'noniid_labeldir':
            sampler_fn = non_iid_label_dir_data_indices
        
        client_data_indices = sampler_fn(nb_clients, self.y_train)

        return client_data_indices

    def assign_data_to_clients(self, clients, sampling_technique):
        sampled_data_indices = self.sampling(sampling_technique, len(clients))
        for client, data_indices in zip(clients, sampled_data_indices):
            x = self.x_train[data_indices]
            y = self.y_train[data_indices]
            client.receive_data(x, y)

