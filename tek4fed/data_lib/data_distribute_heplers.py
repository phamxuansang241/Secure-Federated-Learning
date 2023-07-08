import numpy as np


def iid_data_indices(nb_clients: int, labels: np.ndarray):
    labels = labels.flatten()
    data_len = len(labels)
    indices = np.arange(data_len)
    np.random.shuffle(indices)
    chunks = np.array_split(indices, nb_clients)

    return chunks


def non_iid_data_indices(nb_clients: int, labels: np.ndarray, num: int = 3):
    """
    Generate non-IID data indices for clients in a federated learning framework.

    The function divides the dataset into chunks according to the number of clients
    and the specified number of unique labels per client. The data distribution
    among clients will be non-IID.

    Parameters:
    nb_clients (int): The number of clients in the federated learning framework.
    labels (np.ndarray): A 1D NumPy array containing the labels of the dataset.
    num (int, optional): The number of unique labels per client. Default is 3.

    Returns:
    list: A list of lists containing the indices of data samples for each client.
    """
    nb_class = len(np.unique(labels))
    print("Number of classes: ", nb_class)
    times = [0] * nb_class
    client_labels = []

    for i in range(nb_clients):
        current_labels = [i % nb_class]
        times[i % nb_class] += 1
        j = 1
        while j < num:
            ind = times.index(min(times))
            if ind not in current_labels:
                j = j + 1
                current_labels.append(ind)
                times[ind] += 1

        print(f'Client {i} with labels {current_labels}')
        client_labels.append(current_labels)

    net_dataidx_map = {i: np.ndarray(0, dtype=np.int64) for i in range(nb_clients)}

    for label, count in enumerate(times):
        idx_k = np.where(labels == label)[0]
        np.random.shuffle(idx_k)
        print(f'Label: {label} with {len(idx_k)} samples is divided to {count} clients')

        split = np.array_split(idx_k, count)

        for client_id, label_indices in enumerate(client_labels):
            if label in label_indices:
                net_dataidx_map[client_id] = np.append(net_dataidx_map[client_id], split.pop(0))
                print(f'{net_dataidx_map[client_id]} with client {client_id}')
        
    chunks = [net_dataidx_map[i].tolist() for i in range(nb_clients)]

    return chunks


def non_iid_label_dir_data_indices(nb_clients: int, labels: np.ndarray, beta: float = 0.7):
    """
        Generate non-IID data indices for clients in a federated learning framework using Dirichlet distribution.

        The function creates a non-IID distribution of the dataset among the clients by allocating samples from
        each class to clients based on proportions drawn from a Dirichlet distribution.

        Parameters:
        -----------
        nb_clients: int
            The number of clients to divide the dataset among.

        labels: np.ndarray
            A numpy array containing the labels of the dataset.

        beta: float, optional, default: 0.7
            The concentration parameter of the Dirichlet distribution. Smaller values result in more skewed
            distributions among clients, while larger values result in more balanced distributions. Default is 0.7.
    """
    nb_class = len(np.unique(labels))
    data_len = len(labels)
    min_size = 0
    min_require_size = 10

    net_dataidx_map = []
    idx_batch = None

    while min_size < min_require_size:
        idx_batch = [[] for _ in range(nb_clients)]
        for k in range(nb_class):
            idx_k = np.where(labels == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(beta, nb_clients))

            mask = np.array([len(idx_j) < data_len / nb_clients for idx_j in idx_batch])
            proportions = proportions * mask
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    for client_id, idx_j in enumerate(idx_batch):
        np.random.shuffle(idx_j)
        print(f'Client {client_id} with data len {len(idx_j)}')
        net_dataidx_map.append(idx_j)

    return net_dataidx_map


class DataHandler:
    def __init__(self, x_train, y_train):
        self.nb_classes = len(np.unique(y_train))
        self.x_train = x_train
        self.y_train = y_train

    def sampling(self, sampling_technique, nb_clients):
        sampler_fn = {
            'iid': iid_data_indices,
            'noniid_label_dir': non_iid_label_dir_data_indices,
            'noniid_label_quantity': non_iid_data_indices
        }.get(sampling_technique.lower())

        if not sampler_fn:
            raise ValueError(f"Unsupported sampling technique: {sampling_technique}")

        client_data_indices = sampler_fn(nb_clients, self.y_train)

        return client_data_indices

    def assign_data_to_clients(self, clients, sampling_technique):
        sampled_data_indices = self.sampling(sampling_technique, len(clients))
        for client, data_indices in zip(clients, sampled_data_indices):
            print(data_indices)

            if isinstance(self.x_train, np.ndarray):
                x = self.x_train[data_indices]
            else:
                data_indices_list = data_indices.tolist()
                x = [self.x_train[i] for i in data_indices_list]
            y = self.y_train[data_indices]
            client.receive_data(x, y)
