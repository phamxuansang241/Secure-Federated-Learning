from abc import ABC, abstractmethod
from tek4fed.summarizer_lib.weight_summarizer import WeightSummarizer
from typing import Callable


class ServerInterface(ABC):
    @abstractmethod
    def __init__(
            self,
            model_fn: Callable,
            weight_summarizer: WeightSummarizer,
            dp_mode,
            fed_config=None,
            dp_config=None):
        pass

    @abstractmethod
    def setup(self):
        """
        Sets up the federated learning environment for the server,
        including server's DataLoader and clients' configurations.
        """
        pass

    @abstractmethod
    def create_clients(self):
        """
        Creates clients for training.
        """
        pass

    @abstractmethod
    def select_clients(self):
        """
        Randomly selects a subset of clients to participate in the current
        training round based on the client_fraction attribute.

        Returns:
            selected_clients (numpy.ndarray): The array of selected clients.
        """
        pass

    @abstractmethod
    def create_model_with_updated_weights(self):
        """
        Creates a model using the global model function and sets its weights
        to the current global model weights.

        Returns:
            checkpoint_path (nn.Module): The created model with the updated weights.
        """
        pass

    @abstractmethod
    def init_for_new_epoch(self):
        """
        Clears the client model weights and epoch losses lists for a new communication round.
        """
        pass

    @abstractmethod
    def send_model(self, client):
        """
        Sends the global model to a client.

        Args:
            client (Client): The client to send the model to.
        """
        pass

    @abstractmethod
    def receive_results(self, client):
        """
        Receives the updated model weights from a client and stores them in
        the client_model_weights list. Resets the client's model.

        Args:
            client (Client): The client to receive the results from.
        """
        pass

    @abstractmethod
    def summarize_weights(self, encrypt_mode=False):
        pass

    @abstractmethod
    def test_global_model(self):
        pass

    @abstractmethod
    def save_model_weights(self, path):
        pass

    @abstractmethod
    def load_model_weights(self, path, by_name: bool = False):
        pass

    @abstractmethod
    def receive_data(self, x, y):
        pass
