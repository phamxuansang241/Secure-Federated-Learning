from tek4fed.decorator import timer
from tek4fed.client import BaseClient
import torch
from torch.utils.data import DataLoader
import gc


class NormalClient(BaseClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self, **client_config):
        super().setup(**client_config)

    @timer
    def edge_train(self):
        print(self.device)
        if self.model is None:
            raise ValueError('Model is not created for client: {0}'.format(self.index))
        self.model.to(self.device)
        # set the model_lib in training mode
        self.model.train()

        losses = []
        local_epochs = self.training_config['local_epochs']
        batch_size = self.training_config['batch_size']
        data_loader = DataLoader(self.dataset, batch_size=batch_size)
        for e in range(0, local_epochs):
            # loop over the training set
            for (x_batch, y_batch) in data_loader:
                # send the input to the device
                (x_batch, y_batch) = (x_batch.to(self.device),
                                      y_batch.long().to(self.device))

                # perform a forward pass and calculate the training loss
                pred = self.model(x_batch)
                loss = self.criterion(pred, y_batch)

                # zero out the gradients, perform the backpropagation step,
                # and update the weights
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                losses.append(loss.item())
                torch.cuda.empty_cache()

        print("\t\t Loss value: {:.6f}".format(sum(losses) / len(losses)))

        del data_loader
        gc.collect()

        return losses
