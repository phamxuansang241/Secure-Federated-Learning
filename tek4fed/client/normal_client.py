from tek4fed.decorator import timer
from tek4fed.client import BaseClient
import torch
import gc


class NormalClient(BaseClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self, **client_config):
        super().setup(**client_config)

    @timer
    def edge_train(self):
        if self.model is None:
            raise ValueError('Model is not created for client: {0}'.format(self.index))
        self.model.to(self.device)
        # set the model_lib in training mode
        self.model.train()

        losses = []

        for e in range(0, self.local_epochs):
            # loop over the training set
            for (x_batch, y_batch) in self.data_loader:
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

        gc.collect()

        return losses
