import torch.nn as nn
import torch


def test_client_model(self, client):
    client_model = client.model
    client_model.to(client.device)

    loss_fn = nn.CrossEntropyLoss()
    total_test_loss = 0
    test_correct = 0

    with torch.no_grad():
        client_model.eval()

        for (x_batch, y_batch) in self.data_loader:
            (x_batch, y_batch) = (x_batch.to(self.device),
                                  y_batch.long().to(self.device))

            pred = client_model(x_batch)
            total_test_loss = total_test_loss + loss_fn(pred, y_batch)
            test_correct = test_correct + (pred.argmax(1) == y_batch).type(
                torch.float
            ).sum().item()

    avg_test_loss = (total_test_loss / len(self.data_loader)).cpu().detach().item()
    test_correct = test_correct / len(self.x_test)

