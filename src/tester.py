import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class Tester:
    def __init__(self, model, loss, test_dataloader, batch_size=32, device=None):
        """
        Method to init the tester setting all the parameters needed
        """
        self.model = model
        self.loss = loss
        self.test_dataloader = test_dataloader
        self.batch_size = batch_size
        self.device = device

    def test(self):
        """
        Method to test the network on the test set
        """
        self.model.eval()
        size = len(self.test_dataloader)
        num_batches = len(self.test_dataloader)
        test_loss, correct = 0, 0

        with torch.no_grad():
            for X, y in self.test_dataloader:
                pred = self.model(X)
                test_loss += self.loss(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= size
        print(
            f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
        )
        raise NotImplementedError("This function is not implemented yet")
