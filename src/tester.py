import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class Tester:
    def __init__(self, model, test_dataset, batch_size=32, device=None):
        """
        Method to init the tester setting all the parameters needed
        """
        self.model = model
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.device = device
        pass

    def test(self):
        """
        Method to test the network on the test set
        """
        raise NotImplementedError("This function is not implemented yet")
        """TODO
            model.eval()
            size = len(dataloader.dataset)
            num_batches = len(dataloader)
            test_loss, correct = 0, 0

            # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
            # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
            with torch.no_grad():
                for X, y in dataloader:
                    pred = model(X)
                    test_loss += loss_fn(pred, y).item()
                    correct += (pred.argmax(1) == y).type(torch.float).sum().item()

            test_loss /= num_batches
            correct /= size
            print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        """
