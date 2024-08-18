import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader


class Trainer:
    def __init__(
        self, model, loss_fn, optimizer, train_dataloader, val_dataloader, batch_size, learning_rate, device
    ):
        """
        Method to init the trainer setting all the parameters needed
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device
        self.loss_fn = loss_fn
        self.optimizer = optimizer

        pass

    def train(self, epochs):
        """
        Method to train the network on the training set
        """
        for epoch in range(epochs):
            size = len(self.train_dataloader)
            self.model.train()
            loss = 0.0
            for batch, (X, y) in enumerate(self.train_dataloader):
                """ TODO
                        pred = model(X)
                    loss = loss_fn(pred, y)

                    # Backpropagation
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    if batch % 100 == 0:
                        loss, current = loss.item(), batch * batch_size + len(X)
                        print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
                """       
        raise NotImplementedError("This function is not implemented yet")

    def validate(self):
        """
        Method to validate the network training on the validation set
        """
        raise NotImplementedError("This function is not implemented yet")
