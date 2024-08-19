import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader


class Trainer:
    def __init__(
        self,
        model,
        loss,
        optimizer,
        train_dataloader,
        val_dataloader,
        batch_size,
        learning_rate,
        device,
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
        self.loss = loss
        self.optimizer = optimizer

    def train(self, epochs):
        """
        Method to train the network on the training set
        """
        self.model.train()
        for epoch in range(epochs):
            size = len(self.train_dataloader)
            for b, (X, y) in enumerate(self.train_dataloader):

                # computing prediction, forward step
                pred = self.model(X)

                # computing loss
                self.loss(pred, y)

                # computing gradient of the loss, backward step
                self.loss.backward()

                # updating the model parameters
                self.optimizer.step()

                # reset gradient computational graph
                self.optimizer.zero_grad()

                if b % 100 == 0:
                    loss, current = loss.item(), b * self.batch_size + len(X)
                    print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        raise NotImplementedError("This function is not implemented yet")

    def validate(self):
        """
        Method to validate the network training on the validation set
        """
        raise NotImplementedError("This function is not implemented yet")
