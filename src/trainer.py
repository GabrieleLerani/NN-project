import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR

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
        weight_decay,
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
        self.optimizer = optim.AdamW(
            model.parameters(), learning_rate, weight_decay=weight_decay
        )

    def train(self, epochs):
        """
        Method to train the network on the training set
        """
        # we setup the scheduler for the training as written in the paper
        warm_up_s = LinearLR(
            optimizer=self.optimizer, start_factor=1e-8, total_iters=10
        )
        cosine_s = CosineAnnealingLR(optimizer=self.optimizer, T_max=90, eta_min=0)
        scheduler = SequentialLR(
            optimizer=self.optimizer, schedulers=[warm_up_s, cosine_s], milestones=[10]
        )

        self.model.train()
        for epoch in range(epochs):
            size = len(self.train_dataloader)
            for b, (X, y) in enumerate(self.train_dataloader):

                # reset gradient computational graph
                self.optimizer.zero_grad()

                # computing prediction, forward step
                pred = self.model(X)

                # computing loss
                self.loss(pred, y)

                # computing gradient of the loss, backward step
                self.loss.backward()

                # updating the model parameters
                self.optimizer.step()

                if b % 100 == 0:
                    loss, current = loss.item(), b * self.batch_size + len(X)
                    print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

            scheduler.step()

            print(f"Epoch: {epoch+1}/{epochs} completed")

        raise NotImplementedError("This function is not implemented yet")

    def validate(self):
        """
        Method to validate the network training on the validation set
        """
        raise NotImplementedError("This function is not implemented yet")
