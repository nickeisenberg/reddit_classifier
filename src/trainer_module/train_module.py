import os
from torch.nn import CrossEntropyLoss, Module
from torch import Tensor, argmax, save, load, no_grad
from torch.optim import Adam
from torch.nn import DataParallel

from src.callbacks import (
    Accuracy,
    CSVLogger,
    SaveBestCheckoint,
    ConfusionMatrix
)


class TrainModule(Module):
    def __init__(self,
                 model: Module,
                 device: int | str,
                 accuracy: Accuracy,
                 conf_mat: ConfusionMatrix,
                 logger: CSVLogger,
                 save_best: SaveBestCheckoint,
                 state_dict_root: str,
                 loss_log_root: str,
                 metrics_root: str):
        super().__init__()
        
        self.model = model
        self.device = device

        self.loss_fn = CrossEntropyLoss()
        self.optimizer = Adam(self.model.parameters(), lr=.0001)
        
        self.state_dict_root = state_dict_root
        self.loss_log_root = loss_log_root
        self.metrics_root = metrics_root 
        
        self.accuracy = accuracy 
        self.conf_mat = conf_mat 
        self.logger = logger
        self.save_best = save_best

        self.epochs_run = 0


    def callbacks(self):
        return [
            self.accuracy,
            self.conf_mat,
            self.logger,
            self.save_best
        ]


    def forward(self, x):
        return self.model(x)


    def train_batch_pass(self, *unpacked_loader_data):
        if not self.model.training:
            self.model.train()

        (inputs, masks), targets = unpacked_loader_data 
        
        self.optimizer.zero_grad()
        outputs = self.model(inputs, masks)
        loss: Tensor = self.loss_fn(outputs, targets)
        loss.backward()
        self.optimizer.step()

        targets = targets.detach()
        predictions = argmax(outputs, 1).detach()

        self.accuracy.log(predictions, targets)
        self.conf_mat.log(predictions, targets)

        self.logger.log_batch(
            {
                "total_loss": loss.item(),
                "accuracy": self.accuracy.accuracy
            }
        )


    def validation_batch_pass(self, *unpacked_loader_data):
        if self.model.training:
            self.model.eval()

        (inputs, masks), targets = unpacked_loader_data

        with no_grad():
            outputs: Tensor = self.model(inputs, masks)
        loss: Tensor = self.loss_fn(outputs, targets)

        targets = targets.detach()
        predictions = argmax(outputs, 1).detach()

        self.accuracy.log(predictions, targets)
        self.conf_mat.log(predictions, targets)

        self.logger.log_batch(
            {
                "total_loss": loss.item(),
                "accuracy": self.accuracy.accuracy
            }
        )
