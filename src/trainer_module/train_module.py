from torch.nn import CrossEntropyLoss, Module
from torch import Tensor, argmax, no_grad, load
from torch.optim import Adam

from src.callbacks import (
    Accuracy,
    CSVLogger,
    SaveBestCheckoint,
    ConfusionMatrix,
    ProgressBarUpdater
)


class TrainModule(Module):
    def __init__(self,
                 model: Module,
                 device: int | str,
                 accuracy: Accuracy,
                 conf_mat: ConfusionMatrix,
                 logger: CSVLogger,
                 save_best: SaveBestCheckoint,
                 progress_bar_updater: ProgressBarUpdater):

        super().__init__()
        
        self.device = device
        self.model = model.to(self.device)

        self.loss_fn = CrossEntropyLoss()
        self.optimizer = Adam(self.model.parameters(), lr=.0001)
        
        self.accuracy = accuracy 
        self.conf_mat = conf_mat 
        self.logger = logger
        self.save_best = save_best
        self.progress_bar_updater = progress_bar_updater 


    def callbacks(self):
        return [
            self.accuracy,
            self.conf_mat,
            self.logger,
            self.save_best,
            self.progress_bar_updater
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

        self.logger.log(
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

        self.logger.log(
            {
                "total_loss": loss.item(),
                "accuracy": self.accuracy.accuracy
            }
        )


    def evaluation_batch_pass(self, *unpacked_loader_data):
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

        self.logger.log(
            {
                "total_loss": loss.item(),
                "accuracy": self.accuracy.accuracy
            }
        )
