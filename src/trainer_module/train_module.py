import os
from torch.nn import CrossEntropyLoss, Module
from torch import Tensor, argmax, save, load, no_grad
from torch.optim import Adam
from torch.nn import DataParallel

from src.logger.csv_logger import CSVLogger


class TrainModule(Module):
    def __init__(self,
                 model: Module,
                 device: int | str,
                 accuracy,
                 conf_mat,
                 state_dict_root: str | None = None,
                 loss_log_root: str | None = None):
        super().__init__()
        
        self.model = model
        self.device = device

        # self.loss_fn = YOLOLoss()
        self.loss_fn = CrossEntropyLoss()
        self.optimizer = Adam(self.model.parameters(), lr=.0001)
        
        if state_dict_root is not None:
            self.state_dict_root = state_dict_root
            if os.path.isfile(os.path.join(self.state_dict_root, "train_ckp.pth")):
                self.load_checkpoint()

        if loss_log_root is not None:
            self.loss_log_root = loss_log_root
            self.logger = CSVLogger(self.loss_log_root)

        self.epochs_run = 0

        self.accuracy = accuracy 
        self.conf_mat = conf_mat 

    def metrics(self):
        return [
            self.accuracy,
            self.conf_mat
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


    def val_batch_pass(self, *unpacked_loader_data):
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


    def save_checkpoint(self, which, epoch, save_to: str | None = None):
        checkpoint = {}
        if save_to is None:
            save_to = os.path.join(
                self.state_dict_root, f"{which}_ckp.pth"
            )

        if isinstance(self.model, DataParallel):
            checkpoint["MODEL_STATE"] = self.model.module.state_dict()
            checkpoint["OPTIMIZER_STATE"] = self.optimizer.state_dict()
            checkpoint["EPOCHS_RUN"] = epoch
        else:
            checkpoint["MODEL_STATE"] = self.model.state_dict()
            checkpoint["OPTIMIZER_STATE"] = self.optimizer.state_dict()
            checkpoint["EPOCHS_RUN"] = epoch

        save(checkpoint, save_to)
        print(f"EPOCH {epoch} checkpoint saved at {save_to}")


    def load_checkpoint(self, load_from: str | None = None):
        if load_from is None:
            load_from = os.path.join(
                self.state_dict_root, f"train_ckp.pth"
            )
        train_checkpoint = load(load_from)

        for state in train_checkpoint["OPTIMIZER_STATE"]["state"].values():
            for k, v in state.items():
                if isinstance(v, Tensor):
                    state[k] = v.to(self.device)

        if isinstance(self.model, DataParallel):
            self.model.module.load_state_dict(train_checkpoint["MODEL_STATE"])
            self.optimizer.load_state_dict(train_checkpoint["OPTIMIZER_STATE"])
            self.epochs_run = train_checkpoint["EPOCHS_RUN"]
        else:
            self.model.load_state_dict(train_checkpoint["MODEL_STATE"])
            self.optimizer.load_state_dict(train_checkpoint["OPTIMIZER_STATE"])
            self.epochs_run = train_checkpoint["EPOCHS_RUN"]
