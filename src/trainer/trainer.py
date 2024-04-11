import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Any, Callable
from tqdm import tqdm


class Trainer:
    def __init__(self, train_module: nn.Module):
        
        self.train_module = train_module 


    def fit(self, 
            train_loader: DataLoader,
            num_epochs: int,
            device: str | int,
            unpacker: Callable,
            metrics: bool = False,
            val_loader: DataLoader | None = None):


        self.call("before_all_epochs")

        epochs_run = self.train_module.epochs_run

        for epoch in range(epochs_run + 1, num_epochs + 1):
            self.call("before_train_epoch_pass")

            self.epoch_pass(which="train", 
                            epoch=epoch, 
                            loader=train_loader, 
                            device=device, 
                            unpacker=unpacker)

            self.call("after_train_epoch_pass")

            if val_loader is not None:
                self.call("before_validation_epoch_pass")

                self.epoch_pass(which="val",
                                epoch=epoch,
                                loader=val_loader,
                                device=device,
                                unpacker=unpacker)

                self.call("after_validation_epoch_pass")

        self.call("after_all_epochs")


    def epoch_pass(self, 
                   which: str,
                   epoch: int,
                   loader: DataLoader, 
                   device: str | int, 
                   unpacker: Callable):

        pbar = tqdm(loader)

        for batch_idx, data in enumerate(pbar):
            data = unpacker(data, device)

            if which == "train":
                self.call("before_train_batch_pass")
                self.train_module.train_batch_pass(*data)
                self.call("after_train_batch_pass")
                pbar.set_postfix(
                    None, 
                    True,
                    EPOCH=epoch,
                    **self.train_module.logger._avg_epoch_history
                )

            elif which == "val":
                self.call("before_validation_batch_pass")
                self.train_module.val_batch_pass(*data)
                self.call("after_validation_batch_pass")
                pbar.set_postfix(
                    None, 
                    True, 
                    EPOCH=epoch,
                    **self.train_module.logger._avg_epoch_history
                )


    def call(self, where_at):
        for callback in self.train_module.callbacks:
            if hasattr(callback, where_at):
                callback(self)
