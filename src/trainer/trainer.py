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


        epochs_run = self.train_module.epochs_run
        for epoch in range(epochs_run + 1, num_epochs + 1):
            self.epoch_pass(which="train", 
                            epoch=epoch, 
                            loader=train_loader, 
                            device=device, 
                            unpacker=unpacker)

            self.train_module.logger.log_epoch("train")

            if self.train_module.logger.save_checkpoint_flag("train"):
                self.train_module.save_checkpoint("train", epoch)

            if metrics:
                for metric in self.train_module.metrics():
                    metric.reset_state(epoch=epoch, which="train")

            if val_loader is not None:
                self.epoch_pass(which="val",
                                epoch=epoch,
                                loader=val_loader,
                                device=device,
                                unpacker=unpacker)

                self.train_module.logger.log_epoch("val")

                if self.train_module.logger.save_checkpoint_flag("val"):
                    self.train_module.save_checkpoint("val", epoch)

                if metrics:
                    for metric in self.train_module.metrics():
                        metric.reset_state(epoch=epoch, which="val")


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
                self.train_module.train_batch_pass(*data)
                pbar.set_postfix(
                    None, 
                    True,
                    EPOCH=epoch,
                    **self.train_module.logger._avg_epoch_history
                )

            elif which == "val":
                self.train_module.val_batch_pass(*data)
                pbar.set_postfix(
                    None, 
                    True, 
                    EPOCH=epoch,
                    **self.train_module.logger._avg_epoch_history
                )
