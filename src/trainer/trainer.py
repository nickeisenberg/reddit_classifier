import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Callable
from tqdm import tqdm


class Trainer:
    def __init__(self, train_module: nn.Module):
        self.train_module = train_module 
        
        self.current_epoch = 0 
        self.which_pass = "N/A" 

    def fit(self, 
            train_loader: DataLoader,
            num_epochs: int,
            device: str | int,
            unpacker: Callable,
            val_loader: DataLoader | None = None):

        self.call("before_all_epochs")

        epochs_run = self.train_module.epochs_run

        for epoch in range(epochs_run + 1, num_epochs + 1):
            self.current_epoch = epoch
            self.which_pass = "train" 

            self.epoch_pass(which="train", 
                            epoch=epoch, 
                            loader=train_loader, 
                            device=device, 
                            unpacker=unpacker)

            if val_loader is not None:
                self.which_pass = "validation"

                self.epoch_pass(which="validation",
                                epoch=epoch,
                                loader=val_loader,
                                device=device,
                                unpacker=unpacker)

        self.call("after_all_epochs")


    def epoch_pass(self, 
                   which: str,
                   epoch: int,
                   loader: DataLoader, 
                   device: str | int, 
                   unpacker: Callable):

        self.call(f"before_{which}_epoch_pass")

        pbar = tqdm(loader)

        batch_pass = getattr(self.train_module, f"{which}_batch_pass")
        for batch_idx, data in enumerate(pbar):
            data = unpacker(data, device)

            self.call(f"before_{which}_batch_pass")
            batch_pass(*data)
            self.call(f"after_{which}_batch_pass")

            pbar.set_postfix(
                None, 
                True,
                EPOCH=epoch,
                **self.train_module.logger._avg_epoch_history
            )

        self.call(f"after_{which}_epoch_pass")


    def call(self, where_at):
        for callback in self.train_module.callbacks():
            if hasattr(callback, where_at):
                method = getattr(callback, where_at)
                method(self)
