import os
from torch import save
from torch.nn import Module, DataParallel

from .base import Callback


class SaveBestCheckoint(Callback):
    def __init__(self, key):
        self.key = key

        self.best_train_val = 1e6
        self.train_check = lambda cur, prev: cur < prev

        self.best_validation_val = 1e6
        self.validation_check = lambda cur, prev: cur < prev


    def before_all_epochs(self, trainer: Module):
        assert hasattr(trainer, "train_module")

        assert hasattr(trainer.train_module, "logger")
        assert hasattr(trainer.train_module.logger, "train_history")
        assert hasattr(trainer.train_module.logger, "validation_history")
        assert self.key in trainer.train_module.logger.train_history
        assert self.key in trainer.train_module.logger.validation_history

        assert hasattr(trainer.train_module, "model")
        assert hasattr(trainer.train_module, "optimizer")
        assert hasattr(trainer.train_module, "state_dict_root")

        assert hasattr(trainer.train_module, "which_pass")
        assert hasattr(trainer.train_module, "current_epoch")


    def after_train_epoch_pass(self, trainer: Module):
        if self.save_checkpoint_flag(trainer, "train"):
            self.save_checkpoint(trainer, "train")


    def after_validation_epoch_pass(self, trainer: Module):
        if self.save_checkpoint_flag(trainer, "val"):
            self.save_checkpoint(trainer, "val")

    
    def save_checkpoint_flag(self, trainer, which):
        save_ckp = False
    
        if which == "train":
            value = trainer.train_module.logger.train_history[self.key][-1]
            if self.train_check(value, self.best_train_val):
                save_ckp = True
                self.best_train_val = value
    
        elif which == "val":
            value = trainer.train_module.logger.train_history[self.key][-1]
            if self.validation_check(value, self.best_validation_val):
                save_ckp = True
                self.best_validation_val = value
        
        return save_ckp


    def save_checkpoint(self, trainer, which, *args, **kwargs):
        model = trainer.train_module.model
        optimizer = trainer.train_module.optimizer
        state_dict_root = trainer.train_module.state_dict_root
        current_epoch = trainer.train_module.current_epoch
    
        checkpoint = {}
    
        save_to = os.path.join(
            state_dict_root, f"{which}_ckp.pth"
        )
    
        if isinstance(model, DataParallel):
            checkpoint["MODEL_STATE"] = model.module.state_dict()
            checkpoint["OPTIMIZER_STATE"] = optimizer.state_dict()
            checkpoint["EPOCHS_RUN"] = current_epoch 
        else:
            checkpoint["MODEL_STATE"] = model.state_dict()
            checkpoint["OPTIMIZER_STATE"] = optimizer.state_dict()
            checkpoint["EPOCHS_RUN"] = current_epoch 
    
        save(checkpoint, save_to)
        print(f"EPOCH {current_epoch} checkpoint saved at {save_to}")

