import os
from sys import path
path.append(os.getcwd())

from src.trainer import Trainer 
from config import config_trainer 

config = config_trainer(load_checkpoint=True)

if __name__ == "__main__":
    Trainer(config["train_module"]).fit(
        train_loader=config["train_loader"],
        num_epochs=config["num_epochs"],
        device=config["device"],
        unpacker=config["unpacker"],
        val_loader=config["val_loader"],
    )
