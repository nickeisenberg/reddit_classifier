import os
from sys import path
path.append(os.getcwd())

from src.trainer.trainer import Trainer 
from config import config_trainer 

config = config_trainer(False)

if __name__ == "__main__":
    Trainer(config["train_module"]).evaluate(
        loader=config["evaluation_loader"],
        device=config["device"],
        unpacker=config["unpacker"],
    )
