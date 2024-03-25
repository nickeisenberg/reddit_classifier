import os
from sys import path
path.append(os.getcwd())

from src.experiment.experiment import experiment
from config import config

if __name__ == "__main__":
    experiment(save_root=os.path.split(__file__)[0], **config)
