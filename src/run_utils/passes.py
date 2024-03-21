from typing import Callable
import torch
import torch.nn as nn

def train_batch_pass(model: Callable,
                     inputs: torch.Tensor,
                     masks: torch.Tensor,
                     targets: torch.Tensor, 
                     optimizer, 
                     loss_fn):
    pass

def validation_batch_pass():
    pass

def test_batch_pass():
    pass

