from typing import Callable, Unpack
from tqdm import tqdm
from math import trunc
from torch import no_grad
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader


def experiment(which: str,
               epoch: int,
               model: Module,
               loader: DataLoader,
               unpacker: Callable,
               device: str,
               loss_fn: Module,
               optimizer: Optimizer | None = None):
    pass


def epoch_pass(which: str,
               epoch: int,
               model: Module,
               loader: DataLoader,
               unpacker: Callable,
               device: str,
               loss_fn: Module,
               optimizer: Optimizer | None = None):
    assert which in ["train", "validation"]

    running_loss = 0.0
    pbar = tqdm(loader)
    for batch_id, loader_item in pbar:
        inputs, targets = unpacker(loader_item, device)

        if which == "train":
            assert optimizer is not None
            loss = train_batch_pass(model, inputs, targets, optimizer, loss_fn)
            running_loss += loss.item()

        elif which == "validation":
            loss = validation_batch_pass(model, inputs, targets, loss_fn)
            running_loss += loss.item()

        avg_loss = trunc(running_loss / (batch_id + 1) * 100) / 100
        display = { f"EPOCH_{epoch}_AVG_{which}_LOSS": avg_loss }
        pbar.set_postfix(**display)


def train_batch_pass(model: Module,
                     inputs: Tensor | list[Tensor],
                     targets: Tensor, 
                     optimizer: Optimizer, 
                     loss_fn: Module):
    optimizer.zero_grad()
    outputs = model(inputs) if isinstance(inputs, Tensor) else model(*inputs)
    loss: Tensor = loss_fn(outputs, targets)
    loss.backward()
    optimizer.step()
    return loss


def validation_batch_pass(model: Module,
                          inputs: Tensor | list[Tensor],
                          targets: Tensor, 
                          loss_fn: Module):
    with no_grad():
        outputs = model(inputs) if isinstance(inputs, Tensor) else model(*inputs)
    loss: Tensor = loss_fn(outputs, targets)
    return loss


def default_unpacker(loader_item: tuple, device):
    inputs, class_id = loader_item
    return inputs.to(device), class_id.to(device)


def transformer_unpacker(loader_item: tuple, device):
    inputs, masks, class_id = loader_item
    return [inputs.to(device), masks.to(device)], class_id.to(device)
