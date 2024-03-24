import os
from typing import Callable
from tqdm import tqdm
from math import trunc
from torch import no_grad, save
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from ..metrics.conf_mat import confusion_matrix


def experiment(num_epochs: int,
               model: Module,
               save_root: str,
               device: str,
               train_loader: DataLoader,
               train_unpacker: Callable,
               train_loss_fn: Module,
               optimizer: Optimizer,
               validation_loader: DataLoader | None = None,
               validation_unpacker: Callable | None = None,
               validation_loss_fn: Module | None = None):

    model.to(device)

    best_avg_val_loss = 1e6
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        _ = train_epoch_pass(train_batch_pass=train_batch_pass,
                             epoch=epoch,
                             model=model,
                             loader=train_loader,
                             unpacker=train_unpacker,
                             device=device,
                             loss_fn=train_loss_fn,
                             optimizer=optimizer)

        overfitted_file_name = f"overfitted.pth"
        save(
            model.state_dict(), 
            os.path.join(save_root, overfitted_file_name)
        )

        model.eval()
        if not validation_loader is None:
            assert validation_unpacker is not None
            assert validation_loss_fn is not None

            avg_val_loss = validation_epoch_pass(
                validation_batch_pass=validation_batch_pass,
                epoch=epoch,
                model=model,
                loader=validation_loader,
                unpacker=validation_unpacker,
                device=device,
                loss_fn=validation_loss_fn
            )

            if avg_val_loss < best_avg_val_loss:
                best_avg_val_loss = avg_val_loss
                best_file_name = f"val_ep{epoch}.pth"
                save(
                    model.state_dict(), 
                    os.path.join(save_root, best_file_name)
                )


def train_epoch_pass(train_batch_pass: Callable,
                     epoch: int,
                     model: Module,
                     loader: DataLoader,
                     unpacker: Callable,
                     device: str,
                     loss_fn: Module,
                     optimizer: Optimizer | None = None):

    running_loss = 0.0
    pbar = tqdm(loader)
    for batch_id, loader_item in enumerate(pbar):
        inputs, targets = unpacker(loader_item, device)

        loss = train_batch_pass(model, inputs, targets, optimizer, loss_fn)
        running_loss += loss.item()

        avg_loss = trunc(running_loss / (batch_id + 1) * 100) / 100
        display = { f"EPOCH_{epoch}_AVG_TRAIN_LOSS": avg_loss }
        pbar.set_postfix(ordered_dict=None, refresh=True, **display)

    return running_loss / len(loader) 


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


def validation_epoch_pass(validation_batch_pass: Callable,
                          epoch: int,
                          model: Module,
                          loader: DataLoader,
                          unpacker: Callable,
                          device: str,
                          loss_fn: Module,
                          metrics: list[Callable]):

    running_loss = 0.0
    pbar = tqdm(loader)
    all_targets, all_predictions = [], []
    for batch_id, loader_item in enumerate(pbar):
        inputs, targets = unpacker(loader_item, device)

        loss, predictions = validation_batch_pass(
            model, inputs, targets, loss_fn
        )

        all_targets.append(targets)
        all_predictions.append(predictions)

        running_loss += loss.item()

        avg_loss = trunc(running_loss / (batch_id + 1) * 100) / 100
        display = { f"EPOCH_{epoch}_AVG_VAL_LOSS": avg_loss }
        pbar.set_postfix(ordered_dict=None, refresh=True, **display)
    
    for metric in metrics:
        metric(all_targets, all_predictions)

    return running_loss / len(loader) 


def validation_batch_pass(model: Module,
                          inputs: Tensor | list[Tensor],
                          targets: Tensor, 
                          loss_fn: Module):
    with no_grad():
        outputs: Tensor = model(inputs) if isinstance(inputs, Tensor) else model(*inputs)
    loss: Tensor = loss_fn(outputs, targets)
    return loss, outputs.detach().tolist()




