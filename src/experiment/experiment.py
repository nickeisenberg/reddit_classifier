import os
import numpy as np
from typing import Callable
from tqdm import tqdm
from math import trunc
from torch import no_grad, save
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader


def experiment(num_epochs: int,
               model: Module,
               save_root: str,
               device: str,
               optimizer: Optimizer,
               train_loader: DataLoader,
               train_unpacker: Callable,
               train_loss_fn: Module,
               train_metric: Callable | None = None,
               validation_loader: DataLoader | None = None,
               validation_unpacker: Callable | None = None,
               validation_loss_fn: Module | None = None,
               validation_metric: Callable | None = None):

    model.to(device)

    best_avg_val_loss = 1e6
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        _, targets, predictions = train_epoch_pass(
            train_batch_pass=train_batch_pass,
            epoch=epoch,
            model=model,
            loader=train_loader,
            unpacker=train_unpacker,
            device=device,
            loss_fn=train_loss_fn,
            optimizer=optimizer
        )

        overfitted_file_name = f"overfitted.pth"
        save(
            model.state_dict(), 
            os.path.join(save_root, overfitted_file_name)
        )
        if train_metric is not None:
            metric_file_name = f"train_metric.png"
            train_metric(
                targets=targets, 
                predictions=predictions, 
                save_to=os.path.join(save_root, metric_file_name)
            )

        model.eval()
        if not validation_loader is None:
            assert validation_unpacker is not None
            assert validation_loss_fn is not None

            avg_val_loss, targets, predictions = validation_epoch_pass(
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
                
                if validation_metric is not None:
                    metric_file_name = f"EPOCH_{epoch}_val_metric.png"
                    validation_metric(
                        targets=targets, 
                        predictions=predictions, 
                        save_to=os.path.join(save_root, metric_file_name)
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
    all_targets, all_predictions = [], []

    pbar = tqdm(loader)
    for batch_id, loader_item in enumerate(pbar):
        inputs, targets = unpacker(loader_item, device)

        loss, predictions = train_batch_pass(
            model, inputs, targets, optimizer, loss_fn
        )
        running_loss += loss.item()

        all_targets += targets
        all_predictions += predictions

        accuracy = (np.array(all_targets) == np.array(all_predictions)).sum()
        accuracy = np.round(accuracy / len(all_predictions) * 100, 2)

        avg_loss = trunc(running_loss / (batch_id + 1) * 100) / 100
        display = {
            f"EPOCH_{epoch}_AVG_TRAIN_LOSS": avg_loss,
            f"EPOCH_{epoch}_TRAIN_ACCURACY": accuracy,
        }
        pbar.set_postfix(ordered_dict=None, refresh=True, **display)

    return running_loss / len(loader), all_targets, all_predictions


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
    return loss, outputs.detach().tolist()


def validation_epoch_pass(validation_batch_pass: Callable,
                          epoch: int,
                          model: Module,
                          loader: DataLoader,
                          unpacker: Callable,
                          device: str,
                          loss_fn: Module):

    running_loss = 0.0
    all_targets, all_predictions = [], []

    pbar = tqdm(loader)
    for batch_id, loader_item in enumerate(pbar):
        inputs, targets = unpacker(loader_item, device)

        loss, predictions = validation_batch_pass(
            model, inputs, targets, loss_fn
        )
        running_loss += loss.item()

        all_targets += targets
        all_predictions += predictions
        
        accuracy = (np.array(all_targets) == np.array(all_predictions)).sum()
        accuracy = np.round(accuracy / len(all_predictions) * 100, 2)

        avg_loss = trunc(running_loss / (batch_id + 1) * 100) / 100
        display = {
            f"EPOCH_{epoch}_AVG_VAL_LOSS": avg_loss,
            f"EPOCH_{epoch}_VAL_ACCURACY": accuracy,
        }
        pbar.set_postfix(ordered_dict=None, refresh=True, **display)

    return running_loss / len(loader), all_targets, all_predictions 


def validation_batch_pass(model: Module,
                          inputs: Tensor | list[Tensor],
                          targets: Tensor, 
                          loss_fn: Module):
    with no_grad():
        outputs: Tensor = model(inputs) if isinstance(inputs, Tensor) else model(*inputs)
    loss: Tensor = loss_fn(outputs, targets)
    return loss, outputs.detach().tolist()
