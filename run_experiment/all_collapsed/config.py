import os
import json
from torch import load

from torch.utils.data import DataLoader

from src.transformer.layers import Transformer
from src.data.dataset import TextFolderWithBertTokenizer
from src.data.utils import transformer_unpacker
from src_trainer.callbacks import (
    Accuracy,
    CSVLogger,
    ConfusionMatrix,
    ProgressBarUpdater,
    SaveBestCheckoint
)

from train_module import TrainModule


def config_datasets():
    max_length=256
    instructions = {
        "Bitcoin": "finance",
        "CryptoCurrency": "finance",
        "wallstreetbets": "finance",
        "chelseafc": "soccer",
        "movies": "pop culture",
        "popheads": "pop culture",
        "hiphopheads": "pop culture",
    }
    train_dataset = TextFolderWithBertTokenizer(
        root_dir="data",
        instructions=instructions,
        which="train",
        max_length=max_length
    )
    validation_dataset = TextFolderWithBertTokenizer(
        root_dir="data",
        which="val",
        instructions=instructions,
        label_id_map=train_dataset.label_to_id,
        max_length=max_length
    )
    evalutaion_dataset = TextFolderWithBertTokenizer(
        root_dir="data",
        which="test",
        instructions=instructions,
        label_id_map=train_dataset.label_to_id,
        max_length=max_length
    )

    save_root = os.path.relpath(__file__)
    save_root = save_root.split(os.path.basename(save_root))[0]
    with open(os.path.join(save_root, "label_to_id.json"), "w") as write_json:
        json.dump(train_dataset.label_to_id, write_json)

    return train_dataset, validation_dataset, evalutaion_dataset


def config_loaders(train_dataset, validation_dataset, evaluation_dataset):
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=128, shuffle=True)
    evaluation_loader = DataLoader(evaluation_dataset, batch_size=128, shuffle=True)
    return train_loader, validation_loader, evaluation_loader


def config_save_roots():
    save_root = os.path.relpath(__file__)
    save_root = save_root.split(os.path.basename(save_root))[0]
    loss_log_root = os.path.join(save_root, "loss_logs")
    state_dict_root = os.path.join(save_root, "state_dicts")
    metrics_root = os.path.join(save_root, "metrics")
    if not os.path.isdir(loss_log_root):
        os.makedirs(loss_log_root)
    if not os.path.isdir(state_dict_root):
        os.makedirs(state_dict_root)
    if not os.path.isdir(metrics_root):
        os.makedirs(metrics_root)
    return state_dict_root, loss_log_root, metrics_root


def config_trainer(load_checkpoint=False):
    tdataset, vdataset, edataset = config_datasets()
    vocab_size = tdataset.tokenizer.vocab_size
    tloader, vloader, eloader = config_loaders(tdataset, vdataset, edataset)

    state_dict_root, loss_log_root, metrics_root = config_save_roots()

    max_length = 256

    model = Transformer(
        vocab_size=vocab_size,
        num_classes=5, 
        max_length=max_length, 
        embed_size=64,
        num_layers=5, 
        forward_expansion=4,
        heads=4,
    )

    device = "cuda:0"

    accuracy = Accuracy()
    conf_mat = ConfusionMatrix(
        labels=[
            x[1] 
            for x in sorted(tdataset.id_to_label.items(), key = lambda x: x[0])
        ],
        save_root=metrics_root
    )
    logger = CSVLogger(loss_log_root)
    save_best = SaveBestCheckoint(
        state_dict_root=state_dict_root,
        train_key="total_loss",
        validation_key="accuracy",
        best_validation_val=-1,
        validation_check = lambda cur, prev: cur > prev,
    )
    pbar_updater = ProgressBarUpdater()


    if load_checkpoint: 
        load_state_dict_from = os.path.join(
            state_dict_root, "validation_ckp.pth"
        )
        if os.path.isfile(load_state_dict_from):
            sd = load(load_state_dict_from, map_location="cpu")
            model.load_state_dict(sd["MODEL_STATE"])
            save_best.best_train_val = sd["BEST_TRAIN"]
            save_best.best_validation_val = sd["BEST_VALIDATION"]
            save_best.starting_epoch = sd["EPOCHS_RUN"]
        else:
            print("No state dict found, training from scratch")

    train_module = TrainModule(
        model=model, 
        device=device,
        accuracy=accuracy,
        conf_mat=conf_mat,
        logger=logger,
        save_best=save_best,
        progress_bar_updater=pbar_updater
    )

    num_epochs = 200
    unpacker =transformer_unpacker
    config = {
        "train_module": train_module,
        "train_loader": tloader,
        "num_epochs": num_epochs,
        "device": train_module.device,
        "unpacker": unpacker,
        "val_loader": vloader,
        "evaluation_loader": eloader,
    }
    return config
