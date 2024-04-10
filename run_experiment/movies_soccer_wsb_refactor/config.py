import os
from torch.utils.data import DataLoader
from src.transformer.layers import Transformer
from src.data.dataset import TextFolderWithBertTokenizer
from src.data.utils import transformer_unpacker
from src.trainer_module.train_module import TrainModule


def config_datasets():
    max_length=256
    instructions = {
        "CryptoCurrency": "ignore",
        "formula1": "ignore"
    }
    train_dataset = TextFolderWithBertTokenizer(
        root_dir="data",
        which="train",
        instructions=instructions,
        max_length=max_length
    )
    validation_dataset = TextFolderWithBertTokenizer(
        root_dir="data",
        which="val",
        instructions=instructions,
        label_id_map=train_dataset.label_to_id,
        max_length=max_length
    )
    return train_dataset, validation_dataset


def config_loaders(train_dataset, validation_dataset):
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=128, shuffle=True)
    return train_loader, validation_loader


def config_save_roots():
    save_root = os.path.relpath(__file__)
    save_root = save_root.split(os.path.basename(save_root))[0]
    loss_log_root = os.path.join(save_root, "loss_logs")
    state_dict_root = os.path.join(save_root, "state_dicts")
    if not os.path.isdir(loss_log_root):
        os.makedirs(loss_log_root)
    if not os.path.isdir(state_dict_root):
        os.makedirs(state_dict_root)
    return state_dict_root, loss_log_root


def config_train_module(vocab_size, max_length):
    state_dict_root, loss_log_root = config_save_roots()
    device = "cuda:0"
    model = Transformer(
        vocab_size=vocab_size,
        num_classes=3, 
        max_length=max_length, 
        embed_size=64,
        num_layers=5, 
        forward_expansion=4,
        heads=4,
    )
    train_module = TrainModule(
        model=model.to(device), 
        device=device, 
        state_dict_root=state_dict_root, 
        loss_log_root=loss_log_root
    )
    return train_module


def config_trainer():
    tdataset, vdataset = config_datasets()
    vocab_size = tdataset.tokenizer.vocab_size
    tloader, vloader = config_loaders(tdataset, vdataset)

    train_module = config_train_module(
        vocab_size=vocab_size,
        max_length=256,
    )

    num_epochs = 5
    unpacker =transformer_unpacker
    config = {
        "train_module": train_module,
        "train_loader": tloader,
        "num_epochs": num_epochs,
        "device": train_module.device,
        "unpacker": unpacker,
        "val_loader": vloader,
    }
    return config
