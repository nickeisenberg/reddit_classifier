import os
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from src.transformer.layers import Transformer
from src.data.dataset import TextFolderWithBertTokenizer
from src.data.utils import transformer_unpacker

num_epochs = 1

device = "cuda:0"

max_length=256

train_dataset = TextFolderWithBertTokenizer(
    os.path.join("data"),
    "train",
    max_length=max_length
)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
train_unpacker =transformer_unpacker
train_loss_fn = CrossEntropyLoss()

validation_dataset = TextFolderWithBertTokenizer(
    os.path.join("data"),
    "val",
    max_length=max_length
)
validation_loader = DataLoader(validation_dataset, batch_size=128, shuffle=True)
validation_unpacker =transformer_unpacker
validation_loss_fn = CrossEntropyLoss()

model = Transformer(
    vocab_size=train_dataset.tokenizer.vocab_size, 
    num_classes=3, 
    max_length=max_length, 
    embed_size=64,
    num_layers=5, 
    forward_expansion=4,
    heads=4,
)

optimizer = Adam(model.parameters(), lr=.001)

config = {
    "num_epochs": num_epochs,
    "model": model,
    "device": device,
    "train_loader": train_loader,
    "train_unpacker": train_unpacker,
    "train_loss_fn": train_loss_fn,
    "optimizer": optimizer,
    "validation_loader": validation_loader,
    "validation_unpacker": validation_unpacker,
    "validation_loss_fn": validation_loss_fn
}
