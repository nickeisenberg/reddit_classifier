import os
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from src.transformer.layers import Transformer
from src.data.dataset import TextFolderWithBertTokenizer
from src.data.utils import transformer_unpacker
from src.metrics.conf_mat import ConfusionMatrix

num_epochs = 5

num_classes = 5

device = "cuda:0"

max_length=256

train_dataset = TextFolderWithBertTokenizer(
    "data",
    "train",
    max_length=max_length
)
train_metric = ConfusionMatrix(
    [x[1] for x in sorted(train_dataset.id_to_label.items(), key = lambda x: x[0])]
).run
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
train_unpacker =transformer_unpacker
train_loss_fn = CrossEntropyLoss()

validation_dataset = TextFolderWithBertTokenizer(
    os.path.join("data"),
    "val",
    label_id_map=train_dataset.label_to_id,
    max_length=max_length
)
validation_metric = ConfusionMatrix(
    [x[1] for x in sorted(validation_dataset.id_to_label.items(), key = lambda x: x[0])]
).run
validation_loader = DataLoader(validation_dataset, batch_size=128, shuffle=True)
validation_unpacker =transformer_unpacker
validation_loss_fn = CrossEntropyLoss()

model = Transformer(
    vocab_size=train_dataset.tokenizer.vocab_size, 
    num_classes=num_classes, 
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
    "train_metric": train_metric,
    "optimizer": optimizer,
    "validation_loader": validation_loader,
    "validation_unpacker": validation_unpacker,
    "validation_loss_fn": validation_loss_fn,
    "validation_metric": validation_metric
}
