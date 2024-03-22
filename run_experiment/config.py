import os
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from src.transformer.layers import Transformer
from src.data.dataset import TextFolderWithBertTokenizer
from src.data.utils import transformer_unpacker

num_epochs = 10

device = "cuda:0"

train_dataset = TextFolderWithBertTokenizer(
    os.path.join(os.getcwd(), "data"), 
    max_length=256
)
train_loader = DataLoader(train_dataset, batch_size=32)
train_unpacker =transformer_unpacker
train_loss_fn = CrossEntropyLoss()

# validation_dataset = TextFolderWithBertTokenizer(
#     os.path.join(os.getcwd(), "data"), 
#     max_length=256
# )
# validation_loader = DataLoader(validation_dataset, batch_size=32)
# validation_unpacker =transformer_unpacker
# validation_loss_fn = CrossEntropyLoss()

validation_dataset = None
validation_loader = None
validation_unpacker = None
validation_loss_fn = None

model = Transformer(
    vocab_size=train_dataset.tokenizer.vocab_size, 
    num_classes=5, 
    max_length=256, 
    embed_size=64,
    num_layers=5, 
    forward_expansion=4,
    heads=4,
)

optimizer = Adam(model.parameters(), lr=.0001)

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
