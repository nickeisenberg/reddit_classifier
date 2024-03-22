from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from src.transformer.layers import Transformer
from src.data.dataset import TextFolderWithBertTokenizer
from src.data.utils import transformer_unpacker

num_epochs = 1

model = Transformer(
    vocab_size=10, 
    num_classes=5, 
    max_length=10, 
    embed_size=10,
    num_layers=2, 
    forward_expansion=4,
    heads=4,
)

optimizer = Adam(model.parameters(), lr=.0001)

device = "cuda:0"

train_dataset = TextFolderWithBertTokenizer("")
train_loader = DataLoader(train_dataset, batch_size=32)
train_unpacker =transformer_unpacker
train_loss_fn = CrossEntropyLoss()


validation_dataset = TextFolderWithBertTokenizer("")
validation_loader = DataLoader(validation_dataset, batch_size=32)
validation_unpacker =transformer_unpacker
validation_loss_fn = CrossEntropyLoss()

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
