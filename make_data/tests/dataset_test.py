from torch.utils.data import DataLoader
from src.data.dataset import TextFolderWithBertTokenizer

dataset = TextFolderWithBertTokenizer(
    "data", 
    "train", 
    max_length=256
)

dataset.tokenizer.decode(dataset[6999][0])

dataloader = DataLoader(dataset, 10)

inputs, masks, labels = next(iter(dataloader))
