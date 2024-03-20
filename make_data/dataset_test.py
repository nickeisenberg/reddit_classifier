from torch.utils.data import DataLoader
from src.data.dataset import TextFolderDatasetWithBertTokenizer

dataset = TextFolderDatasetWithBertTokenizer("data", max_length=256)
dataloader = DataLoader(dataset, 10)

inputs, masks, labels = next(iter(dataloader))
