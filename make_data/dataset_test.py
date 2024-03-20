from torch.utils.data import DataLoader
from src.data.dataset import TextFolderDatasetWithBertTokenizer

dataset = TextFolderDatasetWithBertTokenizer("data", 256)

dataloader = DataLoader(dataset, 10)

batch = next(iter(dataloader))

batch[0].shape
batch[1].shape
batch[2].shape
