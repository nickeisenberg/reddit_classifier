from torchvision.datasets import ImageFolder
from transformers import BertTokenizer
import os

comments = os.listdir("data/wallstreetbets")

with open(os.path.join("data", "wallstreetbets", comments[0]), "r") as af:
    text = af.readline()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

tokenizer.encode_plus(
    text, add_special_tokens=True, padding='max_length', max_length=256, truncation=True
)

