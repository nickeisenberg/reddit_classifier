from transformers import BertTokenizer
import os

wsb_path = "data/wallstreetbets/train"
comments = os.listdir(wsb_path)

with open(os.path.join(wsb_path, comments[1]), "r") as af:
    text = af.readline()


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

encoded_text = tokenizer.encode_plus(
    text, add_special_tokens=True, padding='max_length', max_length=256, truncation=True
)

tokenizer.decode(encoded_text["input_ids"])
