from transformers import BertTokenizer

with open("data/wallstreetbets.txt", "r") as af:
    wsb = af.readlines()


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

encoded_inputs = tokenizer(
    wsb, padding=True, truncation=True, max_length=512, return_tensors="pt"
)

enc = encoded_inputs["input_ids"][0]

tokenizer.decode(enc)

tokenizer.convert_ids_to_tokens([3681, 13334, 2102, 20915, 2015])

tokenizer.decode(tokenizer.encode("wallstreetbets"))
