import os
import json
from typing import cast

import torch.nn as nn
from torch import Tensor, no_grad, load, argmax

from flask import Flask, request, render_template
from transformers import BertTokenizer

from sys import path
path.append(__file__.split("app")[0])

from src.transformer.layers import Transformer
from src.data.utils import lower_text_and_remove_all_non_asci


class RedditClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.tokenizer: BertTokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = Transformer(
            vocab_size=self.tokenizer.vocab_size,
            num_classes=10, 
            max_length=256, 
            embed_size=512,
            num_layers=4, 
            forward_expansion=4,
            heads=8,
        )
        
        app_root = os.path.relpath(__file__)
        app_root = app_root.split(os.path.basename(app_root))[0]
        with open(os.path.join(app_root, "label_to_id.json"), "r") as read_json:
            self.id_to_label = json.load(read_json)
        self.id_to_label = {v: k for k, v in self.id_to_label.items()}

        sd = load(os.path.join(app_root, "validation_ckp.pth"), map_location="cpu")
        self.model.load_state_dict(sd["MODEL_STATE"])


    def forward(self, text):
        self.eval()
        input_ids, input_masks = self._tokenize_text(
            lower_text_and_remove_all_non_asci(text)
        )
        with no_grad():
            prediction = argmax(self.model(input_ids, input_masks)[0])
        return self.id_to_label[int(prediction.item())]


    def _tokenize_text(self, text) -> tuple[Tensor, Tensor]:
        input = self.tokenizer.encode_plus(
            text, 
            add_special_tokens=True, 
            max_length=256, 
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        input_ids = cast(Tensor, input["input_ids"])
        input_masks = cast(Tensor, input["attention_mask"])
        return input_ids, input_masks
    

reddit_classifier = RedditClassifier()

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        text = request.form['text']
        classification = reddit_classifier(text)
        return render_template(
            'index.html', text=text, classification=classification
        )
    return render_template('index.html', text="", classification="")

if __name__ == '__main__':
    app.run(debug=True)
