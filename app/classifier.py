import json

from typing import cast

from transformers import BertTokenizer

from torch import Tensor, no_grad, load, argmax
from torch.nn import Module

from src.data.utils import lower_text_and_remove_all_non_asci
from src.transformer.layers import Transformer


class RedditClassifier(Module):
    def __init__(self):
        super().__init__()

        self.tokenizer: BertTokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = Transformer(
            vocab_size=self.tokenizer.vocab_size,
            num_classes=10, 
            max_length=256, 
            embed_size=256,
            num_layers=6, 
            forward_expansion=4,
            heads=4,
        )
        
        with open("app/label_to_id.json", "r") as read_json:
            self.id_to_label = json.load(read_json)
        self.id_to_label = {v: k for k, v in self.id_to_label.items()}

        sd = load("app/validation_ckp.pth", map_location="cpu")
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

