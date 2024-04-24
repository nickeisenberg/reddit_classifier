import os
import json
import re
import string
import math
from typing import cast
from transformers import BertTokenizer

import torch
import torch.nn as nn
from torch import Tensor, no_grad, load, argmax
from torch.nn import Module


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
        
        with open("label_to_id.json", "r") as read_json:
            self.id_to_label = json.load(read_json)
        self.id_to_label = {v: k for k, v in self.id_to_label.items()}

        sd = load("validation_ckp.pth", map_location="cpu")
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


def lower_text_and_remove_all_non_asci(text):
    text = text.lower().strip()
    text = re.sub(f"[{re.escape(string.punctuation)}]", '', text)
    text = re.sub("\n", " ", text)
    text = re.sub(r'[^\x00-\x7F]', '', text)
    text = re.sub(r"\w*emote\w*", "", text)
    text = re.sub(r'\b\w{31,}\b', "", text).strip()
    return text


class Transformer(nn.Module):
    def __init__(self, vocab_size, num_classes, max_length, embed_size, 
                 num_layers=6, forward_expansion=4, heads=8, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.position_encoding = PositionalEncoding(embed_size, max_length)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(embed_size, heads, dropout, forward_expansion)
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(embed_size, num_classes)
        
    def forward(self, x, x_mask):
        x_embedding = self.dropout(self.position_encoding(self.embedding(x)))
        
        for layer in self.layers:
            x_embedding = layer(x_embedding, x_mask)
        
        out = x_embedding.mean(dim=1)
        out = self.fc_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super().__init__()
        self.attention = MultiHeadAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x_embedded, mask):
        attention = self.attention(x_embedded, mask)

        x = self.dropout(self.norm1(attention + x_embedded))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, embed_size)
        for pos in range(max_len):
            for i in range(0, embed_size, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / embed_size)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / embed_size)))
        self.pe = pe.unsqueeze(0)

    def forward(self, x: torch.Tensor):
        device = x.device.type
        x = x + self.pe[:, :x.size(1)].detach().to(device)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super().__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert self.head_dim * heads == embed_size

        self.values = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.keys = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.queries = nn.Linear(self.embed_size, self.embed_size, bias=False)

        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, x_embedded: torch.Tensor, mask: torch.Tensor):
        N = x_embedded.shape[0]
        seq_len = x_embedded.shape[1]

        queries = self.queries(x_embedded)
        keys = self.keys(x_embedded)
        values = self.values(x_embedded)

        queries = queries.reshape(N, seq_len, self.heads, self.head_dim)
        keys = keys.reshape(N, seq_len, self.heads, self.head_dim)
        values = values.reshape(N, seq_len, self.heads, self.head_dim)
        
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        energy = energy.masked_fill(
            mask.unsqueeze(1).unsqueeze(1) == 0, float("-1e20")
        )

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum(
            "nhql,nlhd->nqhd", [attention, values]
        ).reshape(N, seq_len, self.heads * self.head_dim)

        return self.fc_out(out)
