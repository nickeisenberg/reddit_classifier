import torch
import torch.nn as nn
import math


class Transformer(nn.Module):
    def __init__(self, vocab_size, embed_size=256, num_layers=6, 
                 forward_expansion=4, heads=8, dropout=0.1, max_length=100, 
                 num_classes=2):
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
        
    def forward(self, x):
        x_mask = (x != 0).unsqueeze(1).unsqueeze(3)

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

    def forward(self, x):
        # x is the input sequence embedded
        x = x + self.pe[:, :x.size(1)].detach()
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

    def forward(self, x_embedded, mask):
        N = x.shape[0]
        seq_len = x.shape[1]

        values = self.values(x_embedded)
        keys = self.keys(x_embedded)
        queries = self.queries(x_embedded)

        queries = queries.reshape(N, seq_len, self.heads, self.head_dim)
        keys = keys.reshape(N, seq_len, self.heads, self.head_dim)
        values = values.reshape(N, seq_len, self.heads, self.head_dim)
        
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum(
            "nhql,nlhd->nqhd", [attention, values]
        ).reshape(N, seq_len, self.heads * self.head_dim)

        return self.fc_out(out)



m = MultiHeadAttention(12, 3)
x = torch.randn((50, 20, 12))
m(x, None).shape


t = Transformer(100, 12, 1, 4, 3)
x = torch.randint(0, 100, (50, 20))
t(x).shape
