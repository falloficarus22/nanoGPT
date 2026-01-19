import torch
import torch.nn as nn
import torch.nn.functional as F
from hyperparameters import *

class AttentionHead(nn.Module):
    """Single attention head"""

    def __init__(self, head_size, n_embd, block_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias = False)
        self.query = nn.Linear(n_embd, head_size, bias = False)
        self.value = nn.Linear(n_embd, head_size, bias = False)

        self.register_buffer(
            'tril', torch.tril(torch.ones(block_size, block_size))
        )

    def forward(self, x):
        B, T, C = x.shape

        K = self.key(x)
        Q = self.query(x)
        V = self.value(x)

        attn = (Q @ K.transpose(-1, -2)) / (C ** 0.5)
        attn = attn.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        weights = F.softmax(attn, dim=-1)
        out = weights @ V

        return out

class MultiHeadAttention(nn.Module):
    """Multiple heads of attention in parallel."""

    def __init__(self, n_head, n_embd, block_size):
        super().__init__()
        head_size = n_embd // n_head
        self.heads = nn.ModuleList([
            AttentionHead(head_size, n_embd, block_size) for _ in range(n_head)
        ])
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim = -1)
        out = self.proj(out)

        return out

class FeedForwardNetwork(nn.Module):
    """A simple MLP"""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd)
        )
    
    def forward(self, x):
        return self.net(x)

class Transformer(nn.Module):
    """Transformer Block"""

    def __init__(self, n_head, n_embd, block_size):
        super().__init__()
        self.self_attn = MultiHeadAttention(n_head, n_embd, block_size)
        self.ffn = FeedForwardNetwork(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # 1. Attention + residual
        x = x + self.self_attn(self.ln1(x))

        # 2. Feedforward + residual
        x = x + self.ffn(self.ln2(x))

        return x

class LanguageModel(nn.Module):

    def __init__(self, vocab_size, n_embd, block_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.block_size = block_size

        self.transformer_block = nn.Sequential(
            *[Transformer(n_head = 4, n_embd = n_embd, block_size = block_size) for _ in range(n_layers)]
        )

        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets = None):
        B, T = idx.shape
        tok_embd = self.token_embedding_table(idx)
        pos_embd = self.position_embedding_table(torch.arange(T))
        x = tok_embd + pos_embd
        x = self.transformer_block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ =  self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim = -1)
            next_idx = torch.multinomial(probs, num_samples = 1)
            idx = torch.cat((idx, next_idx), dim = 1)

        return idx
