import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionHead(nn.Module):
    """Single attention head"""

    def __init__(self, n_embd, block_size):
        super().__init__()
        self.key = nn.Linear(n_embd, n_embd, bias = False)
        self.query = nn.Linear(n_embd, n_embd, bias = False)
        self.value = nn.Linear(n_embd, n_embd, bias = False)

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


class LanguageModel(nn.Module):

    def __init__(self, vocab_size, n_embd, block_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.block_size = block_size
        self.attn_head = AttentionHead(n_embd, block_size)

    def forward(self, idx, targets = None):
        B, T = idx.shape
        tok_embd = self.token_embedding_table(idx)
        pos_embd = self.position_embedding_table(torch.arange(T))
        x = tok_embd + pos_embd
        x = self.attn_head(x)
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
