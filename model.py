import torch
import torch.nn as nn
import torch.nn.functional as F

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets = None):
        """
        idx: (B, T)
        targets: (B, T)
        """
        logits = self.token_embedding_table(idx)

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
            logits, _ =  self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim = -1)
            next_idx = torch.multinomial(probs, num_samples = 1)
            idx = torch.cat((idx, next_idx), dim = 1)

        return idx

