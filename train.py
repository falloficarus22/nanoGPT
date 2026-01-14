from model import LanguageModel
from hyperparameters import *
import torch

n_embd = 64

with open('input.txt', 'r', encoding = 'utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

print(f"Vocab size is {vocab_size}")
print("Sample Text", text[:500])
print(f"Length of dataset is {len(text)}")

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s):
    return [stoi[c] for c in s]

def decode(l):
    return ''.join(itos[i] for i in l)

data = torch.tensor(encode(text), dtype = torch.long)

# Train-Val split
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]

# Batch sampling function
def get_batch(split):
    data_split = train_data if split == "train" else val_data
    ix = torch.randint(len(data_split) - block_size, (batch_size, ))

    x = torch.stack([data_split[i: i+block_size] for i in ix])
    y = torch.stack([data_split[i+1: i+block_size+1] for i in ix])

    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()

    for split in ["train", 'val']:
        losses = torch.zeros(50)

        for k in range(50):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()

    return out

model = LanguageModel(
    vocab_size = vocab_size,
    n_embd = n_embd,
    block_size = block_size
    ).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-3)

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses["train"]:.4f}, val loss {losses["val"]:.4f}")

    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none = True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1,), dtype = torch.long)
print(decode(model.generate(context, max_new_tokens = 500)[0].tolist()))