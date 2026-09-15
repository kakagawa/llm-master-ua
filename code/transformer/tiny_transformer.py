"""Tiny Transformer: повна модель з нуля

Код зі статті: docs/llm/transformer/tiny_transformer_code.md
Файл згенеровано скриптом tools/extract-code.mjs — правити треба статтю, не цей файл.

Запуск:  python3 code/transformer/tiny_transformer.py
Потрібен лише numpy.
"""

# ── Блок 1 ───────────────────────────────────────────────────────────
import numpy as np

np.random.seed(42)

vocab_size = 10
L = 4
d_model = 8

input_ids = np.array([3, 8, 2, 6])
embedding_table = np.random.randn(vocab_size, d_model)

x = embedding_table[input_ids]
print(x.shape)  # (4, 8)

# ── Блок 2 ───────────────────────────────────────────────────────────
def positional_encoding(L, d_model):
    pe = np.zeros((L, d_model))
    for pos in range(L):
        for i in range(0, d_model, 2):
            pe[pos, i] = np.sin(pos / (10000 ** (i / d_model)))
            if i + 1 < d_model:
                pe[pos, i + 1] = np.cos(pos / (10000 ** (i / d_model)))
    return pe

x = x + positional_encoding(L, d_model)
print(x.shape)  # (4, 8)

# ── Блок 3 ───────────────────────────────────────────────────────────
class TinyBlock:
    def __init__(self, d_model, d_ff):
        self.Wq = np.random.randn(d_model, d_model) * 0.01
        self.Wk = np.random.randn(d_model, d_model) * 0.01
        self.Wv = np.random.randn(d_model, d_model) * 0.01
        self.W1 = np.random.randn(d_model, d_ff) * 0.01
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * 0.01
        self.b2 = np.zeros(d_model)

    def layer_norm(self, x, eps=1e-5):
        mean = x.mean(axis=-1, keepdims=True)
        std = x.std(axis=-1, keepdims=True)
        return (x - mean) / (std + eps)

    def attention(self, x):
        Q = x @ self.Wq
        K = x @ self.Wk
        V = x @ self.Wv
        scores = Q @ K.T / np.sqrt(x.shape[-1])
        weights = np.exp(scores) / np.exp(scores).sum(axis=-1, keepdims=True)
        return weights @ V

    def ffn(self, x):
        hidden = np.maximum(0, x @ self.W1 + self.b1)
        return hidden @ self.W2 + self.b2

    def forward(self, x):
        x = self.layer_norm(x + self.attention(x))
        x = self.layer_norm(x + self.ffn(x))
        return x

# ── Блок 4 ───────────────────────────────────────────────────────────
blocks = [TinyBlock(d_model=8, d_ff=32) for _ in range(2)]

for block in blocks:
    x = block.forward(x)

print(x.shape)  # (4, 8)

# ── Блок 5 ───────────────────────────────────────────────────────────
W_out = np.random.randn(d_model, vocab_size) * 0.01
b_out = np.zeros(vocab_size)

logits = x @ W_out + b_out
print(logits.shape)  # (4, 10)

# ── Блок 6 ───────────────────────────────────────────────────────────
import numpy as np

np.random.seed(42)

vocab_size = 10
L = 4
d_model = 8
d_ff = 32
num_layers = 2

input_ids = np.array([3, 8, 2, 6])
embedding_table = np.random.randn(vocab_size, d_model)

x = embedding_table[input_ids]

# positional encoding
def positional_encoding(L, d_model):
    pe = np.zeros((L, d_model))
    for pos in range(L):
        for i in range(0, d_model, 2):
            pe[pos, i] = np.sin(pos / (10000 ** (i / d_model)))
            if i + 1 < d_model:
                pe[pos, i + 1] = np.cos(pos / (10000 ** (i / d_model)))
    return pe

x = x + positional_encoding(L, d_model)

class TinyBlock:
    def __init__(self, d_model, d_ff):
        self.Wq = np.random.randn(d_model, d_model) * 0.01
        self.Wk = np.random.randn(d_model, d_model) * 0.01
        self.Wv = np.random.randn(d_model, d_model) * 0.01
        self.W1 = np.random.randn(d_model, d_ff) * 0.01
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * 0.01
        self.b2 = np.zeros(d_model)

    def layer_norm(self, x, eps=1e-5):
        mean = x.mean(axis=-1, keepdims=True)
        std = x.std(axis=-1, keepdims=True)
        return (x - mean) / (std + eps)

    def attention(self, x):
        Q = x @ self.Wq
        K = x @ self.Wk
        V = x @ self.Wv
        scores = Q @ K.T / np.sqrt(x.shape[-1])
        weights = np.exp(scores) / np.exp(scores).sum(axis=-1, keepdims=True)
        return weights @ V

    def ffn(self, x):
        hidden = np.maximum(0, x @ self.W1 + self.b1)
        return hidden @ self.W2 + self.b2

    def forward(self, x):
        x = self.layer_norm(x + self.attention(x))
        x = self.layer_norm(x + self.ffn(x))
        return x

blocks = [TinyBlock(d_model, d_ff) for _ in range(num_layers)]

for block in blocks:
    x = block.forward(x)

W_out = np.random.randn(d_model, vocab_size) * 0.01
b_out = np.zeros(vocab_size)
logits = x @ W_out + b_out

print("Підсумковий shape logits:", logits.shape)
print("Передбачений токен на кожній позиції:", logits.argmax(axis=-1))
