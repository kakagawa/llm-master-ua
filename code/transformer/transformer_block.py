"""Transformer Block з нуля

Код зі статті: docs/llm/transformer/transformer_block_code.md
Файл згенеровано скриптом tools/extract-code.mjs — правити треба статтю, не цей файл.

Запуск:  python3 code/transformer/transformer_block.py
Потрібен лише numpy.
"""

# ── Блок 1 ───────────────────────────────────────────────────────────
import numpy as np

class LayerNorm:
    def __init__(self, d_model, eps=1e-5):
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)
        self.eps = eps

    def forward(self, x):
        mean = x.mean(axis=-1, keepdims=True)
        std = x.std(axis=-1, keepdims=True)
        x_norm = (x - mean) / (std + self.eps)
        return self.gamma * x_norm + self.beta

# ── Блок 2 ───────────────────────────────────────────────────────────
class FeedForward:
    def __init__(self, d_model, d_ff):
        self.W1 = np.random.randn(d_model, d_ff) * 0.01
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * 0.01
        self.b2 = np.zeros(d_model)

    def gelu(self, x):
        return 0.5 * x * (1 + np.tanh(
            np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)
        ))

    def forward(self, x):
        hidden = x @ self.W1 + self.b1
        hidden = self.gelu(hidden)
        output = hidden @ self.W2 + self.b2
        return output

# ── Блок 3 ───────────────────────────────────────────────────────────
class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_Q = np.random.randn(d_model, d_model) * 0.01
        self.W_K = np.random.randn(d_model, d_model) * 0.01
        self.W_V = np.random.randn(d_model, d_model) * 0.01
        self.W_O = np.random.randn(d_model, d_model) * 0.01

    def softmax(self, x):
        e = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)

    def split_heads(self, x):
        L, d_model = x.shape
        return x.reshape(L, self.num_heads, self.d_k).transpose(1, 0, 2)

    def forward(self, x):
        L, d_model = x.shape

        Q = x @ self.W_Q
        K = x @ self.W_K
        V = x @ self.W_V

        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        head_outputs = []

        for i in range(self.num_heads):
            scores = Q[i] @ K[i].T / np.sqrt(self.d_k)
            weights = self.softmax(scores)
            out = weights @ V[i]
            head_outputs.append(out)

        concat = np.stack(head_outputs, axis=0)
        concat = concat.transpose(1, 0, 2).reshape(L, d_model)

        output = concat @ self.W_O
        return output

# ── Блок 4 ───────────────────────────────────────────────────────────
class TransformerBlock:
    def __init__(self, d_model, num_heads, d_ff):
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

    def forward(self, x):
        print(f"Вхід x:                {x.shape}")

        attn_out = self.attn.forward(x)
        print(f"Вихід Attention:       {attn_out.shape}")

        x = self.norm1.forward(x + attn_out)
        print(f"Після Add & Norm:      {x.shape}")

        ffn_out = self.ffn.forward(x)
        print(f"Вихід FFN:             {ffn_out.shape}")

        x = self.norm2.forward(x + ffn_out)
        print(f"Підсумковий вихід:     {x.shape}")

        return x

# ── Блок 5 ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(42)

    L = 7
    d_model = 8
    num_heads = 2
    d_ff = 32

    x = np.random.randn(L, d_model)

    block = TransformerBlock(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff
    )

    output = block.forward(x)

    print("\n=== Перевірка результату ===")
    print("Форма входу:", x.shape)
    print("Форма виходу:", output.shape)
    print("Форми збігаються:", x.shape == output.shape)
