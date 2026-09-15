"""Multi-Head Attention з нуля

Код зі статті: docs/llm/transformer/mha_code.md
Файл згенеровано скриптом tools/extract-code.mjs — правити треба статтю, не цей файл.

Запуск:  python3 code/transformer/multi_head_attention.py
Потрібен лише numpy.
"""

# ── Блок 1 ───────────────────────────────────────────────────────────
import numpy as np

# Налаштування гіперпараметрів
L = 7        # довжина послідовності («Удалині росте яблуня», 7 токенів)
d_model = 8  # розмірність embedding (насправді 512, тут 8 для наочності)
h = 2        # кількість голів уваги (зазвичай 8 або 16)
d_k = d_model // h  # розмірність однієї голови = 8 // 2 = 4

np.random.seed(42)

# Імітуємо вхід після embedding
X = np.random.randn(L, d_model)
print(f"Вхід X: {X.shape}")  # (7, 8)

# Три матриці ваг (у реальності це навчані параметри)
W_Q = np.random.randn(d_model, d_model)
W_K = np.random.randn(d_model, d_model)
W_V = np.random.randn(d_model, d_model)

# Лінійне відображення, отримуємо Q, K, V
Q = X @ W_Q   # (7, 8) @ (8, 8) = (7, 8)
K = X @ W_K
V = X @ W_V

print(f"Q: {Q.shape}")  # (7, 8)
print(f"K: {K.shape}")  # (7, 8)
print(f"V: {V.shape}")  # (7, 8)

# ── Блок 2 ───────────────────────────────────────────────────────────
# reshape: (L, d_model) → (L, h, d_k)
Q_split = Q.reshape(L, h, d_k)
K_split = K.reshape(L, h, d_k)
V_split = V.reshape(L, h, d_k)

print(f"\nПісля розбиття:")
print(f"Q_split: {Q_split.shape}")   # (7, 2, 4)
print(f"K_split: {K_split.shape}")   # (7, 2, 4)
print(f"V_split: {V_split.shape}")   # (7, 2, 4)

# Транспонуємо в (h, L, d_k), щоб кожна голова рахувала незалежно
Q_heads = Q_split.transpose(1, 0, 2)
K_heads = K_split.transpose(1, 0, 2)
V_heads = V_split.transpose(1, 0, 2)

print(f"\nПісля транспонування (зручно для паралельних обчислень):")
print(f"Q_heads: {Q_heads.shape}")  # (2, 7, 4)
print(f"K_heads: {K_heads.shape}")  # (2, 7, 4)
print(f"V_heads: {V_heads.shape}")  # (2, 7, 4)

# ── Блок 3 ───────────────────────────────────────────────────────────
def softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

def single_head_attention(Q, K, V):
    """Attention однієї голови, вхід і вихід мають форму (L, d_k)"""
    d_k = Q.shape[-1]
    scores = Q @ K.T               # (L, L)
    scores = scores / np.sqrt(d_k) # масштабування
    weights = softmax(scores)      # (L, L)
    return weights @ V             # (L, d_k)

# Обчислюємо для кожної голови окремо
head_outputs = []
for i in range(h):
    out_i = single_head_attention(Q_heads[i], K_heads[i], V_heads[i])
    head_outputs.append(out_i)
    print(f"Вихід голови {i}: {out_i.shape}")  # (7, 4)

# ── Блок 4 ───────────────────────────────────────────────────────────
# Спершу перетворюємо list на масив (h, L, d_k)
head_outputs = np.stack(head_outputs, axis=0)
print(f"\nПеред склеюванням (stack): {head_outputs.shape}")  # (2, 7, 4)

# Транспонуємо назад у (L, h, d_k), потім reshape у (L, d_model)
head_outputs = head_outputs.transpose(1, 0, 2)
print(f"Після транспонування: {head_outputs.shape}")  # (7, 2, 4)

concat_output = head_outputs.reshape(L, d_model)
print(f"Після склеювання (reshape): {concat_output.shape}")  # (7, 8)

# ── Блок 5 ───────────────────────────────────────────────────────────
# Вихідна проєкційна матриця W_O: (d_model, d_model)
W_O = np.random.randn(d_model, d_model)

# Підсумковий вихід
final_output = concat_output @ W_O
print(f"\nПісля вихідної проєкції: {final_output.shape}")   # (7, 8)
print(f"Форма вхідної X:          {X.shape}")              # (7, 8)
print(f"Форми збігаються:         {final_output.shape == X.shape}")

# ── Блок 6 ───────────────────────────────────────────────────────────
import numpy as np

def multi_head_attention(X, W_Q, W_K, W_V, W_O, h):
    """
    Повна реалізація Multi-Head Attention

    Параметри:
        X:    вхідна матриця, shape (L, d_model)
        W_Q, W_K, W_V: матриці лінійної проєкції, shape (d_model, d_model)
        W_O:  вихідна проєкційна матриця, shape (d_model, d_model)
        h:    кількість голів уваги

    Повертає:
        output: shape (L, d_model)
    """
    L, d_model = X.shape
    d_k = d_model // h

    # ① Лінійне відображення
    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V
    print(f"[① лінійне відображення] Q/K/V: {Q.shape}")

    # ② Розбиття на h голів: (L, d_model) → (h, L, d_k)
    def split_heads(M):
        return M.reshape(L, h, d_k).transpose(1, 0, 2)

    Q_h = split_heads(Q)
    K_h = split_heads(K)
    V_h = split_heads(V)
    print(f"[② розбиття на голови] Q_h/K_h/V_h: {Q_h.shape}")

    # ③ Кожна голова незалежно обчислює Attention
    def softmax(x):
        e = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)

    head_outs = []
    for i in range(h):
        scores = Q_h[i] @ K_h[i].T / np.sqrt(d_k)
        attn = softmax(scores) @ V_h[i]
        head_outs.append(attn)
    print(f"[③ паралельний Attention] вихід кожної голови: {head_outs[0].shape}")

    # ④ Concat: (h, L, d_k) → (L, d_model)
    concat = np.stack(head_outs, axis=0).transpose(1, 0, 2).reshape(L, d_model)
    print(f"[④ Concat] після склеювання: {concat.shape}")

    # ⑤ Вихідна проєкція
    output = concat @ W_O
    print(f"[⑤ вихідна проєкція] підсумковий вихід: {output.shape}")

    return output


# ——— Запуск тесту ———
if __name__ == "__main__":
    np.random.seed(42)
    L, d_model, h = 7, 8, 2

    X   = np.random.randn(L, d_model)
    W_Q = np.random.randn(d_model, d_model)
    W_K = np.random.randn(d_model, d_model)
    W_V = np.random.randn(d_model, d_model)
    W_O = np.random.randn(d_model, d_model)

    print("=== Multi-Head Attention ===")
    out = multi_head_attention(X, W_Q, W_K, W_V, W_O, h)
    print(f"\nФорма входу:      {X.shape}")
    print(f"Форма виходу:     {out.shape}")
    print(f"Форми збігаються: {X.shape == out.shape}")
