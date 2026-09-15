"""Мережа прямого поширення (FFN) з нуля

Код зі статті: docs/llm/transformer/fnn_code.md
Файл згенеровано скриптом tools/extract-code.mjs — правити треба статтю, не цей файл.

Запуск:  python3 code/transformer/ffn.py
Потрібен лише numpy.
"""

# ── Блок 1 ───────────────────────────────────────────────────────────
import numpy as np

# Гіперпараметри
L = 7          # довжина послідовності
d_model = 8    # розмірність embedding
d_ff = 32      # розмірність проміжного шару FFN (зазвичай учетверо більша за d_model)

np.random.seed(42)

# Імітуємо вихід підшару attention як вхід для FFN
x = np.random.randn(L, d_model)
print(f"Вхід FFN: {x.shape}")  # (7, 8)

# ── Блок 2 ───────────────────────────────────────────────────────────
# Ваги й зсув першого шару
W1 = np.random.randn(d_model, d_ff)  # (8, 32)
b1 = np.random.randn(d_ff)           # (32,)

# Лінійне перетворення: x @ W1 + b1
hidden = x @ W1 + b1
print(f"[після підвищення] hidden: {hidden.shape}")  # (7, 32)

# ── Блок 3 ───────────────────────────────────────────────────────────
def relu(x):
    return np.maximum(0, x)

# Застосовуємо функцію активації
hidden_activated = relu(hidden)
print(f"[після активації] hidden: {hidden_activated.shape}")  # (7, 32)
print(f"Кількість від'ємних до активації: {(hidden < 0).sum()}")
print(f"Кількість від'ємних після активації: {(hidden_activated < 0).sum()}")

# ── Блок 4 ───────────────────────────────────────────────────────────
def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

# ── Блок 5 ───────────────────────────────────────────────────────────
# Ваги й зсув другого шару
W2 = np.random.randn(d_ff, d_model)  # (32, 8)
b2 = np.random.randn(d_model)        # (8,)

# Зниження розмірності
output = hidden_activated @ W2 + b2
print(f"[після зниження] output: {output.shape}")  # (7, 8)
print(f"Форма входу:      {x.shape}")
print(f"Форма виходу:     {output.shape}")
print(f"Форми збігаються: {x.shape == output.shape}")

# ── Блок 6 ───────────────────────────────────────────────────────────
import numpy as np

class FeedForwardNetwork:
    def __init__(self, d_model, d_ff, activation='relu'):
        """
        Feed-Forward Network

        Параметри:
            d_model: розмірність входу/виходу
            d_ff:    розмірність проміжного шару (зазвичай = 4 * d_model)
            activation: тип функції активації, 'relu' або 'gelu'
        """
        # Ваги двох лінійних перетворень
        self.W1 = np.random.randn(d_model, d_ff) * 0.01
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * 0.01
        self.b2 = np.zeros(d_model)
        self.activation = activation

    def _activate(self, x):
        if self.activation == 'relu':
            return np.maximum(0, x)
        elif self.activation == 'gelu':
            return 0.5 * x * (1 + np.tanh(
                np.sqrt(2/np.pi) * (x + 0.044715 * x**3)
            ))

    def forward(self, x):
        # ① Перший шар: підвищення розмірності
        hidden = x @ self.W1 + self.b1
        print(f"[① підвищення] {x.shape} → {hidden.shape}")

        # ② Функція активації: вносимо нелінійність
        hidden = self._activate(hidden)
        print(f"[② активація]  shape не змінюється: {hidden.shape}")

        # ③ Другий шар: зниження розмірності
        output = hidden @ self.W2 + self.b2
        print(f"[③ зниження]   {hidden.shape} → {output.shape}")

        return output


# ——— Запуск тесту ———
if __name__ == "__main__":
    np.random.seed(42)
    L, d_model, d_ff = 7, 8, 32

    x = np.random.randn(L, d_model)
    ffn = FeedForwardNetwork(d_model, d_ff, activation='gelu')

    print("=== Feed-Forward Network ===")
    out = ffn.forward(x)
    print(f"\nФорма входу:      {x.shape}")
    print(f"Форма виходу:     {out.shape}")
    print(f"Форми збігаються: {x.shape == out.shape}")
