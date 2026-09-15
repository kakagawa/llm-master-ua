"""Механізм уваги з нуля

Код зі статті: docs/llm/transformer/attention_code.md
Файл згенеровано скриптом tools/extract-code.mjs — правити треба статтю, не цей файл.

Запуск:  python3 code/transformer/attention.py
Потрібен лише numpy.
"""

# ── Блок 1 ───────────────────────────────────────────────────────────
import numpy as np

# Припустімо, вхідна послідовність: Удалині росте яблуня (уже після embedding)
# Довжина послідовності L=7, розмірність embedding d_model=4
sentence = "Удалині росте яблуня"
L = 7  # довжина послідовності
d_model = 4  # розмірність embedding

# Імітуємо вхід після embedding (у реальності це вихід шару embedding)
np.random.seed(42)
X = np.random.randn(L, d_model)

print("Форма вхідної матриці X:", X.shape)
print("Вміст:\n", X[:3, :])  # друкуємо лише перші 3 токени

# ── Блок 2 ───────────────────────────────────────────────────────────
# Визначаємо три матриці ваг (у реальності це навчані параметри)
d_k = 4  # розмірність Q і K
d_v = 4  # розмірність V

W_Q = np.random.randn(d_model, d_k)
W_K = np.random.randn(d_model, d_k)
W_V = np.random.randn(d_model, d_v)

print("\nФорми матриць ваг:")
print("W_Q:", W_Q.shape)  # (4, 4)
print("W_K:", W_K.shape)  # (4, 4)
print("W_V:", W_V.shape)  # (4, 4)

# Обчислюємо Q, K, V
Q = np.dot(X, W_Q)  # (7, 4) @ (4, 4) = (7, 4)
K = np.dot(X, W_K)  # (7, 4) @ (4, 4) = (7, 4)
V = np.dot(X, W_V)  # (7, 4) @ (4, 4) = (7, 4)

print("\nФорми Q, K, V:")
print("Q:", Q.shape)  # (7, 4)
print("K:", K.shape)  # (7, 4)
print("V:", V.shape)  # (7, 4)

# ── Блок 3 ───────────────────────────────────────────────────────────
# Обчислюємо матрицю оцінок уваги
# Q @ K^T: (7, 4) @ (4, 7) = (7, 7)
attention_scores = np.dot(Q, K.T)

print("\nФорма матриці оцінок уваги:", attention_scores.shape)  # (7, 7)
print("Вміст:\n", attention_scores[:3, :3])  # друкуємо фрагмент 3x3

# ── Блок 4 ───────────────────────────────────────────────────────────
# Масштабування: ділимо на sqrt(d_k)
scaled_scores = attention_scores / np.sqrt(d_k)

print("\nФорма оцінок після масштабування:", scaled_scores.shape)  # (7, 7)
print("Діапазон значень до масштабування:", attention_scores.min(), "до", attention_scores.max())
print("Діапазон значень після масштабування:", scaled_scores.min(), "до", scaled_scores.max())

# ── Блок 5 ───────────────────────────────────────────────────────────
def softmax(x):
    """softmax по кожному рядку"""
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))  # віднімаємо максимум, щоб уникнути переповнення
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

# softmax по кожному рядку
attention_weights = softmax(scaled_scores)

print("\nФорма ваг уваги:", attention_weights.shape)  # (7, 7)
print("Розподіл уваги першого токена:\n", attention_weights[0, :])
print("Сума кожного рядка:", np.sum(attention_weights, axis=-1))  # сума кожного рядка дорівнює 1

# ── Блок 6 ───────────────────────────────────────────────────────────
# Зважуємо V вагами уваги
# (7, 7) @ (7, 4) = (7, 4)
output = np.dot(attention_weights, V)

print("\nФорма підсумкового виходу:", output.shape)  # (7, 4)
print("Форма входу:", X.shape)  # (7, 4)
print("\nПерші 3 токени виходу:\n", output[:3, :])

# ── Блок 7 ───────────────────────────────────────────────────────────
import numpy as np

def scaled_dot_product_attention(Q, K, V):
    """
    Найбазовіший масштабований скалярний добуток уваги

    Параметри:
        Q: матриця Query, shape (L, d_k)
        K: матриця Key, shape (L, d_k)
        V: матриця Value, shape (L, d_v)

    Повертає:
        output: вихідна матриця, shape (L, d_v)
        attention_weights: ваги уваги, shape (L, L)
    """
    d_k = Q.shape[-1]

    # 1. Обчислюємо оцінки уваги: Q @ K^T
    scores = np.dot(Q, K.T)
    print(f"Крок 1 - оцінки уваги: {scores.shape}")

    # 2. Масштабування
    scaled_scores = scores / np.sqrt(d_k)
    print(f"Крок 2 - після масштабування: {scaled_scores.shape}")

    # 3. Softmax
    exp_scores = np.exp(scaled_scores - np.max(scaled_scores, axis=-1, keepdims=True))
    attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    print(f"Крок 3 - ваги уваги: {attention_weights.shape}")

    # 4. Зважене підсумовування
    output = np.dot(attention_weights, V)
    print(f"Крок 4 - підсумковий вихід: {output.shape}")

    return output, attention_weights

# Приклад використання
if __name__ == "__main__":
    # Налаштування входу
    L = 7  # довжина послідовності
    d_model = 4  # розмірність embedding

    np.random.seed(42)

    # 1. Вхід
    X = np.random.randn(L, d_model)
    print(f"Вхід X: {X.shape}\n")

    # 2. Генеруємо Q, K, V
    W_Q = np.random.randn(d_model, d_model)
    W_K = np.random.randn(d_model, d_model)
    W_V = np.random.randn(d_model, d_model)

    Q = np.dot(X, W_Q)
    K = np.dot(X, W_K)
    V = np.dot(X, W_V)

    print(f"Q: {Q.shape}")
    print(f"K: {K.shape}")
    print(f"V: {V.shape}\n")

    # 3. Обчислюємо attention
    print("=== Починаємо обчислення attention ===")
    output, weights = scaled_dot_product_attention(Q, K, V)

    print(f"\nФорма підсумкового виходу: {output.shape}")
    print(f"Форма входу: {X.shape}")
    print(f"Форма не змінилася: {output.shape == X.shape}")
