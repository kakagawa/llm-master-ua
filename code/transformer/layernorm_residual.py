"""LayerNorm і залишкове з'єднання з нуля

Код зі статті: docs/llm/transformer/layernorm_residual_code.md
Файл згенеровано скриптом tools/extract-code.mjs — правити треба статтю, не цей файл.

Запуск:  python3 code/transformer/layernorm_residual.py
Потрібен лише numpy.
"""

# ── Блок 1 ───────────────────────────────────────────────────────────
import numpy as np

def residual_connection(x, sublayer_output):
    """
    Залишкове з'єднання: просто додаємо

    Параметри:
        x:               вхід підшару, shape (L, d_model)
        sublayer_output: вихід підшару, shape (L, d_model)

    Повертає:
        shape (L, d_model), повністю збігається з входом
    """
    return x + sublayer_output

# Тест
L, d_model = 7, 8
np.random.seed(42)

x = np.random.randn(L, d_model)             # вхід підшару
sublayer_out = np.random.randn(L, d_model)  # вихід підшару (імітація)

output = residual_connection(x, sublayer_out)
print(f"Форма входу:            {x.shape}")             # (7, 8)
print(f"Вихід підшару:          {sublayer_out.shape}")  # (7, 8)
print(f"Залишковий вихід:       {output.shape}")        # (7, 8)

# ── Блок 2 ───────────────────────────────────────────────────────────
class LayerNorm:
    def __init__(self, d_model, eps=1e-5):
        """
        LayerNorm

        Параметри:
            d_model: розмірність вектора
            eps:     мала константа проти ділення на нуль
        """
        self.gamma = np.ones(d_model)   # навчаний параметр масштабу, ініціалізується одиницями
        self.beta  = np.zeros(d_model)  # навчаний параметр зсуву, ініціалізується нулями
        self.eps   = eps

    def forward(self, x):
        """
        x: shape (L, d_model)
        """
        # Для кожного токена (кожного рядка) окремо рахуємо середнє й станд. відхилення
        mu    = x.mean(axis=-1, keepdims=True)          # (L, 1)
        sigma = x.std(axis=-1, keepdims=True)           # (L, 1)

        # Стандартизація
        x_norm = (x - mu) / (sigma + self.eps)          # (L, d_model)

        # Масштаб + зсув (broadcast на кожен рядок)
        return self.gamma * x_norm + self.beta          # (L, d_model)


# Тест
np.random.seed(42)
L, d_model = 7, 8
x = np.random.randn(L, d_model) * 10  # навмисно збільшуємо значення, імітуючи нестабільність

ln = LayerNorm(d_model)
output = ln.forward(x)

print(f"До нормалізації  - середнє: {x.mean():.2f}, станд. відхилення: {x.std():.2f}")
print(f"Після нормалізації - середнє: {output.mean():.4f}, станд. відхилення: {output.std():.4f}")
print(f"Форма входу: {x.shape}, форма виходу: {output.shape}")

# ── Блок 3 ───────────────────────────────────────────────────────────
def add_and_norm(x, sublayer_output, layer_norm):
    """
    Add & Norm: залишкове з'єднання + LayerNorm

    Параметри:
        x:               вхід підшару,  shape (L, d_model)
        sublayer_output: вихід підшару, shape (L, d_model)
        layer_norm:      екземпляр LayerNorm

    Повертає:
        shape (L, d_model)
    """
    return layer_norm.forward(x + sublayer_output)


# Повний тест
np.random.seed(42)
L, d_model = 7, 8

x            = np.random.randn(L, d_model)
attn_output  = np.random.randn(L, d_model)  # імітуємо вихід підшару attention
ffn_output   = np.random.randn(L, d_model)  # імітуємо вихід підшару FFN

ln1 = LayerNorm(d_model)
ln2 = LayerNorm(d_model)

# Підшар attention → Add & Norm
after_attn = add_and_norm(x, attn_output, ln1)
print(f"Після підшару attention: {after_attn.shape}")  # (7, 8)

# Підшар FFN → Add & Norm
after_ffn  = add_and_norm(after_attn, ffn_output, ln2)
print(f"Після підшару FFN:       {after_ffn.shape}")   # (7, 8)

print(f"\nФорма входу:      {x.shape}")
print(f"Форма виходу:     {after_ffn.shape}")
print(f"Форми збігаються: {x.shape == after_ffn.shape}")

# ── Блок 4 ───────────────────────────────────────────────────────────
# Post-Norm (оригінальний Transformer)
def post_norm(x, sublayer_fn, layer_norm):
    return layer_norm.forward(x + sublayer_fn(x))

# Pre-Norm (сучасні великі моделі)
def pre_norm(x, sublayer_fn, layer_norm):
    return x + sublayer_fn(layer_norm.forward(x))
