---
title: "Пишемо Transformer Block з нуля: складаємо докупи attention, FFN і Norm"
description: Реалізація Transformer Block власноруч. Збираємо multi-head attention, FFN, LayerNorm, залишкові з'єднання та інші компоненти в один повний блок і крок за кроком розбираємо рух даних усередині, зміну розмірностей і логіку прямого проходу. Придатне для вивчення вихідного коду Transformer і написання коду на співбесіді.
keywords: [написати Transformer Block з нуля, код Transformer Block, складання компонентів Transformer, код на співбесіді, питання про Transformer]
tags: [Transformer, код з нуля, співбесіда]
---

# Пишемо Transformer Block з нуля: складаємо докупи attention, FFN і Norm

У попередніх статтях ми вже написали власноруч усі ключові деталі Transformer Block:

- **Multi-head attention:** дає токенам обмінюватися інформацією між собою
- **FFN:** дає кожному токену незалежно виконати нелінійну обробку
- **Залишкові з'єднання:** повертають початковий вхід, щоб інформація не губилася
- **LayerNorm:** утримує значення в стабільному діапазоні

Тепер ми не розглядаємо деталі окремо, а справді складаємо їх разом і пишемо **мінімальну версію Transformer Block**.

> Код у цій статті друкує лише розмірності тензорів. Усі вони однозначно визначаються
> параметрами `L=7`, `d_model=8`, `num_heads=2`, `d_ff=32`.

---

## Який вигляд має один Block?

Найпоширеніший Transformer Block записується так:

```text
вхід x
  ↓
Multi-Head Attention
  ↓
залишкове з'єднання + LayerNorm
  ↓
FFN
  ↓
залишкове з'єднання + LayerNorm
  ↓
вихід
```

Тобто:

$$
x_1 = \text{LayerNorm}(x + \text{MHA}(x))
$$

$$
x_2 = \text{LayerNorm}(x_1 + \text{FFN}(x_1))
$$

Тут є надзвичайно важливий момент:

> **Якої форми вхід — такої самої форми й вихід.**

Припустімо, вхід такий:

```text
x.shape = (L, d_model)
```

де:

- L — довжина послідовності; наприклад, у «Удалині росте яблуня» 7 токенів
- d_model — розмірність вектора кожного токена; тут для демонстрації 8 вимірів

Тоді після одного Transformer Block вихід лишається:

```text
output.shape = (L, d_model)
```

Саме тому Transformer можна нашаровувати шар за шаром.

![Схема структури коду Transformer Block](https://file1.kamacoder.com/i/algo/article14_0425_p1.png)

## Спершу пишемо LayerNorm і FFN

Підготуймо найбазовіші компоненти.

```python
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
```

LayerNorm робить дуже просту річ: нормалізує вектор кожного токена окремо.

Скажімо, вхід має форму `(7, 8)`, тобто 7 токенів по 8 вимірів кожен, — LayerNorm нормалізує ці 7 рядків окремо один від одного.

Далі пишемо FFN:

```python
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
```

Процес FFN такий:

```text
(L, d_model)
   ↓ підвищення розмірності
(L, d_ff)
   ↓ GELU
(L, d_ff)
   ↓ зниження розмірності
(L, d_model)
```

Знову те саме правило: **посередині може відбуватися що завгодно, але наприкінці треба повернутися до d_model.**

## Пишемо Multi-Head Attention

Тепер напишемо мінімальну версію багатоголової уваги.

```python
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
```

Найлегше заплутатися тут саме у формах:

```text
вхід x:                  (L, d_model)
Q/K/V:                   (L, d_model)
розбиття на голови:      (num_heads, L, d_k)
вихід кожної голови:     (L, d_k)
склеювання назад:        (L, d_model)
після вихідної проєкції: (L, d_model)
```

Як бачимо, хоч attention і розбивається посередині на кілька голів, наприкінці все одно повертається до початкової форми.

![Схема виводу прямого проходу Transformer Block](https://file1.kamacoder.com/i/algo/article14_0425_p2.png)

---

## Визначаємо повний Transformer Block

Тепер найголовніше.

```python
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
```

Це і є мінімальна версія Transformer Block.

Зверніть увагу, що тут використано запис Post-Norm:

```python
x = norm(x + sublayer(x))
```

Тобто спершу проходимо підшар, потім залишкове додавання, і наостанок LayerNorm.

Багато сучасних великих моделей використовують Pre-Norm:

```python
x = x + sublayer(norm(x))
```

Але щоб ближче відповідати структурі оригінального Transformer і щоб початківцям було зрозуміліше, тут спершу беремо Post-Norm.

---

## Проганяємо іграшковий приклад

Далі беремо вже знайоме речення:

> Удалині росте яблуня

Припустімо, що воно розбите на 7 токенів, і кожен подано восьмивимірним вектором.

```python
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
```

Вивід під час запуску приблизно такий:

```text
Вхід x:                (7, 8)
Вихід Attention:       (7, 8)
Після Add & Norm:      (7, 8)
Вихід FFN:             (7, 8)
Підсумковий вихід:     (7, 8)

=== Перевірка результату ===
Форма входу: (7, 8)
Форма виходу: (7, 8)
Форми збігаються: True
```

Це означає, що наша мінімальна версія Transformer Block запрацювала.

Від входу до виходу форма незмінно лишається `(7, 8)`.

Але зверніть увагу: **форма не змінилася — це не означає, що не змінився вміст.**

Після attention кожен токен уже містить контекстну інформацію інших токенів.

Після FFN кожен токен окремо пройшов ще одну нелінійну обробку.

Додайте до цього залишкові з'єднання й LayerNorm — і весь Block здатен і виражати складний зміст, і зберігати стабільність навчання.

---

## Підсумок

Один Transformer Block по суті робить ось що:

```text
спершу дає токенам обмінятися інформацією,
потім дає кожному токену подумати самому,
на кожному кроці зберігає початкову інформацію залишковим з'єднанням,
а потім стабілізує значення через LayerNorm.
```

Якщо стиснути це до коду, вийде два рядки:

```python
x = norm1(x + attention(x))
x = norm2(x + ffn(x))
```

Оце і є суть Transformer Block.

Виглядає просто, але великі моделі саме такі блоки й нашаровують — десятками й сотнями шарів, — а потім навчають на величезних даних і обчислювальних потужностях.

У наступній статті **зберемо Tiny Transformer з нуля**.
