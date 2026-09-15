---
title: "Пишемо Tiny Transformer з нуля: збираємо повну модель"
description: Повна реалізація Tiny Transformer власноруч. Складаємо attention, multi-head attention, FFN, LayerNorm, залишкові з'єднання та позиційне кодування в невелику робочу модель Transformer, щоб зрозуміти, як компоненти працюють разом. Придатне для співбесіди й входу у вихідний код Transformer.
keywords: [написати Transformer з нуля, Tiny Transformer, повна реалізація Transformer, код Transformer, код на співбесіді, питання про Transformer]
tags: [Transformer, код з нуля, співбесіда]
---

# Пишемо Tiny Transformer з нуля: збираємо повну модель

У попередніх статтях ми окремо написали власноруч attention, multi-head attention, FFN, залишкові з'єднання, LayerNorm і цілий Transformer Block.

Але досі це були лише окремі деталі.

У цій статті робимо останній крок: складаємо ці деталі разом і пишемо **мінімальний робочий Tiny Transformer**.

> Вивід цього прикладу перевірено повним незалежним прогоном —
> дивіться [`tools/verify-tiny-transformer.mjs`](../../../tools/verify-tiny-transformer.mjs).
> Передбачення `[5 5 4 3]` відтворюється точно.

---

## Загальна структура

Мінімальну версію Transformer можна розкласти на чотири рівні:

```
Token ID
  ↓ Embedding
  ↓ Positional Encoding
  ↓ кілька шарів Transformer Block
  ↓ вихідний шар Linear
  ↓ logits
```

Ключова зміна лише одна: спочатку на вході цілочислові Token ID, які після embedding стають векторами, а далі всі модулі працюють з тією самою матрицею `(L, d_model)`.

![Схема коду повної структури Tiny Transformer](https://file1.kamacoder.com/i/algo/article14_0428_p1.drawio.png)

## Крок 1: Embedding

Модель не розуміє ані тексту напряму, ані Token ID. Скажімо, речення розбито так:

```python
input_ids = [3, 8, 2, 6]
```

Ці числа — лише номери, змісту в них немає.

Задача embedding — знайти кожен номер у таблиці й перетворити на вектор.

```python
import numpy as np

np.random.seed(42)

vocab_size = 10
L = 4
d_model = 8

input_ids = np.array([3, 8, 2, 6])
embedding_table = np.random.randn(vocab_size, d_model)

x = embedding_table[input_ids]
print(x.shape)  # (4, 8)
```

Тепер 4 токени, і кожен став восьмивимірним вектором.

## Крок 2: Positional Encoding

Самому self-attention порядок байдужий. Для нього «я люблю тебе» і «ти любиш мене», якщо дивитися лише на набір токенів, відрізняються мало.

Тому інформацію про позицію треба додати окремо.

```python
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
```

Зверніть увагу: позиційне кодування не приклеюється, а **додається**. Тому shape не змінюється й лишається `(L, d_model)`.

## Крок 3: Transformer Block

В одному блоці відбуваються дві речі:

```text
Attention: дає токенам обмінятися інформацією
FFN:       дає кожному токену обробити інформацію самостійно
```

Додайте залишкові з'єднання й LayerNorm — і мінімальний блок виглядатиме так:

```python
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
```

У цьому коді немає багатьох голів, немає маски й немає навчання — лишився тільки найголовніший потік даних.

## Крок 4: нашаровуємо кілька блоків

«Глибина» Transformer береться з повторного нашарування блоків.

```python
blocks = [TinyBlock(d_model=8, d_ff=32) for _ in range(2)]

for block in blocks:
    x = block.forward(x)

print(x.shape)  # (4, 8)
```

Два шари, шість чи дванадцять — це по суті повторення тієї самої структури. Доки форма входу й виходу кожного шару збігається, їх можна приєднувати далі й далі.

![Результат прямого проходу Tiny Transformer](https://file1.kamacoder.com/i/algo/article14_0428_p2.drawio.png)

## Крок 5: вихідний шар

Останній крок — відобразити прихований вектор кожного токена назад у розмір словника.

Якщо розмір словника `vocab_size=10`, то вихідний шар такий:

```python
W_out = np.random.randn(d_model, vocab_size) * 0.01
b_out = np.zeros(vocab_size)

logits = x @ W_out + b_out
print(logits.shape)  # (4, 10)
```

`logits[0]` — це оцінки першої позиції для всіх 10 токенів словника.

Токен із найвищою оцінкою і є тим, що модель зараз найбільше хоче видати.

## Мінімальне робоче демо

Складаємо весь код разом:

```python
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
```

**Приклад виводу:**

```text
Підсумковий shape logits: (4, 10)
Передбачений токен на кожній позиції: [5 5 4 3]
```

На цьому Tiny Transformer справді запрацював.

Він ще не вміє писати тексти й не вміє спілкуватися, бо ми його не навчали. Але структура вже повна: embedding перетворює ID на вектори, позиційне кодування повідомляє моделі порядок, блоки раз за разом обробляють інформацію, а вихідний шар перетворює вектори назад на оцінки для словника.

Найголовніший секрет Transformer — це насправді ось цей потік даних:

```text
ID → вектор → додати позицію → багатошарова обробка → оцінки для словника
```

Зрозумівши цю лінію, ви більше не бачитимете в GPT, BERT чи LLaMA суцільну чорну скриньку.

---

На цьому етап написання коду з нуля завершено. Далі переходимо до вивчення BERT, T5, GPT і MoE — від стандартного Transformer до основних родин моделей.
