# Код із статей

Тут лежить код зі статей у вигляді файлів, які можна просто запустити. Це не окрема
бібліотека: файли генеруються зі статей скриптом `tools/extract-code.mjs`, тому
**правити треба статтю, а не файл тут** — інакше зміни зникнуть під час наступної генерації.

## Запуск

```bash
python3 -m pip install -r code/requirements.txt   # потрібен лише numpy
python3 code/transformer/attention.py
```

## Що є

| Файл | Стаття |
|---|---|
| [`transformer/attention.py`](transformer/attention.py) | [Пишемо attention з нуля](../docs/llm/transformer/attention_code.md) |
| [`transformer/multi_head_attention.py`](transformer/multi_head_attention.py) | [Пишемо Multi-Head Attention](../docs/llm/transformer/mha_code.md) |
| [`transformer/layernorm_residual.py`](transformer/layernorm_residual.py) | [Пишемо LayerNorm і залишкове з'єднання](../docs/llm/transformer/layernorm_residual_code.md) |
| [`transformer/ffn.py`](transformer/ffn.py) | [Пишемо FFN](../docs/llm/transformer/fnn_code.md) |
| [`transformer/transformer_block.py`](transformer/transformer_block.py) | [Пишемо Transformer Block](../docs/llm/transformer/transformer_block_code.md) |
| [`transformer/tiny_transformer.py`](transformer/tiny_transformer.py) | [Пишемо Tiny Transformer](../docs/llm/transformer/tiny_transformer_code.md) |

Усі шість файлів виконуються без помилок на Python 3.14 і numpy 2.5. Кожен друкує
форми тензорів на кожному кроці — саме те, що в статті розписано текстом.

Однорядкові фрагменти-ілюстрації зі статей (наприклад, порівняння Post-LN і Pre-LN)
у файли не потрапляють: вони не призначені для запуску.
