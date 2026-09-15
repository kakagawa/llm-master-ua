// Витягає код зі статей розділу «Пишемо Transformer з нуля» у файли, які можна запустити.
// Блоки в статті йдуть послідовно й будуються один на одному, тож склеюємо їх у тому ж порядку.
//
// Запуск:  node tools/extract-code.mjs
import { readFileSync, writeFileSync, mkdirSync } from "node:fs";

const СТАТТІ = [
  ["attention_code", "attention.py", "Механізм уваги з нуля"],
  ["mha_code", "multi_head_attention.py", "Multi-Head Attention з нуля"],
  ["layernorm_residual_code", "layernorm_residual.py", "LayerNorm і залишкове з'єднання з нуля"],
  ["fnn_code", "ffn.py", "Мережа прямого поширення (FFN) з нуля"],
  ["transformer_block_code", "transformer_block.py", "Transformer Block з нуля"],
  ["tiny_transformer_code", "tiny_transformer.py", "Tiny Transformer: повна модель з нуля"],
];

mkdirSync("code/transformer", { recursive: true });

for (const [файл, вихід, назва] of СТАТТІ) {
  const шлях = `docs/llm/transformer/${файл}.md`;
  const md = readFileSync(шлях, "utf8");
  const усі = [...md.matchAll(/```python\n([\s\S]*?)```/g)].map(m => m[1].trimEnd());
  // Частина блоків у статтях — це ілюстративні однорядкові фрагменти
  // (наприклад, Post-LN проти Pre-LN), які самі по собі не виконуються. Пропускаємо їх.
  const блоки = усі.filter(b => {
    const рядки = b.split("\n").filter(r => r.trim() && !r.trim().startsWith("#"));
    return рядки.length > 2 || /\b(import|def|class|print)\b/.test(b);
  });

  const шапка = `"""${назва}

Код зі статті: docs/llm/transformer/${файл}.md
Файл згенеровано скриптом tools/extract-code.mjs — правити треба статтю, не цей файл.

Запуск:  python3 code/transformer/${вихід}
Потрібен лише numpy.
"""

`;
  const тіло = блоки
    .map((b, i) => `# ── Блок ${i + 1} ${"─".repeat(Math.max(0, 60 - String(i + 1).length))}\n${b}`)
    .join("\n\n");

  writeFileSync(`code/transformer/${вихід}`, шапка + тіло + "\n");
  console.log(`${вихід.padEnd(26)} ${блоки.length}/${усі.length} блоків, ${(шапка + тіло).split("\n").length} рядків`);
}
