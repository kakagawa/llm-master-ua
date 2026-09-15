// Скільки токенів коштує українська мова.
// Міряємо на реальному паралельному корпусі: китайський оригінал (коміт baseline)
// проти українського перекладу (HEAD) — ті самі статті, той самий зміст.
//
// Запуск:  npm install && node tools/tokens-ua.mjs
import { execSync } from "node:child_process";
import { createRequire } from "node:module";
import { readFileSync } from "node:fs";

const require = createRequire(import.meta.url);
const o200k = require("gpt-tokenizer/encoding/o200k_base"); // GPT-5, GPT-4o
const cl100k = require("gpt-tokenizer/encoding/cl100k_base"); // GPT-4, GPT-3.5

const BASELINE = process.env.BASELINE || "82137ed";

// Лишаємо тільки прозу: без frontmatter, коду, лінків, розмітки й картинок.
function proza(md) {
  return md
    .replace(/^---\n[\s\S]*?\n---\n/, "")
    .replace(/```[\s\S]*?```/g, "")
    .replace(/!\[[^\]]*\]\([^)]*\)/g, "")
    .replace(/\[([^\]]*)\]\([^)]*\)/g, "$1")
    .replace(/^[>#\-*|]+ ?/gm, "")
    .replace(/[`*_~]/g, "")
    .replace(/\n{2,}/g, "\n")
    .trim();
}

const cyr = s => (s.match(/\p{Script=Cyrillic}/gu) || []).length;
const han = s => (s.match(/\p{Script=Han}/gu) || []).length;

const files = execSync(`git ls-tree -r --name-only ${BASELINE} -- docs`, { encoding: "utf8" })
  .split("\n")
  .filter(f => f.endsWith(".md"));

let zh = { ch: 0, o: 0, c: 0, n: 0 };
let ua = { ch: 0, o: 0, c: 0, n: 0 };
const pары = [];

for (const f of files) {
  let stary, novy;
  try {
    stary = proza(execSync(`git show ${BASELINE}:${f}`, { encoding: "utf8", maxBuffer: 1e8 }));
    novy = proza(execSync(`git show HEAD:${f}`, { encoding: "utf8", maxBuffer: 1e8 }));
  } catch {
    continue; // файл вирізали з форку
  }
  if (f.endsWith("README.md")) continue; // індекси — це списки заголовків, не проза
  if (han(stary) < 500 || cyr(novy) < 500) continue;

  const z = { ch: stary.length, o: o200k.encode(stary).length, c: cl100k.encode(stary).length };
  const u = { ch: novy.length, o: o200k.encode(novy).length, c: cl100k.encode(novy).length };
  zh.ch += z.ch; zh.o += z.o; zh.c += z.c; zh.n++;
  ua.ch += u.ch; ua.o += u.o; ua.c += u.c; ua.n++;
  pары.push({ f, z, u });
}

const f2 = x => x.toFixed(2);
console.log(`Паралельний корпус: ${zh.n} статей (китайський оригінал ↔ український переклад)\n`);
console.log("                     символів     o200k      cl100k   o200k/симв  cl100k/симв");
for (const [naz, d] of [["китайська", zh], ["українська", ua]]) {
  console.log(
    `${naz.padEnd(18)} ${String(d.ch).padStart(9)} ${String(d.o).padStart(9)} ${String(d.c).padStart(9)}` +
    `  ${f2(d.o / d.ch).padStart(10)}  ${f2(d.c / d.ch).padStart(11)}`
  );
}
console.log(`\nТокенів на ту саму статтю (укр / кит):`);
console.log(`  o200k_base  (GPT-5, GPT-4o):  ×${f2(ua.o / zh.o)}`);
console.log(`  cl100k_base (GPT-4, GPT-3.5): ×${f2(ua.c / zh.c)}`);
console.log(`\nСимволів на токен:`);
console.log(`  китайська  o200k ${f2(zh.ch / zh.o)}   cl100k ${f2(zh.ch / zh.c)}`);
console.log(`  українська o200k ${f2(ua.ch / ua.o)}   cl100k ${f2(ua.ch / ua.c)}`);

// Скільки коштує сам перехід cl100k → o200k для української
console.log(`\nВиграш від нового токенізатора (cl100k → o200k):`);
console.log(`  українська: −${f2((1 - ua.o / ua.c) * 100)} %`);
console.log(`  китайська:  −${f2((1 - zh.o / zh.c) * 100)} %`);

// ── Частина 2: українська проти англійської ────────────────────────────────
// Паралельний набір із 12 типових для LLM-застосунку текстів (tools/data/parallel-ua-en.json).
const pairs = JSON.parse(readFileSync(new URL("./data/parallel-ua-en.json", import.meta.url), "utf8")).pairs;
const en = { ch: 0, o: 0, c: 0, w: 0 }, uk = { ch: 0, o: 0, c: 0, w: 0 };
for (const p of pairs) {
  for (const [lang, acc] of [["ua", uk], ["en", en]]) {
    acc.ch += p[lang].length;
    acc.o += o200k.encode(p[lang]).length;
    acc.c += cl100k.encode(p[lang]).length;
    acc.w += p[lang].trim().split(/\s+/).length;
  }
}
console.log(`\n── Українська проти англійської (${pairs.length} паралельних текстів) ──\n`);
console.log("                     символів     слів     o200k    cl100k");
console.log(`англійська        ${String(en.ch).padStart(10)} ${String(en.w).padStart(8)} ${String(en.o).padStart(9)} ${String(en.c).padStart(9)}`);
console.log(`українська        ${String(uk.ch).padStart(10)} ${String(uk.w).padStart(8)} ${String(uk.o).padStart(9)} ${String(uk.c).padStart(9)}`);
console.log(`\nТокенів на той самий зміст (укр / англ):`);
console.log(`  o200k_base:  ×${f2(uk.o / en.o)}`);
console.log(`  cl100k_base: ×${f2(uk.c / en.c)}`);
console.log(`Токенів на слово: англ ${f2(en.o / en.w)}, укр ${f2(uk.o / uk.w)} (o200k)`);
console.log(`Символів на токен: англ ${f2(en.ch / en.o)}, укр ${f2(uk.ch / uk.o)} (o200k)`);

// Найдорожчі й найдешевші статті українською
pары.sort((a, b) => b.u.o / b.z.o - a.u.o / a.z.o);
console.log(`\nНайбільший розрив укр/кит (o200k):`);
for (const p of pары.slice(0, 3)) console.log(`  ×${f2(p.u.o / p.z.o)}  ${p.f}`);
console.log(`Найменший розрив:`);
for (const p of pары.slice(-3)) console.log(`  ×${f2(p.u.o / p.z.o)}  ${p.f}`);
