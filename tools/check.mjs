#!/usr/bin/env node
// Перевірка: залишки китайської + биті відносні посилання.
import { readFileSync, existsSync } from 'node:fs'
import { execSync } from 'node:child_process'
import { dirname, resolve, join } from 'node:path'

const ROOT = resolve(new URL('..', import.meta.url).pathname)
const files = execSync(`find ${join(ROOT,'docs')} -name '*.md'`, {encoding:'utf8'}).trim().split('\n')
const only = process.argv[2]   // необов'язковий фільтр за шляхом

let cjkFiles = 0, broken = 0, checked = 0
for (const f of files) {
  if (only && !f.includes(only)) continue
  checked++
  const src = readFileSync(f, 'utf8')
  const rel = f.slice(ROOT.length + 1)

  // 1) китайські ієрогліфи поза блоками коду
  const noCode = src.replace(/```[\s\S]*?```/g, '').replace(/`[^`\n]*`/g, '')
  const cjk = [...noCode].filter(c => { const x=c.codePointAt(0); return x>=0x4e00&&x<=0x9fff }).length
  if (cjk) { console.log(`  ${rel}: ще ${cjk} ієрогліфів`); cjkFiles++ }

  // 2) биті відносні посилання
  for (const m of src.matchAll(/\[[^\]]*\]\((?!https?:|#|mailto:)([^)#]+)(?:#[^)]*)?\)/g)) {
    const target = resolve(dirname(f), m[1])
    if (!existsSync(target)) { console.log(`  ${rel} → битий лінк: ${m[1]}`); broken++ }
  }
}
console.log(`\nперевірено ${checked} файлів | з китайською: ${cjkFiles} | битих лінків: ${broken}`)
