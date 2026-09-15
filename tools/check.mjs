#!/usr/bin/env node
// Перевірка: залишки китайської + биті відносні посилання.
import { readFileSync, existsSync } from 'node:fs'
import { execSync } from 'node:child_process'
import { dirname, resolve, join } from 'node:path'

const ROOT = resolve(new URL('..', import.meta.url).pathname)
const files = execSync(`find ${join(ROOT,'docs')} -name '*.md'`, {encoding:'utf8'}).trim().split('\n')
const only = process.argv[2]   // необов'язковий фільтр за шляхом

let cjkFiles = 0, broken = 0, checked = 0, intentional = 0, stray = 0
for (const f of files) {
  if (only && !f.includes(only)) continue
  checked++
  const src = readFileSync(f, 'utf8')
  const rel = f.slice(ROOT.length + 1)

  // 1) китайські ієрогліфи поза блоками коду
  const noCode = src.replace(/```[\s\S]*?```/g, '').replace(/`[^`\n]*`/g, '')
  const cjk = [...noCode].filter(c => { const x=c.codePointAt(0); return x>=0x4e00&&x<=0x9fff }).length
  if (cjk) {
    // <!-- cjk-ok: причина --> позначає навмисні китайські приклади (дані вимірювань,
    // які не можна перекладати). Такі файли не вважаються недоперекладеними.
    if (/<!--\s*cjk-ok/.test(src)) { intentional++; console.log(`  ${rel}: ${cjk} ієрогліфів — навмисні (cjk-ok)`) }
    else { console.log(`  ${rel}: ще ${cjk} ієрогліфів`); cjkFiles++ }
  }

  // 2) сторонні писемності (хангиль, кана, тощо) — сміття від помилок вводу
  const junk = [...noCode].filter(c => {
    const x = c.codePointAt(0)
    return (x>=0x3040&&x<=0x30ff) || (x>=0xac00&&x<=0xd7af) || (x>=0x1100&&x<=0x11ff)
  })
  if (junk.length) { console.log(`  ${rel}: сторонні символи: ${[...new Set(junk)].join(' ')}`); stray++ }

  // 3) биті відносні посилання
  for (const m of src.matchAll(/\[[^\]]*\]\((?!https?:|#|mailto:)([^)#]+)(?:#[^)]*)?\)/g)) {
    const target = resolve(dirname(f), m[1])
    if (!existsSync(target)) { console.log(`  ${rel} → битий лінк: ${m[1]}`); broken++ }
  }
}
console.log(`\nперевірено ${checked} файлів | недоперекладених: ${cjkFiles} | навмисна китайська: ${intentional} | сторонні символи: ${stray} | битих лінків: ${broken}`)
