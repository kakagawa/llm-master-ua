#!/usr/bin/env node
// Вирізає промо-блоки оригіналу (банер KamaClaude, QR публічки, рекламні врізки),
// не чіпаючи змістовні ілюстрації з file1.kamacoder.com.
import { readFileSync, writeFileSync } from 'node:fs'
import { execSync } from 'node:child_process'

const PROMO = [
  // <a href="...programmercarl.com..."> <img ...> </a>  — банер-картинка
  /<a\s+href="https:\/\/(?:www\.)?(?:programmercarl|notes\.kamacoder)\.com[^"]*"[^>]*>\s*<img[^>]*>\s*<\/a>\s*/gi,
  // самотній QR-код публічки
  /<img[^>]*(?:卡码大模型二维码|卡哥直播)[^>]*>\s*/gi,
  /!\[[^\]]*\]\(https:\/\/file1\.kamacoder\.com\/i\/web\/(?:卡码大模型二维码|卡哥直播)[^)]*\)\s*/gi,
  // <p align="center"> з QR всередині
  /<p align="center">\s*<img[^>]*(?:二维码|卡哥直播)[^>]*>\s*<\/p>\s*/gi,
]

const files = execSync(`find ${process.argv[2]} -name '*.md'`, { encoding: 'utf8' })
  .trim().split('\n').filter(Boolean)

let touched = 0, removed = 0
for (const f of files) {
  const before = readFileSync(f, 'utf8')
  let after = before
  for (const re of PROMO) after = after.replace(re, '')
  after = after.replace(/\n{4,}/g, '\n\n\n')
  if (after !== before) { writeFileSync(f, after); touched++; removed += (before.length - after.length) }
}
console.log(`очищено файлів: ${touched}/${files.length}, прибрано ${(removed/1024).toFixed(1)} КБ промо`)
