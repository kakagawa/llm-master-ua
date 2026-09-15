#!/usr/bin/env node
/**
 * Перекладач llm-master → українською.
 *
 *   node tools/translate.mjs count  docs                 # оцінити токени й вартість
 *   node tools/translate.mjs sample docs/llm/app/x.md    # синхронно, для перевірки якості
 *   node tools/translate.mjs submit docs                 # відправити Batch (−50% ціни)
 *   node tools/translate.mjs status <batch_id>
 *   node tools/translate.mjs fetch  <batch_id>           # забрати й записати результати
 */
import Anthropic from '@anthropic-ai/sdk'
import { readFileSync, writeFileSync, mkdirSync, existsSync, readdirSync, statSync } from 'node:fs'
import { dirname, join, relative, resolve } from 'node:path'

const MODEL = 'claude-opus-5'
const EFFORT = 'high'
const MAX_TOKENS = 64000
const ADDRESS = 'ви'          // ← перемкни на 'ти', якщо хочеш неформальний тон
const ROOT = resolve(new URL('..', import.meta.url).pathname)

// підтягуємо .env (ANTHROPIC_API_KEY), якщо змінна ще не виставлена в оточенні
if (!process.env.ANTHROPIC_API_KEY && existsSync(join(ROOT, '.env'))) {
  for (const line of readFileSync(join(ROOT, '.env'), 'utf8').split('\n')) {
    const m = line.match(/^\s*(?:export\s+)?([A-Z_][A-Z0-9_]*)\s*=\s*(.*)\s*$/)
    if (m) process.env[m[1]] ??= m[2].replace(/^["']|["']$/g, '')
  }
}

const client = new Anthropic()

const glossary = readFileSync(join(ROOT, 'GLOSSARY.md'), 'utf8')

const SYSTEM = [
  {
    type: 'text',
    text: `Ти — технічний перекладач і редактор. Перекладаєш навчальні статті про великі мовні моделі з китайської на українську для репозиторію llm-master-ua.

Читач — досвідчений розробник (Java/Go/Python/C++/фронтенд), який заходить у розробку застосунків на LLM. Він знає програмування, але не знає ML-жаргону.

# Що має вийти

Не підрядник, а стаття, яка читається так, ніби її одразу написали українською. Китайський технічний текст любить короткі рубані речення, риторичні питання й повтори — частину повторів прибирай, ритм роби природним для української.

# Жорсткі правила

1. **Код не чіпай взагалі.** Вміст блоків \`\`\`...\`\`\` та інлайн-коду \`...\` залишається байт у байт. Виняток — коментарі всередині коду китайською: їх перекладай.
2. **Frontmatter (YAML між --- на початку файлу):** переклади значення полів title, description, keywords, tags. Назви полів і структуру YAML не змінюй. Якщо frontmatter немає — не додавай.
3. **Посилання й картинки не чіпай.** Відносні шляхи (\`../claude/claude_md.md\`), URL, імена файлів — залишаються як є. Перекладається лише текст підпису в \`[текст](посилання)\`.
4. **HTML-теги, mermaid-діаграми, таблиці** — зберігай структуру. У mermaid перекладай лише текст у вузлах.
5. **Рівні заголовків (#, ##, ###) зберігай точно.**
6. **Нічого не додавай і не викидай.** Без передмов, без «Примітка перекладача», без підсумків від себе.
7. Звертання до читача — на «${ADDRESS}».
8. Чинний український правопис: «проєкт», «матеріал».

# Термінологія

Глосарій нижче обов'язковий. Якщо терміна в ньому немає — обирай варіант, який реально вживають українські розробники, і радше залиши англійський термін, ніж вигадуй кальку.

${glossary}

# Формат відповіді

Поверни ЛИШЕ повний markdown перекладеної статті. Без обгортки \`\`\`markdown, без коментарів до або після.`,
    cache_control: { type: 'ephemeral' },   // глосарій стабільний → кешуємо префікс
  },
]

const userMsg = (src, relPath) =>
  `Файл: ${relPath}\n\nПереклади цю статтю українською за правилами вище.\n\n<article>\n${src}\n</article>`

function mdFiles(dir) {
  const out = []
  ;(function walk(d) {
    for (const e of readdirSync(d)) {
      const p = join(d, e)
      if (statSync(p).isDirectory()) walk(p)
      else if (e.endsWith('.md')) out.push(p)
    }
  })(resolve(dir))
  return out.sort()
}

const idFor = (p) => relative(ROOT, p).replace(/[^\w]/g, '_').slice(0, 60)

function params(src, relPath) {
  return {
    model: MODEL,
    max_tokens: MAX_TOKENS,
    system: SYSTEM,
    thinking: { type: 'adaptive' },
    output_config: { effort: EFFORT },
    messages: [{ role: 'user', content: userMsg(src, relPath) }],
  }
}

// ── count ───────────────────────────────────────────────────────────────────
async function cmdCount(dir) {
  const files = mdFiles(dir)
  let inTok = 0
  for (const f of files) {
    const src = readFileSync(f, 'utf8')
    const r = await client.messages.countTokens({
      model: MODEL,
      system: SYSTEM,
      messages: [{ role: 'user', content: userMsg(src, relative(ROOT, f)) }],
    })
    inTok += r.input_tokens
    process.stdout.write('.')
  }
  // українська приблизно в 1.7× довша за китайську в токенах
  const outTok = Math.round(inTok * 1.7)
  const price = { 'claude-opus-5': [5, 25], 'claude-sonnet-5': [2, 10], 'claude-haiku-4-5': [1, 5] }
  console.log(`\n\nфайлів: ${files.length}`)
  console.log(`вхідних токенів:  ${inTok.toLocaleString('uk')}`)
  console.log(`вихідних (оцінка): ${outTok.toLocaleString('uk')}\n`)
  console.log('модель            звичайний API   Batch (−50%)')
  for (const [m, [pi, po]] of Object.entries(price)) {
    const usd = (inTok / 1e6) * pi + (outTok / 1e6) * po
    console.log(`${m.padEnd(18)} $${usd.toFixed(2).padStart(8)}   $${(usd / 2).toFixed(2).padStart(8)}`)
  }
}

// ── sample ──────────────────────────────────────────────────────────────────
async function cmdSample(...files) {
  for (const f of files) {
    const p = resolve(f)
    const src = readFileSync(p, 'utf8')
    console.error(`→ ${relative(ROOT, p)} (${src.length} байт)`)
    const stream = client.messages.stream(params(src, relative(ROOT, p)))
    const msg = await stream.finalMessage()
    if (msg.stop_reason === 'refusal') { console.error('  ВІДМОВА:', msg.stop_details); continue }
    const text = msg.content.filter((b) => b.type === 'text').map((b) => b.text).join('')
    const out = join(ROOT, 'out', relative(ROOT, p))
    mkdirSync(dirname(out), { recursive: true })
    writeFileSync(out, text.replace(/^```markdown\n|\n```$/g, ''))
    const u = msg.usage
    console.error(`  ok → out/${relative(ROOT, p)}  (in ${u.input_tokens}, out ${u.output_tokens}, cache_read ${u.cache_read_input_tokens ?? 0})`)
  }
}

// ── submit / status / fetch ─────────────────────────────────────────────────
async function cmdSubmit(dir) {
  const files = mdFiles(dir)
  const map = {}
  const requests = files.map((f) => {
    const id = idFor(f)
    map[id] = relative(ROOT, f)
    return { custom_id: id, params: params(readFileSync(f, 'utf8'), relative(ROOT, f)) }
  })
  const batch = await client.messages.batches.create({ requests })
  mkdirSync(join(ROOT, 'batches'), { recursive: true })
  writeFileSync(join(ROOT, 'batches', `${batch.id}.json`), JSON.stringify(map, null, 2))
  console.log(`батч ${batch.id}: ${requests.length} запитів`)
  console.log(`мапу custom_id→файл збережено у batches/${batch.id}.json`)
}

async function cmdStatus(id) {
  const b = await client.messages.batches.retrieve(id)
  console.log(b.processing_status, JSON.stringify(b.request_counts))
}

async function cmdFetch(id) {
  const map = JSON.parse(readFileSync(join(ROOT, 'batches', `${id}.json`), 'utf8'))
  let ok = 0, bad = 0
  for await (const r of await client.messages.batches.results(id)) {
    const rel = map[r.custom_id]
    if (r.result.type !== 'succeeded') { console.error(`✗ ${rel}: ${r.result.type}`); bad++; continue }
    const msg = r.result.message
    if (msg.stop_reason === 'refusal') { console.error(`✗ ${rel}: refusal`); bad++; continue }
    const text = msg.content.filter((b) => b.type === 'text').map((b) => b.text).join('')
    const out = join(ROOT, rel)          // пишемо поверх — diff покаже git
    writeFileSync(out, text.replace(/^```markdown\n|\n```$/g, ''))
    ok++
  }
  console.log(`записано ${ok}, помилок ${bad}`)
}

const [cmd, ...args] = process.argv.slice(2)
const cmds = { count: cmdCount, sample: cmdSample, submit: cmdSubmit, status: cmdStatus, fetch: cmdFetch }
if (!cmds[cmd]) { console.error('команди: count | sample | submit | status | fetch'); process.exit(1) }
if (!existsSync(join(ROOT, 'node_modules'))) { console.error('спершу: npm install'); process.exit(1) }
await cmds[cmd](...args)
