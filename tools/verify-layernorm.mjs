#!/usr/bin/env node
// Перевірка числових тверджень у docs/llm/transformer/layernorm_residual_code.md
import { NumpyRandom } from './numpy-rng.mjs'
const mean = a => a.reduce((s,v)=>s+v,0)/a.length
const std  = a => { const m=mean(a); return Math.sqrt(mean(a.map(v=>(v-m)**2))) }

const r = new NumpyRandom(42)
const L=7, d=8
const x = r.randn(L,d).map(row => row.map(v => v*10))
const flat = x.flat()
console.log(`до нормалізації  — середнє: ${mean(flat).toFixed(2)}, std: ${std(flat).toFixed(2)}`)
console.log(`   (в оригіналі було -0.37 / 9.87 — помилка)`)

const eps = 1e-5
const out = x.map(row => { const m=mean(row), s=std(row); return row.map(v => (v-m)/(s+eps)) })
const of = out.flat()
const m = mean(of)
console.log(`після нормалізації — середнє: ${m.toFixed(4)} (сире ${m.toExponential(3)}), std: ${std(of).toFixed(4)}`)
console.log(`   (в оригіналі 0.0000 / 1.0000 — правильно, це випливає з математики)`)
