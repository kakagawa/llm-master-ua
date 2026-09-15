#!/usr/bin/env node
// Перевірка числових тверджень у docs/llm/transformer/fnn_code.md
import { NumpyRandom } from './numpy-rng.mjs'
const mul=(A,B)=>{const Bt=B[0].map((_,j)=>B.map(r=>r[j]));return A.map(r=>Bt.map(c=>r.reduce((s,v,i)=>s+v*c[i],0)))}

const r = new NumpyRandom(42)
const L=7, d_model=8, d_ff=32
const x  = r.randn(L, d_model)
const W1 = r.randn(d_model, d_ff)
const b1 = r.randn(1, d_ff)[0]
const hidden = mul(x, W1).map(row => row.map((v,j) => v + b1[j]))
const flat = hidden.flat()

console.log(`hidden: ${hidden.length}×${hidden[0].length} = ${flat.length} елементів`)
console.log(`від'ємних до ReLU: ${flat.filter(v=>v<0).length}   (в оригіналі було 112 — помилка)`)
console.log(`від'ємних після ReLU: ${flat.map(v=>Math.max(0,v)).filter(v=>v<0).length}`)

// співвідношення кількості параметрів
const attn = 4*d_model**2, ffn = 2*d_model*d_ff
console.log(`\nпараметри: attention ${attn}, FFN ${ffn}, відношення ${ffn/attn}× (у статті «вдвічі»)`)
console.log(`частка FFN: ${(ffn/(ffn+attn)*100).toFixed(1)}% (у статті «близько двох третин»)`)
