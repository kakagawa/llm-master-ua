#!/usr/bin/env node
// Перевірка числових виходів у docs/llm/transformer/attention_code.md
import { NumpyRandom } from './numpy-rng.mjs'
const T = m => m[0].map((_,j)=>m.map(r=>r[j]))
const mul = (A,B) => { const Bt=T(B); return A.map(r=>Bt.map(c=>r.reduce((s,v,i)=>s+v*c[i],0))) }
const soft = M => M.map(r=>{const mx=Math.max(...r);const e=r.map(v=>Math.exp(v-mx));const s=e.reduce((a,b)=>a+b,0);return e.map(v=>v/s)})
const f8 = m => m.map(r=>r.map(v=>v.toFixed(8)))

const r = new NumpyRandom(42)
const L=7, d=4
const X   = r.randn(L,d)
const W_Q = r.randn(d,d), W_K = r.randn(d,d), W_V = r.randn(d,d)
const Q = mul(X,W_Q), K = mul(X,W_K), V = mul(X,W_V)

const scores = mul(Q, T(K))
console.log('attention_scores[:3,:3] (реально):')
f8(scores).slice(0,3).forEach(r=>console.log(' ', r.slice(0,3).join(' ')))
console.log('у статті: [-2.16484311 -1.22337089 -0.61920408] / [-1.0635925 -3.80432623 0.21339416] / [-0.86007595 1.31510269 -1.63627839]')

const flat = scores.flat()
console.log('\nдіапазон до масштабування (реально): %s .. %s', Math.min(...flat).toFixed(2), Math.max(...flat).toFixed(2))
console.log('у статті: -4.23 .. 3.87')
const sc = scores.map(r=>r.map(v=>v/Math.sqrt(d)))
const scf = sc.flat()
console.log('діапазон після масштабування (реально): %s .. %s', Math.min(...scf).toFixed(2), Math.max(...scf).toFixed(2))
console.log('у статті: -2.12 .. 1.94')

const W = soft(sc)
console.log('\nattention_weights[0] (реально):')
console.log(' ', W[0].map(v=>v.toFixed(8)).join(' '))
console.log('у статті: [0.08734234 0.21548765 0.13987621 0.18745632 0.15234987 0.11234876 0.10513885]')
console.log('сума рядків:', W.map(r=>r.reduce((a,b)=>a+b,0).toFixed(6)).join(' '))

const out = mul(W,V)
console.log('\noutput[:3] (реально):')
f8(out).slice(0,3).forEach(r=>console.log(' ', r.join(' ')))
console.log('у статті: [-0.39482749 0.23487123 -0.18745632 0.45123876] / [0.18234987 -0.28745632 0.39487123 -0.12348765] / [-0.28745632 0.39482749 -0.18234987 0.28745632]')
