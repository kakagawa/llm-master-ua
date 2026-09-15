#!/usr/bin/env node
// Повний прогін Tiny Transformer з docs/llm/transformer/tiny_transformer_code.md
import { NumpyRandom } from './numpy-rng.mjs'
const T   = m => m[0].map((_,j)=>m.map(r=>r[j]))
const mul = (A,B) => { const Bt=T(B); return A.map(r=>Bt.map(c=>r.reduce((s,v,i)=>s+v*c[i],0))) }
const add = (A,B) => A.map((r,i)=>r.map((v,j)=>v+B[i][j]))
const addVec = (A,v) => A.map(r=>r.map((x,j)=>x+v[j]))
const mean = a => a.reduce((s,v)=>s+v,0)/a.length
const std  = a => { const m=mean(a); return Math.sqrt(mean(a.map(v=>(v-m)**2))) }

const r = new NumpyRandom(42)
const vocab_size=10, L=4, d_model=8, d_ff=32, num_layers=2
const input_ids=[3,8,2,6]

const embedding_table = r.randn(vocab_size, d_model)   // 80 значень
let x = input_ids.map(id => embedding_table[id].slice())

// positional encoding
const pe = Array.from({length:L},(_,pos)=>{
  const row = new Array(d_model).fill(0)
  for (let i=0;i<d_model;i+=2){
    row[i] = Math.sin(pos / Math.pow(10000, i/d_model))
    if (i+1 < d_model) row[i+1] = Math.cos(pos / Math.pow(10000, i/d_model))
  }
  return row
})
x = add(x, pe)

// два блоки: ваги тягнуться з того самого потоку в порядку конструктора
const blocks = []
for (let b=0;b<num_layers;b++){
  blocks.push({
    Wq: r.randn(d_model,d_model).map(row=>row.map(v=>v*0.01)),
    Wk: r.randn(d_model,d_model).map(row=>row.map(v=>v*0.01)),
    Wv: r.randn(d_model,d_model).map(row=>row.map(v=>v*0.01)),
    W1: r.randn(d_model,d_ff).map(row=>row.map(v=>v*0.01)),
    b1: new Array(d_ff).fill(0),
    W2: r.randn(d_ff,d_model).map(row=>row.map(v=>v*0.01)),
    b2: new Array(d_model).fill(0),
  })
}
const layerNorm = (m,eps=1e-5) => m.map(row=>{const mu=mean(row),s=std(row);return row.map(v=>(v-mu)/(s+eps))})
const attention = (m,B) => {
  const Q=mul(m,B.Wq), K=mul(m,B.Wk), V=mul(m,B.Wv)
  const sc=mul(Q,T(K)).map(row=>row.map(v=>v/Math.sqrt(m[0].length)))
  const W=sc.map(row=>{const e=row.map(Math.exp),s=e.reduce((a,b)=>a+b,0);return e.map(v=>v/s)})
  return mul(W,V)
}
const ffn = (m,B) => {
  const h = addVec(mul(m,B.W1), B.b1).map(row=>row.map(v=>Math.max(0,v)))
  return addVec(mul(h,B.W2), B.b2)
}
for (const B of blocks){
  x = layerNorm(add(x, attention(x,B)))
  x = layerNorm(add(x, ffn(x,B)))
}

const W_out = r.randn(d_model, vocab_size).map(row=>row.map(v=>v*0.01))
const b_out = new Array(vocab_size).fill(0)
const logits = addVec(mul(x, W_out), b_out)
const argmax = logits.map(row => row.indexOf(Math.max(...row)))

console.log('logits shape:', `(${logits.length}, ${logits[0].length})`)
console.log('передбачений токен на кожній позиції:', '[' + argmax.join(' ') + ']')
console.log('у статті                            : [5 5 4 3]')
