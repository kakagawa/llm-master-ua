#!/usr/bin/env node
// Незалежний перерахунок прикладу з docs/llm/transformer/qkv_cal.md
// X = [я, люблю, тебе], Q = K = V = X, d_k = 2
const X = [[1,0],[0,1],[1,1]]
const T = m => m[0].map((_,j)=>m.map(r=>r[j]))
const mul = (A,B) => A.map(r => T(B).map(c => r.reduce((s,v,i)=>s+v*c[i],0)))
const r2 = x => Math.round(x*100)/100

const S = mul(X, T(X))
console.log('QKᵀ:              ', JSON.stringify(S))

const dk = Math.sqrt(2)
const Sc = S.map(r => r.map(v => v/dk))
console.log('після / √d_k:     ', JSON.stringify(Sc.map(r=>r.map(r2))))

const soft = r => { const e = r.map(Math.exp), s = e.reduce((a,b)=>a+b,0); return e.map(v=>v/s) }
const A = Sc.map(soft)
console.log('softmax (точно):  ', JSON.stringify(A.map(r=>r.map(v=>Math.round(v*10000)/10000))))
console.log('softmax (2 знаки):', JSON.stringify(A.map(r=>r.map(r2))))
console.log('суми рядків:      ', JSON.stringify(A.map(r=>r2(r.reduce((a,b)=>a+b,0)))))

const O = mul(A, X)
console.log('вихід A·V:        ', JSON.stringify(O.map(r=>r.map(r2))))
console.log()
console.log('у статті рядок «тебе»: [0.27, 0.27, 0.46], вихід [0.73, 0.73]')
console.log('правильно:             [%s], вихід [%s]',
  A[2].map(r2).join(', '), O[2].map(r2).join(', '))
