// Відтворення numpy.random.RandomState(seed).randn() — MT19937 + polar-метод,
// як у legacy-генераторі numpy. Потрібне, щоб перевіряти числові приклади в статтях
// без встановленого numpy.
export class NumpyRandom {
  constructor(seed) {
    this.mt = new Uint32Array(624); this.idx = 625
    this.hasGauss = false; this.gauss = 0
    this.mt[0] = seed >>> 0
    for (let i = 1; i < 624; i++) {
      const s = this.mt[i-1] ^ (this.mt[i-1] >>> 30)
      // 1812433253 * s + i, у 32-бітній арифметиці
      const lo = (s & 0xffff) * 1812433253
      const hi = ((s >>> 16) * 1812433253) & 0xffff
      this.mt[i] = (((hi << 16) >>> 0) + lo + i) >>> 0
    }
    this.idx = 624
  }
  _gen() {
    if (this.idx >= 624) {
      for (let i = 0; i < 624; i++) {
        const y = ((this.mt[i] & 0x80000000) | (this.mt[(i+1)%624] & 0x7fffffff)) >>> 0
        let n = (this.mt[(i+397)%624] ^ (y >>> 1)) >>> 0
        if (y & 1) n = (n ^ 0x9908b0df) >>> 0
        this.mt[i] = n
      }
      this.idx = 0
    }
    let y = this.mt[this.idx++]
    y = (y ^ (y >>> 11)) >>> 0
    y = (y ^ ((y << 7) & 0x9d2c5680)) >>> 0
    y = (y ^ ((y << 15) & 0xefc60000)) >>> 0
    y = (y ^ (y >>> 18)) >>> 0
    return y
  }
  random() { // rk_double
    const a = this._gen() >>> 5, b = this._gen() >>> 6
    return (a * 67108864.0 + b) / 9007199254740992.0
  }
  gaussVal() { // rk_gauss
    if (this.hasGauss) { this.hasGauss = false; return this.gauss }
    let x1, x2, r2
    do { x1 = 2*this.random()-1; x2 = 2*this.random()-1; r2 = x1*x1 + x2*x2 }
    while (r2 >= 1.0 || r2 === 0.0)
    const f = Math.sqrt(-2.0*Math.log(r2)/r2)
    this.gauss = f*x1; this.hasGauss = true
    return f*x2
  }
  randn(rows, cols) {
    return Array.from({length: rows}, () => Array.from({length: cols}, () => this.gaussVal()))
  }
}
