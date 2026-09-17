/* Pure, deterministic toy experiment. No pathology weights or observations. */
(function (root) {
  "use strict";
  const GRID = 14, COUNT = GRID * GRID, REPEATS = 20;
  function rng(seed) {
    let state = seed >>> 0;
    return () => {state = (Math.imul(1664525, state) + 1013904223) >>> 0; return state / 4294967296;};
  }
  function shuffled(seed) {
    const a = Array.from({length: COUNT}, (_, i) => i), random = rng(seed);
    for (let i = a.length - 1; i > 0; i--) {const j = Math.floor(random() * (i + 1)); [a[i], a[j]] = [a[j], a[i]];}
    return a;
  }
  function ranks(values) {
    const order = values.map((_, i) => i).sort((a, b) => values[a] - values[b] || a - b);
    const result = Array(values.length);
    for (let i = 0; i < order.length;) {
      let j = i + 1;
      while (j < order.length && values[order[j]] === values[order[i]]) j++;
      for (let k = i; k < j; k++) result[order[k]] = (i + j - 1) / 2 + 1;
      i = j;
    }
    return result;
  }
  function spearman(a, b) {
    const x = ranks(a), y = ranks(b), mx = x.reduce((s,v) => s+v,0)/x.length, my = y.reduce((s,v) => s+v,0)/y.length;
    let xy=0, xx=0, yy=0;
    x.forEach((v,i) => {xy+=(v-mx)*(y[i]-my);xx+=(v-mx)**2;yy+=(y[i]-my)**2;});
    return xx && yy ? xy/Math.sqrt(xx*yy) : NaN;
  }
  function makeTrial(seed=17, mode="faithful") {
    const random=rng(seed), x=[], w=[], misleading=[];
    const cx=4+(seed%3), cy=6+(seed%2);
    for(let i=0;i<COUNT;i++) {
      const row=Math.floor(i/GRID), col=i%GRID;
      const supportive=(col-cx)**2+(row-cy)**2<=10;
      const inhibitory=!supportive && col>=10 && row>=8 && row<=11;
      x.push(supportive ? .78+random()*.22 : inhibitory ? .65+random()*.25 : .08+random()*.12);
      w.push(supportive ? .12 : inhibitory ? -.045 : 0);
      misleading.push(Math.exp(-((col-11)**2+(row-2)**2)/10)+random()*.001);
    }
    const effects=x.map((v,i)=>v*w[i]);
    const attribution=mode==="faithful" ? effects.slice() : misleading;
    const baseline=-.8+effects.reduce((a,b)=>a+b,0);
    const indices=Array.from({length:COUNT},(_,i)=>i);
    const top=indices.slice().sort((a,b)=>attribution[b]-attribution[a]||a-b);
    const bottom=indices.slice().sort((a,b)=>attribution[a]-attribution[b]||a-b);
    const randomOrders=Array.from({length:REPEATS},(_,r)=>shuffled(seed*1009+r*7919));
    const remaining=(order,n)=>baseline-order.slice(0,n).reduce((s,i)=>s+effects[i],0);
    const curves=Array.from({length:COUNT+1},(_,n)=>({n,fraction:n/COUNT,top:remaining(top,n),bottom:remaining(bottom,n),random:randomOrders.reduce((s,o)=>s+remaining(o,n),0)/REPEATS}));
    return {seed,mode,x,w,effects,attribution,baseline,top,bottom,randomOrders,remaining,curves,correlation:spearman(attribution,effects)};
  }
  const api={GRID,COUNT,REPEATS,rng,ranks,spearman,makeTrial};
  if(typeof module!=="undefined" && module.exports) module.exports=api;
  else root.ToyAudit=api;
})(typeof globalThis!=="undefined" ? globalThis : this);
