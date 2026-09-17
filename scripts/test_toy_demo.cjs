"use strict";
const assert = require("node:assert/strict");
const {makeTrial,COUNT,REPEATS,ranks,spearman} = require("../docs/toy-model.js");
function close(a,b) {assert.ok(Math.abs(a-b)<1e-10,`${a} != ${b}`);}
assert.deepEqual(ranks([2,1,2]),[2.5,1,2.5]);
close(spearman([1,2,3],[3,2,1]),-1);
assert.ok(Number.isNaN(spearman([1,1],[2,3])));
for(const seed of [17,18,19,89,181]) {
  const faithful=makeTrial(seed,"faithful"),misleading=makeTrial(seed,"misleading");
  assert.deepEqual(faithful.x,misleading.x);
  assert.deepEqual(faithful.randomOrders,misleading.randomOrders);
  close(faithful.correlation,1);
  assert.ok(misleading.correlation<faithful.correlation);
  assert.equal(faithful.randomOrders.length,REPEATS);
  for(const trial of [faithful,misleading]) {
    for(const order of [trial.top,trial.bottom,...trial.randomOrders]) {
      assert.equal(new Set(order).size,COUNT);
      close(trial.remaining(order,0),trial.baseline);
      close(trial.remaining(order,COUNT),-.8);
      close(trial.remaining(order,39),-.8+trial.effects.reduce((s,v,i)=>s+(order.slice(0,39).includes(i)?0:v),0));
    }
    for(let i=0;i<COUNT;i++) close(trial.baseline-trial.remaining([i],1),trial.effects[i]);
    close(trial.curves[COUNT].random,-.8);
    assert.equal(trial.curves.length,COUNT+1);
  }
  for(const p of faithful.curves) {
    assert.ok(p.top<=p.random+1e-10);
    assert.ok(p.random<=p.bottom+1e-10);
  }
  assert.ok(misleading.curves[39].top>misleading.curves[39].random);
}
console.log("PASS: exact interventions, complete deletion, tied ranks, shared random controls, faithful ordering, and misleading-map counterexample (five seeds).");
