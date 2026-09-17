(function () {
  "use strict";
  const {GRID,COUNT,REPEATS,makeTrial,ranks}=window.ToyAudit;
  const $=id=>document.getElementById(id);
  let seed=17, mode="faithful", trial=makeTrial(seed,mode);
  const format=v=>Number.isFinite(v) ? v.toFixed(3) : "Undefined";
  function blend(a,b,t) {return `rgb(${a.map((v,i)=>Math.round(v+(b[i]-v)*t)).join(",")})`;}
  function drawGrid(id, kind, removed=new Set()) {
    const canvas=$(id),ctx=canvas.getContext("2d"),size=canvas.width/GRID;
    const rank=ranks(trial.attribution);
    ctx.clearRect(0,0,canvas.width,canvas.height);
    for(let i=0;i<COUNT;i++) {
      let color;
      if(kind==="attribution") color=blend([239,246,243],[8,127,117],(rank[i]-1)/(COUNT-1));
      else if(removed.has(i)) color="#fff";
      else if(trial.w[i]>0) color=blend([237,247,243],[8,127,117],trial.x[i]);
      else if(trial.w[i]<0) color=blend([255,239,234],[200,78,65],trial.x[i]);
      else color=blend([247,249,248],[169,184,178],trial.x[i]*2);
      const x=i%GRID*size,y=Math.floor(i/GRID)*size;
      ctx.fillStyle=color;ctx.fillRect(x,y,size,size);
      ctx.strokeStyle="#d6e1dc";ctx.lineWidth=.65;ctx.strokeRect(x+.35,y+.35,size-.7,size-.7);
      if(removed.has(i)){ctx.strokeStyle="#b7c7c0";ctx.beginPath();ctx.moveTo(x+size*.35,y+size*.5);ctx.lineTo(x+size*.65,y+size*.5);ctx.stroke();}
    }
  }
  function drawChart(n) {
    const canvas=$("deletion-chart"),ratio=window.devicePixelRatio||1;
    const width=canvas.clientWidth,height=canvas.clientHeight;
    canvas.width=Math.round(width*ratio);canvas.height=Math.round(height*ratio);
    const ctx=canvas.getContext("2d");ctx.scale(ratio,ratio);
    const left=44,right=12,top=23,bottom=36,w=width-left-right,h=height-top-bottom;
    const values=trial.curves.flatMap(p=>[p.top,p.random,p.bottom]);
    const lo=Math.floor(Math.min(...values)*2)/2-.15,hi=Math.ceil(Math.max(...values)*2)/2+.15;
    const px=f=>left+f*w,py=v=>top+(hi-v)/(hi-lo)*h;
    ctx.font="11px -apple-system, sans-serif";ctx.lineWidth=1;
    for(let i=0;i<=4;i++){const v=lo+(hi-lo)*i/4,y=py(v);ctx.strokeStyle="#e5ece8";ctx.beginPath();ctx.moveTo(left,y);ctx.lineTo(width-right,y);ctx.stroke();ctx.fillStyle="#576968";ctx.textAlign="right";ctx.fillText(v.toFixed(1),left-8,y+4);}
    for(let i=0;i<=4;i++){ctx.textAlign="center";ctx.fillText(`${i*25}%`,px(i/4),height-18);}
    ctx.textAlign="left";ctx.fillText("Target logit",left,12);
    ctx.textAlign="right";ctx.fillText("Patches removed",width-right,height-1);
    [["bottom","#3773bd"],["random","#a37017"],["top","#087f75"]].forEach(([key,color])=>{ctx.beginPath();ctx.strokeStyle=color;ctx.lineWidth=2.3;ctx.setLineDash(key==="random"?[5,4]:[]);trial.curves.forEach((p,i)=>{if(i)ctx.lineTo(px(p.fraction),py(p[key]));else ctx.moveTo(px(p.fraction),py(p[key]));});ctx.stroke();});
    ctx.setLineDash([3,4]);ctx.strokeStyle="#70837c";ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(px(n/COUNT),top);ctx.lineTo(px(n/COUNT),top+h);ctx.stroke();ctx.setLineDash([]);
  }
  function render() {
    const percent=Number($("fraction").value),n=Math.round(COUNT*percent/100),strategy=$("strategy").value;
    const order=strategy==="random"?trial.randomOrders[0]:trial[strategy];
    const remaining=trial.remaining(order,n);
    $("fraction-label").textContent=`${percent}%`;
    $("removed-count").textContent=`${n} / ${COUNT}`;
    $("map-name").textContent=mode==="faithful"?"Faithful":"Misleading";
    $("baseline").textContent=format(trial.baseline);
    $("remaining").textContent=format(remaining);
    $("drop").textContent=format(trial.baseline-remaining);
    $("correlation").textContent=format(trial.correlation);
    $("preview-note").textContent=strategy==="random"?"First seeded random draw; chart shows the mean of 20 draws.":`${strategy==="top"?"Highest":"Lowest"}-ranked patches replaced with zero.`;
    const advantage=trial.curves[n].random-trial.curves[n].top;
    $("interpretation").textContent=Math.abs(advantage)<1e-10?"Top and random deletion have the same mean effect at this step.":`At this step, top deletion is ${advantage>0?"more":"less"} damaging than random by ${format(Math.abs(advantage))} logit units.`;
    drawGrid("input-map","input");drawGrid("attribution-map","attribution");drawGrid("deleted-map","input",new Set(order.slice(0,n)));drawChart(n);
    $("deleted-map").setAttribute("aria-label",`${strategy} deletion of ${n} of ${COUNT} patches; remaining toy target logit ${format(remaining)}`);
  }
  document.querySelectorAll('input[name="map"]').forEach(input=>input.addEventListener("change",()=>{mode=input.value;trial=makeTrial(seed,mode);render();}));
  $("fraction").addEventListener("input",render);$("strategy").addEventListener("change",render);
  $("new-trial").addEventListener("click",()=>{seed++;trial=makeTrial(seed,mode);render();});
  $("download").addEventListener("click",()=>{
    const lines=["experiment,seed,map,target,patches_removed,fraction_removed,baseline_logit,top_remaining_logit,random_mean_remaining_logit,bottom_remaining_logit,random_repeats"];
    trial.curves.forEach(p=>lines.push(["synthetic_additive_toy",seed,mode,"toy_target",p.n,p.fraction,trial.baseline,p.top,p.random,p.bottom,REPEATS].join(",")));
    const url=URL.createObjectURL(new Blob([lines.join("\n")+"\n"],{type:"text/csv"}));
    const a=document.createElement("a");a.href=url;a.download=`toy-deletion-${mode}-seed-${seed}.csv`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  });
  window.addEventListener("resize",render);render();
})();
