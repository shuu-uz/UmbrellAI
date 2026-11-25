(function (root) {
  'use strict';
  const integer = (v, lo, hi) => Number.isInteger(v) && v >= lo && v <= hi;
  function score(data, condition, exposure, carry, wet) {
    if (!integer(carry,1,5) || !integer(wet,1,5)) throw Error('불편도는 정수 1~5점이어야 합니다.');
    let z;
    if (condition === '50') {
      if (![10,60].includes(exposure)) throw Error('POP 50%에서는 이동 10분·60분만 지원합니다.');
      const m = data.model50, x = [exposure,carry,wet];
      z = m.intercept + x.reduce((s,v,i) => s + m.coefficients[i]*(v-m.mean[i])/m.scale[i],0);
    } else if (condition === '60') {
      if (![0,1].includes(exposure)) throw Error('이동 시간대 비 예보는 없음·있음만 지원합니다.');
      const m = data.model60;
      z = m.intercept + [exposure,carry,wet].reduce((s,v,i) => s+m.weights[i]*v,0);
    } else throw Error('지원하지 않는 설문 조건입니다.');
    return 100*(z>=0 ? 1/(1+Math.exp(-z)) : Math.exp(z)/(1+Math.exp(z)));
  }
  function recommend(peak, threshold) {
    if (!Number.isFinite(peak) || peak<0 || peak>100) throw Error('유효하지 않은 POP');
    if (threshold==='always') return true;
    if (!integer(threshold,20,90) || threshold%10) throw Error('유효하지 않은 지참 기준');
    return peak>=threshold;
  }
  function outcome(take, wet) {
    if (![0,1].includes(wet)) throw Error('실측은 0 또는 1이어야 합니다.');
    return take ? (wet ? '강수 적중' : '헛경보') : (wet ? '놓침' : '무강수 적중');
  }
  function counts(windows, threshold) {
    return windows.reduce((s,w) => {
      const take = recommend(w.peak_pop,threshold);
      if(take&&!w.wet) s.falseAlarm++;
      if(!take&&w.wet) s.miss++;
      if(take) s.take++;
      return s;
    },{falseAlarm:0,miss:0,take:0});
  }
  const api={score,recommend,outcome,counts};
  if(typeof module!=='undefined'&&module.exports) module.exports=api;
  else root.UmbrellAI=api;
})(typeof globalThis!=='undefined'?globalThis:this);
