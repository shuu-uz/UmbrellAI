"""Build a self-contained offline HTML demo from frozen analysis artifacts."""
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT.parents[1]/'outputs'

def finite_array(value, length):
    return isinstance(value,list) and len(value)==length and all(type(v) in (int,float) and math.isfinite(v) for v in value)

def payload():
    folder=ROOT/'results/final_20261123'
    m50=json.loads((folder/'intention_model.json').read_text(encoding='utf-8'))
    m60=json.loads((folder/'rain_score.json').read_text(encoding='utf-8'))
    metrics=json.loads((folder/'survey_metrics.json').read_text(encoding='utf-8'))
    if m50['feature_names']!=['outdoor_minutes','carry_discomfort','wet_discomfort'] or m50['target']!='stated_carry_intention':
        raise ValueError('POP50 model schema differs')
    if any(not finite_array(m50[k],3) for k in ['coefficients','mean','scale']) or any(s<=0 for s in m50['scale']):
        raise ValueError('Invalid POP50 coefficients/scaler')
    if m60['features']!=['travel_rain_forecast','carry_discomfort','wet_discomfort'] or m60['fixed_daily_pop']!=60 or not finite_array(m60['weights'],3):
        raise ValueError('POP60 model schema differs')
    for m in [m50,m60]:
        if not isinstance(m['intercept'],(float,int)) or not math.isfinite(m['intercept']):raise ValueError('Invalid intercept')
    windows=[]
    with (ROOT/'commute_results/windows.csv').open(encoding='utf-8-sig',newline='') as f:
        for row in csv.DictReader(f):
            import datetime
            datetime.date.fromisoformat(row['day'])
            peak,wet=int(row['peak_pop']),int(row['wet'])
            if row['window'] not in ['등교','하교'] or not 0<=peak<=100 or wet not in [0,1]:raise ValueError('Invalid case')
            windows.append(dict(day=row['day'],window=row['window'],peak_pop=peak,wet=wet))
    if len(windows)!=44 or sum(w['wet'] for w in windows)!=2 or len({(w['day'],w['window']) for w in windows})!=44:
        raise ValueError('Expected frozen 44 unique cases / two wet windows')
    for day in ['2026-10-13','2026-10-21']:
        if not any(w['day']==day and w['window']=='등교' for w in windows):raise ValueError('Default demo case missing')
    def convert(m):
        result={'brier':m['brier'],'auc':m.get('auc',m.get('roc_auc')),'accuracy':m['accuracy']}
        if any(type(v) not in (float,int) or not math.isfinite(v) or not 0<=v<=1 for v in result.values()):raise ValueError('Invalid metrics')
        return result
    try: revision=subprocess.check_output(['git','rev-parse','--short','HEAD'],cwd=ROOT,text=True).strip()
    except subprocess.CalledProcessError: raise ValueError('Source version unavailable')
    digest=hashlib.sha256((ROOT/'demo/template.html').read_bytes()+(ROOT/'demo/core.js').read_bytes()).hexdigest()[:10]
    sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [folder/'intention_model.json',folder/'rain_score.json',folder/'survey_metrics.json',ROOT/'commute_results/windows.csv']}
    return dict(model50=m50,model60=m60,windows=windows,metrics={
        '50':{'baseline':convert(metrics['models']['time_only']['oof']),'personalized':convert(metrics['models']['personalized']['oof'])},
        '60':{'baseline':convert(m60['validation']['weather_only']),'personalized':convert(m60['validation']['combined'])}},
        version=f'{revision} / demo {digest}',source_hashes=sources)

def build():
    data=payload()
    text=(ROOT/'demo/template.html').read_text(encoding='utf-8')
    if text.count('__PAYLOAD__')!=1 or text.count('__CORE__')!=1:raise ValueError('Template marker missing')
    encoded=json.dumps(data,ensure_ascii=False,allow_nan=False).replace('<','\\u003c')
    text=text.replace('__PAYLOAD__',encoded).replace('__CORE__',(ROOT/'demo/core.js').read_text(encoding='utf-8'))
    OUT.mkdir(exist_ok=True,parents=True)
    published=ROOT/'docs/demo';published.mkdir(exist_ok=True,parents=True)
    for p in [OUT/'UmbrellAI_demo.html',published/'index.html']:p.write_text(text,encoding='utf-8')
    (ROOT/'docs/.nojekyll').touch()
    print(OUT/'UmbrellAI_demo.html')

if __name__=='__main__':build()
