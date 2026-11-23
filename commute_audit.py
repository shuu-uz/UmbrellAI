"""Reproduce the retrospective commute policy audit from the three final Notion downloads."""
from pathlib import Path
from zipfile import ZipFile
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss,roc_auc_score

d=Path.home()/'Downloads'
out=Path(__file__).resolve().parent/'commute_results'
out.mkdir(exist_ok=True)
def read_member(pattern, suffix):
    with ZipFile(next(d.glob(pattern))) as z:
        return pd.read_csv(z.open(next(n for n in z.namelist() if n.endswith(suffix))))
with ZipFile(next(d.glob('*최종 예보 데이터.zip'))) as z:
    f=pd.concat([pd.read_csv(z.open(n)) for n in z.namelist() if n.startswith('data/forecast_clean_') and n.endswith('.csv')])
m=read_member('*실측 병합 데이터.zip','/merged_forecast_actual.csv')
keys=['baseDate','baseTime','date','time','nx','ny','POP']
m=m.merge(f.rename(columns={'base_date':'baseDate','base_time':'baseTime'})[keys+['collected_at']],on=keys,validate='one_to_one')
assert len(m)==len(f)==1025
def timestamp(day,hour):
    return pd.to_datetime(day.astype(str)+hour.astype(str).str.zfill(4),format='%Y%m%d%H%M')
m['target']=timestamp(m.date,m.time)
m['issued']=timestamp(m.baseDate,m.baseTime)
m['collected']=pd.to_datetime(m.collected_at,utc=True).dt.tz_convert('Asia/Seoul').dt.tz_localize(None)
def usable(rows,deadline):
    rows=rows[(rows.issued<=deadline)&(rows.collected<=deadline)]
    return rows.sort_values('issued').drop_duplicates('target',keep='last')
def observed(rows):
    return rows[(rows.rain_missing==0)&rows.real_rain.notna()&(rows.real_rain>=0)]
h=observed(usable(m,m.target-pd.Timedelta(hours=1))).copy()
h['wet']=h.real_rain>0
h.groupby('POP').wet.agg(['size','sum','mean']).to_csv(out/'hourly_pop.csv')
print('HOURS',len(h),h.wet.sum(),h.loc[h.wet,'date'].nunique())
# Interval-ending timestamps: 08–09 uses 09:00; 17–19 uses 18:00 and 19:00.
windows=[]
for day in pd.date_range(m.target.min().normalize(),m.target.max().normalize()):
    if day.weekday()>=5: continue
    for label,start,end in [('등교',8,9),('하교',17,19)]:
        departure=day+pd.Timedelta(hours=start)
        hours=pd.date_range(departure+pd.Timedelta(hours=1),day+pd.Timedelta(hours=end),freq='h')
        rows=observed(usable(m[m.target.isin(hours)],departure))
        if len(rows)==len(hours):
            windows.append((day,label,rows.POP.max(),int(rows.real_rain.gt(0).any())))
w=pd.DataFrame(windows,columns=['day','window','peak_pop','wet'])
w.to_csv(out/'windows.csv',index=False,encoding='utf-8-sig')
print(w.groupby('window').wet.agg(['size','sum']), 'days',w.day.nunique(),'wet days',w[w.wet==1].day.nunique())
s=pd.read_csv(next(d.glob('*설문지 응답 시트1.csv')))
q=lambda i:s.iloc[:,i]
t=q(7).str.extract(r'(\d+)')[0].astype(float)
assert q(7)[t.isna()].str.contains('무조건').all()
t=t.fillna(0)
pref=np.sign(q(6)-q(5))
# Illustrative burden ratio assumes two Likert scores can be treated as comparable costs.
ratio=100*q(5)/(q(5)+q(6))
policies={'공통60':np.full(len(s),60),'직접응답':t,'불편도비율':ratio}
outrows=[]
for name,thresholds in policies.items():
    for group,mask in [('전체',np.ones(len(s),bool)),('우산짐',pref<0),('같음',pref==0),('비맞음',pref>0)]:
        th=np.asarray(thresholds)[mask]
        carry=w.peak_pop.to_numpy()[None,:]>=th[:,None]
        wet=w.wet.to_numpy()[None,:].astype(bool)
        fp=(carry&~wet).sum(axis=1);fn=(~carry&wet).sum(axis=1)
        outrows.append([name,group,len(th),len(w),np.median(th),fp.mean(),fn.mean()])
p=pd.DataFrame(outrows,columns=['policy','group','people','windows','median_threshold','mean_false_alarm','mean_miss'])
p.to_csv(out/'policies.csv',index=False,encoding='utf-8-sig')
print(p.round(2).to_string(index=False))
for cutoff in [30,50,60,70]:
    take=w.peak_pop>=cutoff
    print('cutoff',cutoff,'fp',sum(take&(w.wet==0)),'fn',sum(~take&(w.wet==1)))
rng=np.random.default_rng(42)
ix=rng.integers(0,len(s),size=(5000,len(s)))
for a,b in [(1,2),(3,4)]:
    for undecided in [0,.5,1]:
        trans={'챙긴다':1,'고민된다':undecided,'챙기지 않는다':0}
        delta=(q(b).map(trans)-q(a).map(trans)).to_numpy()
        print('paired',a,undecided,delta.mean(),np.quantile(delta[ix].mean(axis=1),[.025,.975]))
# Paired OOF differences. Resample people (not predictions independently); conditional on fitted folds.
for a,b in [(1,2),(3,4)]:
    group=np.tile(np.arange(len(s)),2)
    condition=np.repeat([0,1],len(s))
    x=np.column_stack([condition,np.tile(q(5),2),np.tile(q(6),2)])
    y=np.r_[q(a),q(b)]=='챙긴다'
    preds=[]
    for count in [1,3]:
        pred=np.zeros(len(y))
        for tr,te in GroupKFold(5).split(x,y,group):
            model=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000))
            model.fit(x[tr,:count],y[tr])
            pred[te]=model.predict_proba(x[te,:count])[:,1]
        preds.append(pred)
    dif=((preds[1]-y)**2-(preds[0]-y)**2).reshape(2,-1).mean(axis=0)
    print('paired Brier change',a,dif.mean(),np.quantile(dif[ix].mean(axis=1),[.025,.975]))
