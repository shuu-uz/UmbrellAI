from pathlib import Path
import argparse
import json
import pandas as pd
from scipy import stats

parser = argparse.ArgumentParser(description='우산 설문 전처리 및 대응표본 분석')
parser.add_argument('csv', type=Path, help='원본 설문 CSV 경로')
args = parser.parse_args()
out = Path(__file__).resolve().parent
raw = pd.read_csv(args.csv, encoding='utf-8-sig')
names = ['응답시각', '이동10분', '이동60분', '이동시간비없음', '이동시간비있음',
         '우산짐불편도', '비맞음불편도', '강수확률기준', '이슬비지참',
         '중요도_강수확률', '중요도_강수량', '중요도_강수시간대',
         '중요도_야외이동시간', '중요도_지역날씨', '이동수단', '사용의향']
if raw.shape[1] != len(names):
    raise ValueError('예상한 16개 열과 다릅니다. 문항 구조를 확인하세요.')
pd.DataFrame({'원본열':raw.columns, '변경열':names}).to_csv(out/'column_mapping.csv', index=False, encoding='utf-8-sig')
df = raw.copy()
df.columns = names
for c in df.select_dtypes(include='object'):
    df[c] = df[c].str.strip().replace('', pd.NA)
missing = df.isna().sum().to_dict()
for c in ['우산짐불편도', '비맞음불편도']:
    df[c] = pd.to_numeric(df[c], errors='coerce')
    df.loc[~df[c].isin([1,2,3,4,5]), c] = float('nan')
choices = {'챙긴다', '챙기지 않는다', '고민된다'}
for c in ['이동10분','이동60분','이동시간비없음','이동시간비있음','이슬비지참']:
    df.loc[~df[c].isin(choices), c] = pd.NA
importance = {'전혀 중요하지 않다':1,'중요하지 않다':2,'보통이다':3,'중요하다':4,'매우 중요하다':5}
for c in names[9:14]:
    df[c] = df[c].map(importance)
# 결측치는 일괄 삭제하지 않고 분석별 필요한 열에서만 제외합니다.
df.to_csv(out/'cleaned_responses.csv', index=False, encoding='utf-8-sig')
pair = df[['우산짐불편도','비맞음불편도']].dropna()
a, b = pair['우산짐불편도'], pair['비맞음불편도']
diff = b-a
t = stats.ttest_rel(b, a)
w = stats.wilcoxon(b, a, zero_method='wilcox', alternative='two-sided', method='auto')
ci = stats.t.interval(0.95, len(pair)-1, loc=diff.mean(), scale=stats.sem(diff))
summary = {
    '원본응답수':len(raw), '원본결측치':missing,
    '전처리후결측치':df.isna().sum().to_dict(),
    'T검정유효응답수':len(pair), 'T검정제외응답수':len(df)-len(pair),
    '우산짐평균':a.mean(), '비맞음평균':b.mean(),
    '평균차이_비맞음빼기우산짐':diff.mean(),
    '차이95퍼센트신뢰구간':list(ci), 't값':float(t.statistic),
    '자유도':len(pair)-1, 'p값':float(t.pvalue),
    'Wilcoxon통계량':float(w.statistic), 'Wilcoxon_p값':float(w.pvalue),
    '단순비용모형기준확률':a.mean()/(a.mean()+b.mean()),
    '기준확률주의':'1~5점 서열척도를 비율 비용으로 간주한 탐색적 값이며 검증된 추천 기준이 아님',
    '범주화':'실제 체류 시간 미수집으로 15분 단위 응답자 범주화 불가'
}
(out/'statistics.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
pd.DataFrame([
    {'항목':c, '유효응답수':df[c].notna().sum(), '평균':df[c].mean(), '표준편차':df[c].std(ddof=1)}
    for c in ['우산짐불편도','비맞음불편도']+names[9:14]
]).to_csv(out/'descriptive_statistics.csv',index=False,encoding='utf-8-sig')
print(json.dumps(summary,ensure_ascii=False,indent=2))

# 같은 사람의 응답 변화: 챙긴다 vs 그 외로 명시적으로 이분화합니다.
comparisons = []
for x, y in [('이동10분','이동60분'),('이동시간비없음','이동시간비있음')]:
    valid = df[[x,y]].dropna()
    ct = pd.crosstab(valid[x],valid[y]).reindex(
        index=['챙긴다','챙기지 않는다','고민된다'],
        columns=['챙긴다','챙기지 않는다','고민된다'],fill_value=0)
    ct.to_csv(out/f'transition_{x}.csv',encoding='utf-8-sig')
    up = int(((valid[x]!='챙긴다')&(valid[y]=='챙긴다')).sum())
    down = int(((valid[x]=='챙긴다')&(valid[y]!='챙긴다')).sum())
    p = stats.binomtest(up,up+down,0.5).pvalue if up+down else 1.0
    comparisons.append({'비교':f'{x} → {y}','유효응답수':len(valid),
        '전조건지참비율':(valid[x]=='챙긴다').mean(),
        '후조건지참비율':(valid[y]=='챙긴다').mean(),
        '챙긴다로전환':up,'그외로전환':down,'정확McNemar_p':p})
# 두 시나리오 비교에 Holm 보정을 적용합니다.
ranked = sorted(range(len(comparisons)),key=lambda i:comparisons[i]['정확McNemar_p'])
previous = 0.0
for rank, i in enumerate(ranked):
    adjusted = min(1.0,max(previous,comparisons[i]['정확McNemar_p']*(len(ranked)-rank)))
    comparisons[i]['Holm보정_p'] = adjusted
    previous = adjusted
pd.DataFrame(comparisons).to_csv(out/'scenario_tests.csv',index=False,encoding='utf-8-sig')

group_table = df.groupby('이동수단')[['우산짐불편도','비맞음불편도']].agg(['count','mean','std'])
group_table.to_csv(out/'transport_summary.csv',encoding='utf-8-sig')
group_tests=[]
for c in ['우산짐불편도','비맞음불편도']:
    groups=[v[c].dropna().values for k,v in df.groupby('이동수단') if k!='기타']
    kw=stats.kruskal(*groups)
    an=stats.f_oneway(*groups)
    group_tests.append({'불편도':c,'Kruskal_H':float(kw.statistic),'Kruskal_p':float(kw.pvalue),
        '일원분산분석_F':float(an.statistic),'일원분산분석_p':float(an.pvalue),
        '주의':'기타 1명 제외. 서열척도에 대한 Kruskal–Wallis를 주 분석으로 해석. 분산분석은 참고.'})
pd.DataFrame(group_tests).to_csv(out/'transport_tests.csv',index=False,encoding='utf-8-sig')

q7=df['강수확률기준'].value_counts().reindex(
    ['20%','30%','40%','50%','60%','70%','80%','90% 이상','강수확률과 관계없이 무조건 챙긴다'],fill_value=0)
pd.DataFrame({'응답수':q7,'비율':q7/len(df)}).to_csv(out/'threshold_distribution.csv',encoding='utf-8-sig')
records=[]
for c in ['이동10분','이동60분','이동시간비없음','이동시간비있음','이슬비지참','사용의향']:
    for value,n in df[c].value_counts().items():
        records.append({'문항':c,'응답':value,'응답수':n,'비율':n/df[c].notna().sum()})
pd.DataFrame(records).to_csv(out/'response_distribution.csv',index=False,encoding='utf-8-sig')
