from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

folder = Path(__file__).resolve().parent
windows = pd.read_csv(folder / "windows.csv")
groups = pd.read_csv(folder / "policies.csv").query("policy == '직접응답' and group != '전체'")
plt.rcParams.update({"font.family":"Malgun Gothic","font.size":11,"axes.unicode_minus":False})
fig,(ax,bx)=plt.subplots(2,1,figsize=(9,7.7),layout="constrained")
x=np.arange(len(windows))
for wet,color,label in [(0,"#70899b","실측 무강수"),(1,"#006758","실측 강수")]:
    match=windows.wet.eq(wet)
    ax.scatter(x[match],windows.peak_pop[match],s=52 if wet else 28,color=color,label=label,zorder=4)
for threshold,color in [(50,"#a15c04"),(70,"#683980")]:
    ax.axhline(threshold,color=color,ls="--",lw=1.4,label=f"개인 기준 {threshold}%")
ax.set(ylim=(-6,86),ylabel="구간 최고 강수확률 (%)",xlabel="평가한 등·하교 구간 (시간순)",title="① 예보 점이 개인 기준선 이상이면 우산을 권고")
ticks=[0,11,21,32,43]
ax.set_xticks(ticks,[windows.day[i][5:]+" "+windows.window[i] for i in ticks])
ax.legend(ncol=2,loc="upper left",fontsize=9)
y=np.arange(len(groups))
for offset,col,color,label in [(-.18,"mean_false_alarm","#386888","비가 없었지만 권고"),(.18,"mean_miss","#b76532","비가 왔지만 권고 안 함")]:
    values=groups[col]
    bars=bx.barh(y+offset,values,height=.32,color=color,label=label)
    bx.bar_label(bars,fmt="%.1f",padding=4)
bx.set_yticks(y,[f"{g} (n={n}, 기준 중앙값 {t:.0f}%)" for g,n,t in zip(["우산 짐이 더 불편","두 불편도가 같음","비 맞음이 더 불편"],groups.people,groups.median_threshold)])
bx.invert_yaxis()
bx.set(xlim=(0,7.7),xlabel="응답자 한 명당 44구간에서의 평균 횟수",title="② 각 응답자가 직접 고른 기준으로 권고한 결과")
bx.legend(loc="lower right",fontsize=9)
for a in [ax,bx]:
    a.spines[["top","right"]].set_visible(False)
    a.grid(alpha=.17)
    a.set_axisbelow(True)
fig.savefig(folder/"commute_policy.png",dpi=155)
