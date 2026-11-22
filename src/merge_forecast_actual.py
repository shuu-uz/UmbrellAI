import pandas as pd
from pathlib import Path

DATA_DIR = Path("./data")
FORECASTS = sorted(DATA_DIR.glob("forecast_clean_*.csv"))
ACTUAL = DATA_DIR / "real_rain_all.csv"
OUT = DATA_DIR / "merged_forecast_actual.csv"

print("읽은 예보 파일:", len(FORECASTS), "개")
fc = pd.concat(
    [pd.read_csv(f, dtype=str, encoding="utf-8-sig") for f in FORECASTS],
    ignore_index=True,
)
ac = pd.read_csv(ACTUAL, dtype=str)

key_fc = ["date", "time", "nx", "ny", "base_date", "base_time"]
print("예보 중복 행:", fc.duplicated(key_fc).sum())
fc = fc.drop_duplicates(key_fc)

fc = fc.rename(columns={"base_date": "baseDate", "base_time": "baseTime"})
for c in ["POP", "PCP", "PTY"]:
    fc[c] = pd.to_numeric(fc[c], errors="coerce")
for c in ["real_rain", "rain_missing", "temp", "humidity"]:
    ac[c] = pd.to_numeric(ac[c], errors="coerce")
for df in (fc, ac):
    df["nx"] = df["nx"].astype(int)
    df["ny"] = df["ny"].astype(int)

# "1mm 미만"은 PCP 빈칸 유지(임의 대체 안 함), 표시 컬럼만 추가
fc["pcp_under1mm"] = (fc["PCP_raw"] == "1mm 미만").astype(int)

keys = ["date", "time", "nx", "ny"]
ac = ac.drop_duplicates(subset=keys)
merged = fc.merge(
    ac[keys + ["real_rain", "rain_missing", "temp", "humidity"]],
    on=keys, how="left", validate="m:1",
)
# 실측 행이 아예 없으면 평가 제외
merged["rain_missing"] = merged["rain_missing"].fillna(1).astype(int)

issued = pd.to_datetime(merged["baseDate"] + merged["baseTime"], format="%Y%m%d%H%M")
target = pd.to_datetime(merged["date"] + merged["time"], format="%Y%m%d%H%M")
merged["lead_hours"] = (target - issued).dt.total_seconds() / 3600

ok = merged[merged["rain_missing"] == 0]
print(f"예보 행수: {len(fc)}, 평가 가능: {len(ok)}, 제외: {len(merged) - len(ok)}")
print(f"예보 기간: {fc['date'].min()} ~ {fc['date'].max()}")
print("--- 임시 확인용 (예보 PTY>0 vs 실측 real_rain>0) ---")
print(pd.crosstab(ok["PTY"] > 0, ok["real_rain"] > 0,
                  rownames=["예보 비"], colnames=["실제 비"]))

cols = ["baseDate", "baseTime", "date", "time", "lead_hours", "nx", "ny",
        "POP", "PCP", "PTY", "PCP_raw", "pcp_under1mm",
        "real_rain", "rain_missing", "temp", "humidity"]
merged[cols].to_csv(OUT, index=False, encoding="utf-8-sig")
print("저장 완료:", OUT)