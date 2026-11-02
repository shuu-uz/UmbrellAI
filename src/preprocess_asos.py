import pandas as pd
from pathlib import Path

DATA_DIR = Path("./data")
FILES = ["asos_2023.csv", "asos_2024.csv", "asos_2025.csv", "asos_2026.csv"]
OUT_FILE = DATA_DIR / "real_rain_all.csv"
NX, NY = 58, 121  # 한양대 ERICA 격자 (안산 545 지점에 동일 부여)
COLS = ["stn_id", "stn_name", "datetime", "temp", "real_rain", "humidity"]


def read_any(path):
    for enc in ("utf-8-sig", "cp949"):
        try:
            df = pd.read_csv(path, encoding=enc)
            df = df.iloc[:, :6]      # 앞 6개 열만 사용
            df.columns = COLS        # 파일마다 다른 컬럼명을 위치 기준으로 통일
            print(f"읽음: {path.name} ({enc}, {len(df)}행)")
            return df
        except UnicodeDecodeError:
            continue
    raise ValueError(f"인코딩을 읽을 수 없음: {path}")


raw = pd.concat([read_any(DATA_DIR / f) for f in FILES], ignore_index=True)
raw["datetime"] = pd.to_datetime(raw["datetime"])
raw = raw.drop_duplicates(subset="datetime")

# 전체 1시간 시간표에 맞춰 누락된 시간도 행으로 만들기
full_idx = pd.date_range("2023-01-01 00:00", "2026-10-31 23:00", freq="h")
raw = raw.set_index("datetime").reindex(full_idx)
raw.index.name = "datetime"
raw = raw.reset_index()

# 원래 비어 있던 강수량 표시 (fillna 하기 전에!)
raw["rain_missing"] = raw["real_rain"].isna().astype(int)
print("전체 행수:", len(raw), "/ 강수량 없는 시간:", raw["rain_missing"].sum())

# 팀 규칙대로 0.0 채움 (구분은 rain_missing으로 가능)
raw["real_rain"] = raw["real_rain"].fillna(0.0)

raw["date"] = raw["datetime"].dt.strftime("%Y%m%d")
raw["time"] = raw["datetime"].dt.strftime("%H00")
raw["nx"] = NX
raw["ny"] = NY

out = raw[["date", "time", "nx", "ny", "real_rain", "rain_missing", "temp", "humidity"]]
out.to_csv(OUT_FILE, index=False, encoding="utf-8-sig")
print("저장 완료:", OUT_FILE)