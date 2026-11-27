from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile
import hashlib
import pandas as pd
from src.run_analysis import weather_analysis, survey_analysis, charts, dump
d = Path.home() / "Downloads"
out = Path(__file__).resolve().parent / "results" / "final_20261123"
out.mkdir(exist_ok=True)
weather_zip = d / "최종 실측 병합 데이터.zip"
forecast_zip = next(d.glob("*최종 예보 데이터.zip"))
survey_file = next(d.glob("*설문지 응답 시트1.csv"))
with TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    with ZipFile(weather_zip) as z:
        merged = tmp / "merged.csv"
        merged.write_bytes(z.read(next(n for n in z.namelist() if n.endswith("/merged_forecast_actual.csv"))))
    with ZipFile(forecast_zip) as z:
        files = [n for n in z.namelist() if n.startswith("data/forecast_clean_") and n.endswith(".csv")]
        forecasts = pd.concat([pd.read_csv(z.open(n)) for n in files]).rename(columns={"base_date":"baseDate","base_time":"baseTime"})
    fields = ["date","time","nx","ny","baseDate","baseTime","POP","PTY","PCP"]
    def ordered(frame):
        return frame[fields].sort_values(fields).reset_index(drop=True)
    pd.testing.assert_frame_equal(ordered(forecasts), ordered(pd.read_csv(merged)), check_dtype=False)
    s = pd.read_csv(survey_file)
    clean = s.iloc[:,[1,2,3,4,5,6,8]].copy()
    clean.columns = ["이동10분","이동60분","이동시간비없음","이동시간비있음","우산짐불편도","비맞음불편도","이슬비지참"]
    clean.to_csv(tmp / "survey.csv",index=False)
    choices = ["챙기지 않는다", "고민된다", "챙긴다"]
    for before, after in [("이동10분","이동60분"),("이동시간비없음","이동시간비있음")]:
        transitions = pd.crosstab(clean[before],clean[after]).reindex(index=choices, columns=choices, fill_value=0)
        transitions.to_csv(out / f"{before}_{after}_전환표.csv",encoding="utf-8-sig")
    w = weather_analysis(merged, out)
    survey = survey_analysis(tmp / "survey.csv", out)
    charts(w,survey,out)
    v = pd.read_csv(out / "selected_forecasts.csv").query("rain_missing == 0 and real_rain >= 0 and 0 <= POP <= 100")
    print("wet days",v.loc[v.real_rain.gt(0),"date"].nunique(), "all days",v.date.nunique(), v.target_at.min(),v.target_at.max())
    dump(out / "inputs.json",{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [weather_zip,forecast_zip,survey_file]})
