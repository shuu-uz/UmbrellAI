import json
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv


# 경로: 이 파이썬 파일이 있는 폴더 기준
ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

API_KEY = os.getenv("KMA_API_KEY", "").strip()

URL = (
    "https://apis.data.go.kr/1360000/"
    "VilageFcstInfoService_2.0/getVilageFcst"
)

# 현재 테스트 격자. ERICA 대응 좌표는 팀과 최종 확인 필요.
# 이미지의 59, 121은 예시이므로 임의로 변경하지 않음.
NX, NY = 58, 121

# 한국 시간
KST = timezone(timedelta(hours=9))


def convert_pcp(value):
    """강수량을 숫자로 변환하고 원본은 별도 열에 보존."""
    if pd.isna(value):
        return float("nan")

    text = str(value).strip()

    # 강수없음 → 숫자 0
    if text == "강수없음":
        return 0.0

    # 1.0mm → 숫자 1.0
    if re.fullmatch(r"\d+(?:\.\d+)?\s*(?:mm)?", text):
        return float(text.replace("mm", "").strip())

    # '1mm 미만' 등 정확한 수치가 아닌 표현은 결측으로 유지
    return float("nan")


def collect_forecast(base_date):
    """당일 14시 발표 예보의 전체 페이지를 수집."""
    params = {
        "serviceKey": API_KEY,
        "dataType": "JSON",
        "base_date": base_date,
        "base_time": "1400",
        "nx": NX,
        "ny": NY,
        "numOfRows": 1000,
    }

    items = []
    page = 1

    while True:
        params["pageNo"] = page

        response = requests.get(
            URL,
            params=params,
            timeout=30,
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"HTTP 오류: {response.status_code}"
            )

        try:
            payload = response.json()
        except ValueError:
            raise RuntimeError(
                "JSON 응답이 아닙니다. "
                "인증키와 API 승인 상태를 확인해주세요."
            ) from None

        header = payload["response"]["header"]

        if header["resultCode"] != "00":
            raise RuntimeError(
                f"API 오류 {header['resultCode']}: "
                f"{header['resultMsg']}"
            )

        body = payload["response"]["body"]
        batch = (body.get("items") or {}).get("item", [])

        if not batch:
            raise RuntimeError(
                "예보 데이터가 없거나 페이지가 누락됐습니다."
            )

        items.extend(batch)

        if len(items) >= int(body["totalCount"]):
            break

        page += 1

    return items


def preprocess(items, start, end, collected_at):
    """팀 공통 열 이름에 맞춰 전처리."""
    raw = pd.DataFrame(items)

    keys = [
        "baseDate",
        "baseTime",
        "fcstDate",
        "fcstTime",
        "nx",
        "ny",
    ]

    # 강수확률·강수량·강수형태만 추출
    selected = raw[
        raw["category"].isin(["POP", "PCP", "PTY"])
    ].copy()

    selected = selected.drop_duplicates()

    if selected.duplicated(keys + ["category"]).any():
        raise ValueError(
            "같은 시간·항목에 서로 다른 값이 있습니다."
        )

    # 항목별 데이터를 시간별 한 행으로 변환
    df = selected.pivot(
        index=keys,
        columns="category",
        values="fcstValue",
    ).reset_index()

    df.columns.name = None

    for column in ["POP", "PCP", "PTY"]:
        if column not in df.columns:
            raise ValueError(f"{column} 항목이 없습니다.")

    # 공통 병합 키: 예보 대상 날짜·시간 사용
    df["date"] = df["fcstDate"].astype(str)
    df["time"] = df["fcstTime"].astype(str).str.zfill(4)
    df["nx"] = df["nx"].astype(int)
    df["ny"] = df["ny"].astype(int)

    df["forecast_at"] = pd.to_datetime(
        df["date"] + df["time"],
        format="%Y%m%d%H%M",
    ).dt.tz_localize(KST)

    # 당일 15시 ~ 다음 날 15시, 양 끝 포함
    df = df[
        df["forecast_at"].between(start, end)
    ].copy()

    # 25개 시간대가 모두 있는지 확인
    expected = pd.date_range(
        start=start,
        end=end,
        freq="h",
    )

    missing = expected.difference(df["forecast_at"])

    if len(missing):
        missing_text = ", ".join(
            timestamp.strftime("%m-%d %H:%M")
            for timestamp in missing
        )
        raise ValueError(
            f"예보 시간대가 누락됐습니다: {missing_text}"
        )

    # 숫자형 변환
    df["POP"] = pd.to_numeric(
        df["POP"], errors="coerce"
    )
    df["PTY"] = pd.to_numeric(
        df["PTY"], errors="coerce"
    )

    if df[["POP", "PTY"]].isna().any().any():
        raise ValueError(
            "POP 또는 PTY에 누락·비숫자 값이 있습니다."
        )

    if not df["POP"].between(0, 100).all():
        raise ValueError(
            "강수확률이 0~100 범위를 벗어났습니다."
        )

    if not df["PTY"].isin([0, 1, 2, 3, 4]).all():
        raise ValueError(
            "예상하지 못한 강수형태 코드가 있습니다."
        )

    df["PTY"] = df["PTY"].astype(int)

    # 강수량 원본 보존 및 숫자 변환
    df["PCP_raw"] = df["PCP"]
    df["PCP"] = df["PCP_raw"].apply(convert_pcp)

    # 발표 시각과 수집 시각 보존
    df["base_date"] = df["baseDate"].astype(str)
    df["base_time"] = (
        df["baseTime"].astype(str).str.zfill(4)
    )
    df["collected_at"] = collected_at.isoformat()

    # 병합 키 중복 확인
    merge_keys = ["date", "time", "nx", "ny"]

    if df.duplicated(merge_keys).any():
        raise ValueError(
            "date, time, nx, ny 기준으로 중복이 있습니다."
        )

    columns = [
        "date",
        "time",
        "nx",
        "ny",
        "POP",
        "PCP",
        "PTY",
        "PCP_raw",
        "base_date",
        "base_time",
        "collected_at",
    ]

    return df.sort_values(["date", "time"])[columns]


def main():
    if not API_KEY:
        raise ValueError(
            ".env의 KMA_API_KEY를 확인해주세요."
        )

    now = datetime.now(KST)

    start = now.replace(
        hour=15,
        minute=0,
        second=0,
        microsecond=0,
    )
    end = start + timedelta(days=1)

    if now < start:
        raise ValueError(
            "이 코드는 오후 3시 이후에 실행해주세요."
        )

    print("당일 14시 발표 예보를 수집합니다.")
    print(f"대상 격자: nx={NX}, ny={NY}")
    print(f"저장 시작: {start.isoformat()}")
    print(f"저장 종료: {end.isoformat()}")

    items = collect_forecast(now.strftime("%Y%m%d"))

    output_dir = ROOT / "data"
    output_dir.mkdir(exist_ok=True)

    stamp = now.strftime("%Y%m%d_%H%M%S_%f")

    # API 전체 원본 JSON 저장
    raw_path = (
        output_dir / f"forecast_raw_{stamp}.json"
    )
    raw_path.write_text(
        json.dumps(
            items,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # 전처리 후 지정 구간 CSV 저장
    df = preprocess(items, start, end, now)

    csv_path = (
        output_dir / f"forecast_clean_{stamp}.csv"
    )
    df.to_csv(
        csv_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"수집 완료: {len(df)}행")
    print(f"원본 저장: {raw_path.name}")
    print(f"CSV 저장: {csv_path.name}")
    print(df.head().to_string(index=False))

    missing_pcp = df["PCP"].isna().sum()
    if missing_pcp:
        print(
            f"강수량 범위 표현 또는 누락 {missing_pcp}건: "
            "PCP_raw를 확인해주세요."
        )


if __name__ == "__main__":
    try:
        main()

    except requests.RequestException:
        # 인증키가 포함된 요청 URL은 출력하지 않음
        print(
            "통신 오류: 인터넷 연결을 확인해주세요."
        )
        raise SystemExit(1)

    except Exception as error:
        if isinstance(error, (ValueError, RuntimeError)):
            print(f"실행 실패: {error}")
        else:
            print(
                f"실행 실패: {type(error).__name__}"
            )
        raise SystemExit(1)