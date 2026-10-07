import json
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

API_KEY = os.getenv("KMA_API_KEY", "").strip()
URL = (
    "https://apis.data.go.kr/1360000/"
    "VilageFcstInfoService_2.0/getVilageFcst"
)

# 현재 테스트 좌표. ERICA 대응 격자는 최종 확인 필요.
NX, NY = 58, 121
REGION = "안산_주변_테스트"
KST = timezone(timedelta(hours=9))


def parse_pcp(value):
    """강수량 원본을 보존하고 정확한 수치만 숫자로 변환."""
    if pd.isna(value):
        return float("nan"), "missing"

    text = str(value).strip()

    if text == "강수없음":
        return 0.0, "none"

    if re.fullmatch(r"\d+(?:\.\d+)?\s*(?:mm)?", text):
        return float(text.replace("mm", "").strip()), "numeric"

    if "미만" in text or "이상" in text or "~" in text:
        return float("nan"), "range"

    return float("nan"), "unknown"


def main():
    if not API_KEY:
        raise ValueError(".env의 KMA_API_KEY를 확인해주세요.")

    now = datetime.now(KST)

    # 매일 당일 14시 발표 예보 사용
    issued_at = now.replace(
        hour=14, minute=0, second=0, microsecond=0
    )

    # 저장 구간: 당일 15시 ~ 다음 날 15시 (양 끝 포함)
    window_start = now.replace(
        hour=15, minute=0, second=0, microsecond=0
    )
    window_end = window_start + timedelta(days=1)

    if now < window_start:
        raise ValueError(
            "이 코드는 오후 3시 이후에 실행해주세요."
        )

    params = {
        "serviceKey": API_KEY,
        "dataType": "JSON",
        "base_date": issued_at.strftime("%Y%m%d"),
        "base_time": "1400",
        "nx": NX,
        "ny": NY,
        "numOfRows": 1000,
    }

    print(f"예보 발표 시각: {issued_at.isoformat()}")
    print(f"저장 시작: {window_start.isoformat()}")
    print(f"저장 종료: {window_end.isoformat()}")

    items = []
    page = 1

    # 모든 페이지 수집
    while True:
        params["pageNo"] = page

        response = requests.get(
            URL, params=params, timeout=30
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"HTTP 오류: {response.status_code}"
            )

        try:
            payload = response.json()
        except ValueError:
            raise RuntimeError(
                "JSON 응답이 아닙니다. 인증키와 승인 상태를 확인해주세요."
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

    output_dir = ROOT / "data"
    output_dir.mkdir(exist_ok=True)

    stamp = now.strftime("%Y%m%d_%H%M%S_%f")

    # API에서 받은 전체 원본 보관
    raw_path = output_dir / f"forecast_raw_{stamp}.json"
    raw_path.write_text(
        json.dumps(items, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    raw = pd.DataFrame(items)
    keys = [
        "baseDate", "baseTime",
        "fcstDate", "fcstTime",
        "nx", "ny",
    ]

    selected = raw[
        raw["category"].isin(["POP", "PCP", "TMP", "REH"])
    ].copy()

    # 완전히 동일한 원본 행만 제거
    selected = selected.drop_duplicates()

    if selected.duplicated(keys + ["category"]).any():
        raise ValueError("같은 시간·항목에 서로 다른 값이 있습니다.")

    df = selected.pivot(
        index=keys,
        columns="category",
        values="fcstValue",
    ).reset_index()
    df.columns.name = None

    for column in ["POP", "PCP", "TMP", "REH"]:
        if column not in df:
            df[column] = pd.NA

    # 날짜와 시간 합치기
    for name, date_col, time_col in [
        ("issued_at", "baseDate", "baseTime"),
        ("forecast_at", "fcstDate", "fcstTime"),
    ]:
        text = (
            df[date_col].astype(str)
            + df[time_col].astype(str).str.zfill(4)
        )
        df[name] = pd.to_datetime(
            text, format="%Y%m%d%H%M"
        ).dt.tz_localize(KST)

    # 당일 15시 ~ 다음 날 15시만 선택
    df = df[
        df["forecast_at"].between(window_start, window_end)
    ].copy()

    # 25개 시간대가 모두 있는지 확인
    expected = pd.date_range(
        start=window_start,
        end=window_end,
        freq="h",
    )
    missing = expected.difference(df["forecast_at"])

    if len(missing):
        raise ValueError(
            "누락된 예보 시간대: "
            + ", ".join(t.strftime("%m-%d %H:%M") for t in missing)
        )

    for column in ["POP", "TMP", "REH"]:
        df[column] = pd.to_numeric(
            df[column], errors="coerce"
        )

    for column in ["POP", "REH"]:
        invalid = (
            df[column].notna()
            & ~df[column].between(0, 100)
        )
        if invalid.any():
            raise ValueError(f"{column} 값이 0~100 범위를 벗어났습니다.")

    df = df.rename(columns={"PCP": "pcp_raw"})
    parsed = df["pcp_raw"].apply(parse_pcp)
    df["pcp_mm"] = parsed.apply(lambda value: value[0])
    df["pcp_type"] = parsed.apply(lambda value: value[1])

    df["region"] = REGION
    df["collected_at"] = now.isoformat()
    df = df.sort_values("forecast_at")

    csv_path = output_dir / f"forecast_clean_{stamp}.csv"
    df.to_csv(
        csv_path, index=False, encoding="utf-8-sig"
    )

    print(f"수집 완료: {len(df)}행")
    print(f"원본 저장: {raw_path.name}")
    print(f"CSV 저장: {csv_path.name}")
    print(df[["forecast_at", "POP", "pcp_raw", "pcp_mm"]])


if __name__ == "__main__":
    try:
        main()
    except requests.RequestException:
        print("통신 오류: 인터넷 연결을 확인해주세요.")
        raise SystemExit(1)
    except Exception as error:
        # 요청 URL 등 인증키가 포함될 수 있는 정보는 출력하지 않음
        if isinstance(error, (ValueError, RuntimeError)):
            print(f"실행 실패: {error}")
        else:
            print(f"실행 실패: {type(error).__name__}")
        raise SystemExit(1)