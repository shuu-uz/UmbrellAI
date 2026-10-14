# 저장소 확인 결과 (2026-10-14)

확인은 코드와 예보 CSV 샘플 기준이며 담당자 확인을 대신하지 않습니다.

## 현재 컬럼 대응

| 분석용 이름 | 예보 수집 브랜치 현재 이름 | 상태 |
|---|---|---|
| fcstDate / fcstTime | date / time | normalize_forecast에서 대응 |
| nx / ny / POP | nx / ny / POP | 그대로 사용 |
| pcp_raw / pcp_mm / pcp_type | PCP_raw / PCP / PTY | 대응 가능, 세부 변환 의미 확인 필요 |
| baseDate / baseTime | base_date / base_time | 발표 시각 보존됨 |
| 수집 시각 | collected_at | 발표 시각과 구별 |
| 실제 강수량 | real_rain | 실측 코드상 확인, 미현님 최종 확인 필요 |
| 관측 누락 | rain_missing | 실측 원본 강수량 결측을 1로 표시 |

## 중요한 확인 사항

- feature/weather-collection의 2026-10-12 CSV에서 nx=58, ny=121이 확인됩니다. actual-rain 코드도 이 격자를 부여합니다. 격자 일치만으로 관측 대표성·집계 구간이 검증되지는 않습니다.
- actual-rain/src/preprocess_asos.py는 2023~2025 파일과 시간표를 처리합니다. 이 코드의 산출물로 2026년 예보 정확도를 계산할 수 없습니다. 2026년 실측 확보·처리는 실측 담당과 협의해야 합니다.
- 실측 코드가 강수량 결측을 모두 rain_missing=1로 표시한 뒤 0으로 채웁니다. 원자료에서 빈칸이 실제 관측 누락인지 무강수 표현인지 담당자가 확인해야 합니다. 의미가 확정되기 전 제외 비율을 주의해서 해석하세요.
- 소현님 브랜치는 main 기준으로 만들며 다른 담당 브랜치는 수정하지 않습니다. 병합·수집 코드는 이 변경에 포함하지 않습니다.
- 별도 팀 AGENTS.md는 main과 확인한 예보·실측 브랜치 파일 목록에서 발견되지 않았습니다.

## 실행

저장소 루트에서 `python -m unittest discover -s tests -v`.
evaluate는 현재 컬럼을 가진 병합 결과도 받으며 기본 실측 이름은 real_rain입니다.
normalize_forecast는 컬럼 대응만 수행하며 실제 데이터 병합·발표 시각 선택·평가 기간 제한을 수행하지 않습니다.
tests의 actual_mm은 합성 테스트 전용 이름입니다. 모든 테스트 수치는 실제 성능이 아닙니다.

## 확인한 소스

- https://github.com/shuu-uz/UmbrellAI/blob/feature/weather-collection/collect_weather.py
- https://github.com/shuu-uz/UmbrellAI/blob/feature/weather-collection/data/forecast_clean_20261012_164137_886701.csv
- https://github.com/shuu-uz/UmbrellAI/blob/actual-rain/src/preprocess_asos.py
