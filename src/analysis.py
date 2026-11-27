"""Merged hourly weather evaluation. Standard library only; no data collection."""
from math import isfinite


def normalize_forecast(row):
    """Map the collector's current columns without modifying its code."""
    result = dict(row)
    aliases = {'fcstDate': 'date', 'fcstTime': 'time', 'pcp_raw': 'PCP_raw',
               'pcp_mm': 'PCP', 'pcp_type': 'PTY',
               'baseDate': 'base_date', 'baseTime': 'base_time'}
    for target, source in aliases.items():
        if target not in result and source in result:
            result[target] = result[source]
    for column in ('fcstTime', 'baseTime'):
        if column in result and result[column] is not None:
            result[column] = str(result[column]).strip().zfill(4)
    return result


def number(value):
    try:
        result = float(value)
        return result if isfinite(result) else None
    except (TypeError, ValueError):
        return None


def evaluate(rows, *, rain_column='real_rain', pop_threshold=50, rain_threshold_mm=0,
             synthetic=False):
    """Input must contain one selected forecast per target time/grid.

    Rain is observed when amount > rain_threshold_mm. Ratios are None
    when their denominator is zero. Missing observations are excluded.
    """
    if not 0 <= pop_threshold <= 100 or rain_threshold_mm < 0:
        raise ValueError('Invalid thresholds')
    required = {'fcstDate', 'fcstTime', 'nx', 'ny', 'POP',
                rain_column, 'rain_missing'}
    seen = set()
    counts = dict(TN=0, FP=0, FN=0, TP=0)
    excluded = dict(observation_missing=0, invalid_value=0)
    for row in rows:
        row = normalize_forecast(row)
        missing = required - row.keys()
        if missing:
            raise ValueError(f'Missing columns: {sorted(missing)}')
        key = tuple(str(row[k]).strip() for k in ('fcstDate', 'fcstTime', 'nx', 'ny'))
        if any(not x or x == 'None' for x in key):
            raise ValueError('Missing target key')
        if key in seen:
            raise ValueError(f'Duplicate target: {key}; select forecast issuance first')
        seen.add(key)
        flag = number(row['rain_missing'])
        if flag not in (0, 1):
            excluded['invalid_value'] += 1
            continue
        if flag == 1:
            excluded['observation_missing'] += 1
            continue
        pop, rain = number(row['POP']), number(row[rain_column])
        if pop is None or rain is None or not 0 <= pop <= 100 or rain < 0:
            excluded['invalid_value'] += 1
            continue
        predicted, actual = pop >= pop_threshold, rain > rain_threshold_mm
        counts[('TP' if actual else 'FP') if predicted else ('FN' if actual else 'TN')] += 1
    tn, fp, fn, tp = (counts[k] for k in ('TN', 'FP', 'FN', 'TP'))
    ratio = lambda n, d: n / d if d else None
    return dict(data_kind='SYNTHETIC_ONLY' if synthetic else 'REAL_DATA',
                evaluated_rows=tn+fp+fn+tp, excluded=excluded,
                confusion_matrix=[[tn, fp], [fn, tp]], counts=counts,
                false_alarm_ratio=ratio(fp, tp+fp),
                false_positive_rate=ratio(fp, tn+fp),
                miss_rate=ratio(fn, tp+fn),
                accuracy=ratio(tp+tn, tn+fp+fn+tp))


def recommend_umbrella(periods, outdoor_minutes, *, pop_threshold=50,
                       amount_threshold_mm=1, exposure_threshold_minutes=15):
    """periods: selected travel-time hourly forecasts with POP and pcp_mm.

    Draft heuristic only. Missing/invalid input returns insufficient_data.
    Precipitation type must be checked upstream (snow is not rain).
    """
    if not 0 <= pop_threshold <= 100 or amount_threshold_mm <= 0 or exposure_threshold_minutes <= 0:
        raise ValueError('Invalid recommendation thresholds')
    minutes = number(outdoor_minutes)
    if minutes is None or minutes < 0:
        raise ValueError('outdoor_minutes must be a nonnegative finite number')
    if minutes == 0:
        return dict(level='low_priority', reason='No outdoor travel', draft=True)
    periods = [normalize_forecast(p) for p in periods]
    values = [(number(p.get('POP')), number(p.get('pcp_mm'))) for p in periods]
    if not values or any(p is None or a is None or not 0 <= p <= 100 or a < 0 for p, a in values):
        return dict(level='insufficient_data', reason='Missing/invalid travel forecast', draft=True)
    max_pop = max(p for p, _ in values)
    max_amount = max(a for _, a in values)
    if max_amount >= amount_threshold_mm or (max_pop >= pop_threshold and minutes >= exposure_threshold_minutes):
        level = 'recommended'
    elif max_pop >= pop_threshold or max_amount > 0:
        level = 'consider'
    else:
        level = 'low_priority'
    return dict(level=level, max_pop=max_pop, max_hourly_pcp_mm=max_amount,
                outdoor_minutes=minutes, draft=True)
