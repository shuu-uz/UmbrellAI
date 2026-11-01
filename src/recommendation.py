"""Connect evidence to scoring without extrapolating beyond measured conditions."""
import math
from .intention_model import intention_score


def recommend_from_travel_forecasts(periods, model, *, outdoor_minutes,
                                    carry_discomfort, wet_discomfort,
                                    travel_window_confirmed=False):
    """Caller selects the user's travel hours/location and an available issuance.

    H1: retain original POP; calibration is not learned from one wet day.
    H2: use travel-window POP and the measured outdoor duration.
    H3: pass measured discomfort preferences to the fitted intention model.
    No arbitrary 15-minute cutoff, calibrated POP, or hard recommendation cutoff.
    """
    if not isinstance(outdoor_minutes, (int, float)) or not math.isfinite(outdoor_minutes) or outdoor_minutes < 0:
        raise ValueError('outdoor_minutes must be finite and nonnegative')
    base = {'score_kind': 'stated_carry_intention', 'calibration': 'raw_POP_retained',
            'actual_need_validated': False, 'hard_recommendation': None}
    if outdoor_minutes == 0:
        return {**base, 'status': 'no_outdoor_exposure', 'score': None}
    if not travel_window_confirmed:
        return {**base, 'status': 'travel_window_required', 'score': None,
                'reason': 'Select forecasts for the actual travel time and location first'}
    periods = list(periods)
    if not periods:
        return {**base, 'status': 'insufficient_data', 'score': None}
    pops = []
    for row in periods:
        try:
            pop = float(row['POP'])
        except (KeyError, TypeError, ValueError):
            return {**base, 'status': 'insufficient_data', 'score': None}
        if not math.isfinite(pop) or not 0 <= pop <= 100:
            return {**base, 'status': 'insufficient_data', 'score': None}
        pops.append(pop)
    # Q1/Q2 do not specify precipitation type/amount; refuse known snow/mix.
    if any(str(row.get('PTY', '0')).strip() not in ('0', '1', '0.0', '1.0') for row in periods):
        return {**base, 'status': 'unsupported_precipitation_type', 'score': None}
    if any(pop != 50 for pop in pops):
        return {**base, 'status': 'outside_survey_domain', 'score': None,
                'travel_pops': pops, 'reason': 'No validated model for varying POP or POP other than 50%'}
    answer = intention_score(model, pop=50, outdoor_minutes=outdoor_minutes,
                             carry_discomfort=carry_discomfort, wet_discomfort=wet_discomfort)
    return {**base, **answer, 'travel_pops': pops,
            'algorithm_links': {'H1': 'Keep supplied POP; no probability recalibration',
                                'H2': 'Confirmed travel window and measured 10/60 minute condition',
                                'H3': 'Fitted carry/wet discomfort inputs'}}
