"""Exploratory stated-intention model; never a probability of physical need."""
import math

FEATURES = ['outdoor_minutes', 'carry_discomfort', 'wet_discomfort']


def intention_score(model, *, pop, outdoor_minutes, carry_discomfort, wet_discomfort):
    """Only score the two measured scenarios (POP=50%, 10/60 minutes).

    The rain amount and travel-window conditions were not measured in Q1/Q2.
    Refuse extrapolation rather than imply validation across weather conditions.
    """
    values = [pop, outdoor_minutes, carry_discomfort, wet_discomfort]
    if any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
        raise ValueError('Inputs must be finite numbers')
    if pop != 50 or outdoor_minutes not in (10, 60):
        return {'status': 'outside_survey_domain', 'score': None,
                'reason': 'Measured scenarios only: POP=50 and outdoor_minutes=10 or 60'}
    if not 1 <= carry_discomfort <= 5 or not 1 <= wet_discomfort <= 5:
        raise ValueError('Discomfort must be on the measured 1-5 scale')
    if model.get('feature_names') != FEATURES or model.get('target') != 'stated_carry_intention':
        raise ValueError('Unsupported model schema/target')
    x = [outdoor_minutes, carry_discomfort, wet_discomfort]
    z = model['intercept'] + sum(
        c * (v - mean) / scale for c, v, mean, scale in
        zip(model['coefficients'], x, model['mean'], model['scale']))
    probability = 1 / (1 + math.exp(-z)) if z >= 0 else math.exp(z) / (1 + math.exp(z))
    return {'status': 'exploratory', 'score': round(100 * probability, 2),
            'meaning': 'Model-estimated stated carry intention, not actual umbrella need',
            'domain': 'POP=50%; 10 or 60 outdoor minutes; survey 1-5 discomfort scales'}
