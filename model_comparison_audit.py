"""Match uncertainty analysis to the exact published model comparisons.

No changes to the existing point estimates. Only aggregate outputs are saved.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.run_analysis import grouped_model, FEATURES
from rain_score import survey_frame, evaluate_models

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results' / 'final_20261123'


def paired_brier_interval(y, baseline, personalized, groups, *, repetitions=5000, seed=42):
    y, baseline, personalized = [np.asarray(v, dtype=float) for v in (y, baseline, personalized)]
    groups = np.asarray(groups)
    if not (len(y) == len(baseline) == len(personalized) == len(groups)) or len(y) == 0:
        raise ValueError('Paired predictions and group IDs must have matching nonzero lengths')
    if not all(np.isfinite(v).all() for v in (y, baseline, personalized)):
        raise ValueError('Predictions/outcomes must be finite')
    if not np.isin(y, [0, 1]).all() or not all(((v >= 0) & (v <= 1)).all() for v in (baseline, personalized)):
        raise ValueError('Binary outcomes and probabilities required')
    frame = pd.DataFrame({'group': groups, 'delta': (personalized-y)**2 - (baseline-y)**2})
    if frame.group.isna().any() or frame.groupby('group').size().nunique() != 1:
        raise ValueError('Need complete groups with equal scenario counts for this audit')
    paired = frame.groupby('group', sort=True).delta.mean().to_numpy()
    index = np.random.default_rng(seed).integers(0, len(paired), size=(repetitions, len(paired)))
    low, high = np.quantile(paired[index].mean(axis=1), [.025, .975])
    return dict(baseline_brier=float(np.mean((baseline-y)**2)),
                personalized_brier=float(np.mean((personalized-y)**2)),
                delta=float(paired.mean()), low=float(low), high=float(high),
                respondents=len(paired), scenario_rows=len(y), repetitions=repetitions, seed=seed,
                delta_definition='personalized minus baseline; negative favors personalized',
                interval='Paired respondent percentile bootstrap, conditional on fixed OOF predictions',
                contains_zero=bool(low <= 0 <= high))


def audit_models(raw):
    clean = raw.iloc[:, [1,2,5,6]].copy()
    clean.columns = ['short','long','carry','wet']
    long = []
    for i, r in clean.iterrows():
        for col, minutes in [('short',10),('long',60)]:
            if str(r[col]).strip() not in ['챙긴다','챙기지 않는다','고민된다']:
                raise ValueError('Invalid survey answer')
            long.append(dict(respondent=i, outdoor_minutes=minutes,
                             carry_discomfort=r['carry'], wet_discomfort=r['wet'],
                             y=int(str(r[col]).strip() == '챙긴다')))
    long = pd.DataFrame(long)
    base50, _, pred0 = grouped_model(long, ['outdoor_minutes'], return_predictions=True)
    personal50, _, pred1 = grouped_model(long, FEATURES, return_predictions=True)
    expected50 = json.loads((OUT/'survey_metrics.json').read_text(encoding='utf-8'))['models']
    for actual, expected in [(base50, expected50['time_only']), (personal50, expected50['personalized'])]:
        for key in ['brier','accuracy','roc_auc']:
            if not np.isclose(actual['oof'][key], expected['oof'][key], atol=1e-12, rtol=0):
                raise ValueError(f'POP50 table reproduction failed: {key}')
    result50 = paired_brier_interval(long.y, pred0, pred1, long.respondent)
    result50['cv'] = 'StratifiedGroupKFold(5, shuffle=True, random_state=42); original person-first order'
    frame60 = survey_frame(raw)
    metrics60, predictions60 = evaluate_models(frame60)
    expected60 = json.loads((OUT/'rain_score.json').read_text(encoding='utf-8'))['validation']
    for model in ['weather_only','combined']:
        for key in ['brier','accuracy','auc']:
            if not np.isclose(metrics60[model][key], expected60[model][key], atol=1e-12, rtol=0):
                raise ValueError(f'POP60 table reproduction failed: {key}')
    result60 = paired_brier_interval(frame60.target, predictions60['weather_only'],
                                    predictions60['combined'], frame60.person)
    result60['cv'] = 'GroupKFold(5); original scenario-first order'
    return {'POP50': result50, 'daily_POP60': result60}


def save_audit(raw, *, source_sha256=None):
    results = audit_models(raw)
    payload = {'conditions': results, 'survey_sha256': source_sha256,
               'point_estimates_reproduced': True,
               'versions': {p: importlib.metadata.version(p) for p in ['numpy','pandas','scikit-learn']},
               'limitations': ['Conditional intervals do not include refitting/model selection uncertainty.',
                               '80 source submissions are treated as respondent groups; repeat submissions unknown.',
                               'Survey intentions only; not independent external or behavioral validation.']}
    (OUT/'paired_brier_uncertainty.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2)+'\n',encoding='utf-8')
    pd.DataFrame([{'condition': key, **value} for key,value in results.items()]).to_csv(
        OUT/'paired_brier_uncertainty.csv',index=False,encoding='utf-8-sig')
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--survey', type=Path)
    args = parser.parse_args()
    source = args.survey or next((Path.home()/'Downloads').glob('*설문지 응답 시트1.csv'))
    sha = hashlib.sha256(source.read_bytes()).hexdigest()
    expected = json.loads((OUT/'inputs.json').read_text(encoding='utf-8'))
    expected_sha, = [v for k,v in expected.items() if k.endswith('설문지 응답 시트1.csv')]
    if sha != expected_sha:
        raise ValueError('Survey file differs from the final report input hash')
    payload = save_audit(pd.read_csv(source), source_sha256=sha)
    for key, value in payload['conditions'].items():
        print(key, value['baseline_brier'], value['personalized_brier'],
              'delta', value['delta'], '95% interval', value['low'], value['high'])


if __name__ == '__main__':
    main()
