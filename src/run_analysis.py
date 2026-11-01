"""Reproduce weather evaluation and grouped survey intention analysis.

python -m src.run_analysis --weather data/merged_forecast_actual.csv \
    --survey work_inputs/survey_analysis/cleaned_responses.csv --out results/sohyun
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, brier_score_loss,
                             log_loss, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .analysis import evaluate
from .intention_model import FEATURES, intention_score

KEY = ['date', 'time', 'nx', 'ny']
SCENARIOS = ['이동10분', '이동60분', '이동시간비없음', '이동시간비있음', '이슬비지참']
VALID_CHOICES = {'챙긴다', '챙기지 않는다', '고민된다'}


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def ratio(n, d):
    return n / d if d else None


def select_forecasts(df):
    """Preserve original rows; select latest issuance <= target minus one hour.

    This is retrospective and conditional on the supplied archive: availability
    at decision time is unknown (collected_at absent). Never infer live availability.
    """
    df = df.copy()
    required = set(KEY + ['baseDate', 'baseTime', 'POP', 'real_rain', 'rain_missing', 'lead_hours'])
    if required - set(df):
        raise ValueError(f'Missing weather columns: {sorted(required - set(df))}')
    for c in KEY + ['baseDate', 'baseTime']:
        if df[c].isna().any():
            raise ValueError(f'Missing weather key: {c}')
        df[c] = df[c].astype(str).str.strip()
    for c in ['time', 'baseTime']:
        df[c] = df[c].str.zfill(4)
    for c in ['nx', 'ny']:
        grid = pd.to_numeric(df[c], errors='raise')
        if not np.isfinite(grid).all() or (grid % 1 != 0).any():
            raise ValueError('Grid keys must be finite integers')
        df[c] = grid.astype(int)
    df['target_at'] = pd.to_datetime(df['date'] + df['time'], format='%Y%m%d%H%M', errors='raise')
    df['issued_at'] = pd.to_datetime(df['baseDate'] + df['baseTime'], format='%Y%m%d%H%M', errors='raise')
    lead = (df.target_at - df.issued_at).dt.total_seconds() / 3600
    supplied_lead = pd.to_numeric(df.lead_hours, errors='raise')
    if not np.isfinite(supplied_lead).all() or not np.allclose(lead, supplied_lead):
        raise ValueError('lead_hours conflicts with target/issuance timestamps')
    if df.duplicated(KEY + ['baseDate', 'baseTime']).any():
        raise ValueError('Duplicate issuance/target key; resolve upstream instead of silently dropping')
    # Multiple forecasts for one observation must agree on the observation.
    for c in ['real_rain', 'rain_missing']:
        if (df.groupby(KEY)[c].nunique(dropna=False) > 1).any():
            raise ValueError(f'Conflicting observation for repeated target: {c}')
    df['decision_at'] = df.target_at - pd.Timedelta(hours=1)
    eligible = df[df.issued_at <= df.decision_at]
    selected = eligible.sort_values('issued_at').drop_duplicates(KEY, keep='last').sort_values(KEY)
    audit = {'input_rows': len(df), 'unique_targets': len(df.drop_duplicates(KEY)),
             'ineligible_issuance_rows': len(df) - len(eligible),
             'superseded_rows': len(eligible) - len(selected), 'selected_rows': len(selected),
             'timezone': 'Asia/Seoul (source assumption; timestamps have no offsets)',
             'selection': 'Latest supplied issuance <= target_at - 1 hour per date/time/nx/ny',
             'availability_limit': 'collected_at absent; operational availability not verified',
             'period': [str(df.date.min()), str(df.date.max())],
             'grids': df[['nx', 'ny']].drop_duplicates().values.tolist()}
    return selected, audit


def weather_analysis(path, out):
    raw = pd.read_csv(path, dtype=str, encoding='utf-8-sig')
    selected, audit = select_forecasts(raw)
    result = evaluate(selected.to_dict('records'), pop_threshold=50)
    result.update(audit)
    valid = selected.copy()
    for c in ['rain_missing', 'real_rain', 'POP', 'PTY', 'lead_hours']:
        if c in valid:
            valid[c] = pd.to_numeric(valid[c], errors='coerce')
    valid = valid[(valid.rain_missing == 0) & valid.real_rain.ge(0)
                  & valid.POP.between(0, 100) & np.isfinite(valid.real_rain)]
    result['observed_wet_hours'] = int(valid.real_rain.gt(0).sum())
    result['observed_dry_hours'] = int(valid.real_rain.eq(0).sum())
    result['evaluated_period'] = [str(valid.date.min()), str(valid.date.max())]
    result['source_missing_rows'] = int(pd.to_numeric(raw.rain_missing).eq(1).sum())
    result['under1mm_imputed_rows'] = int(pd.to_numeric(raw.get('pcp_under1mm', pd.Series(dtype=float))).eq(1).sum())
    result['notes'] = ['PCP 0.5mm substitution is upstream assumption; not used for POP evaluation.',
                       'temp/humidity are observations, excluded from prediction features.',
                       'Metrics are descriptive for this small supplied sample, not future performance.',
                       'Observation source/aggregation interval verified only from team documentation, not raw AWS.']
    selected.to_csv(out / 'selected_forecasts.csv', index=False, encoding='utf-8-sig')
    sensitivity = []
    for threshold in [20, 30, 50, 60, 70]:
        r = evaluate(selected.to_dict('records'), pop_threshold=threshold)
        sensitivity.append({'pop_threshold': threshold, **r['counts'],
                            **{k: r[k] for k in ['false_alarm_ratio', 'false_positive_rate', 'miss_rate', 'accuracy']}})
    pd.DataFrame(sensitivity).to_csv(out / 'weather_thresholds.csv', index=False)
    bins = pd.cut(valid.POP, [-0.01, 19.999, 39.999, 59.999, 79.999, 100],
                  labels=['0-<20', '20-<40', '40-<60', '60-<80', '80-100'])
    tables = []
    for label in bins.cat.categories:
        group = valid[bins == label]
        n, wet = len(group), int(group.real_rain.gt(0).sum())
        tables.append({'pop_bin': str(label), 'n': n, 'wet': wet, 'observed_wet_rate': ratio(wet, n)})
    pd.DataFrame(tables).to_csv(out / 'weather_pop_bins.csv', index=False)
    result['pop_bins'] = tables
    result['lead_time_groups'] = []
    for lo, hi in [(1, 6), (7, 12), (13, 24), (25, 25)]:
        part = valid[valid.lead_hours.between(lo, hi)]
        result['lead_time_groups'].append({'lead_hours': [lo, hi], **evaluate(part.to_dict('records'))})
    dump(out / 'weather_metrics.json', result)
    return result


def survey_validation(survey):
    required = SCENARIOS + ['우산짐불편도', '비맞음불편도']
    if set(required) - set(survey):
        raise ValueError('Survey required columns missing')
    if len(survey) < 10:
        raise ValueError('Too few respondents for 5 grouped folds')
    for c in SCENARIOS:
        survey[c] = survey[c].str.strip()
        if not survey[c].isin(VALID_CHOICES).all():
            raise ValueError(f'Missing/unknown survey choice: {c}')
    for c in ['우산짐불편도', '비맞음불편도']:
        survey[c] = pd.to_numeric(survey[c], errors='raise')
        if not survey[c].between(1, 5).all() or not np.isfinite(survey[c]).all():
            raise ValueError(f'Invalid discomfort: {c}')


def paired_test(survey, before, after):
    a, b = survey[before].eq('챙긴다'), survey[after].eq('챙긴다')
    gain, loss = int((~a & b).sum()), int((a & ~b).sum())
    p = float(stats.binomtest(gain, gain + loss, .5).pvalue) if gain + loss else 1.
    return {'before': before, 'after': after, 'n_respondents': len(survey),
            'before_carry': int(a.sum()), 'after_carry': int(b.sum()),
            'gain': gain, 'loss': loss, 'exact_mcnemar_p': p}


def grouped_model(long, columns):
    x = long[columns].to_numpy(dtype=float)
    y = long.y.to_numpy(dtype=int)
    groups = long.respondent.to_numpy()
    if len(np.unique(y)) < 2 or any(long.groupby('respondent').size() > 2):
        raise ValueError('Need two outcome classes and at most two paired rows per respondent')
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    predictions = np.empty(len(long))
    baseline = np.empty(len(long))
    folds = []
    for fold, (train, test) in enumerate(splitter.split(x, y, groups), 1):
        if set(groups[train]) & set(groups[test]):
            raise AssertionError('Respondent leakage')
        if len(np.unique(y[train])) < 2:
            raise ValueError('Training fold has only one class')
        model = make_pipeline(StandardScaler(), LogisticRegression(C=1., max_iter=2000, random_state=42))
        model.fit(x[train], y[train])
        predictions[test] = model.predict_proba(x[test])[:, 1]
        dummy = DummyClassifier(strategy='prior').fit(x[train], y[train])
        baseline[test] = dummy.predict_proba(x[test])[:, 1]
        folds.append({'fold': fold, 'train_respondents': len(set(groups[train])),
                      'test_respondents': len(set(groups[test])), 'test_rows': len(test),
                      'group_overlap': 0})
    def metrics(prob):
        return {'accuracy': float(accuracy_score(y, prob >= .5)),
                'balanced_accuracy': float(balanced_accuracy_score(y, prob >= .5)),
                'roc_auc': float(roc_auc_score(y, prob)),
                'brier': float(brier_score_loss(y, prob)),
                'log_loss': float(log_loss(y, prob, labels=[0, 1]))}
    fitted = make_pipeline(StandardScaler(), LogisticRegression(C=1., max_iter=2000, random_state=42)).fit(x, y)
    scale, lr = fitted.steps[0][1], fitted.steps[1][1]
    artifact = {'target': 'stated_carry_intention', 'feature_names': columns,
                'mean': scale.mean_.tolist(), 'scale': scale.scale_.tolist(),
                'coefficients': lr.coef_[0].tolist(), 'intercept': float(lr.intercept_[0]),
                'classes': lr.classes_.tolist(), 'C': 1.,
                'score_domain': {'POP': [50], 'outdoor_minutes': [10, 60], 'discomfort': [1, 5]},
                'limitations': 'Hypothetical intentions only; not physical need or calibrated real-world probability'}
    return {'oof': metrics(predictions), 'prior_baseline_oof': metrics(baseline),
            'folds': folds, 'n_rows': len(long), 'n_respondents': len(set(groups)),
            'carry_rows': int(y.sum()), 'features': columns}, artifact


def survey_analysis(path, out):
    survey = pd.read_csv(path, dtype=str, encoding='utf-8-sig')
    survey_validation(survey)
    result = {'n_respondents': len(survey), 'paired_tests': [
        paired_test(survey, '이동10분', '이동60분'),
        paired_test(survey, '이동시간비없음', '이동시간비있음')]}
    ps = np.array([p['exact_mcnemar_p'] for p in result['paired_tests']])
    order = np.argsort(ps)
    adjusted = np.maximum.accumulate(ps[order] * np.arange(len(ps), 0, -1)).clip(0, 1)
    for index, corrected in zip(order, adjusted):
        result['paired_tests'][int(index)]['holm_p'] = float(corrected)
    distributions, long = [], []
    for scenario in SCENARIOS:
        for choice in sorted(VALID_CHOICES):
            n = int(survey[scenario].eq(choice).sum())
            distributions.append({'scenario': scenario, 'choice': choice, 'n': n,
                                  'denominator': len(survey), 'rate': n / len(survey)})
    pd.DataFrame(distributions).to_csv(out / 'survey_scenarios.csv', index=False, encoding='utf-8-sig')
    for i, row in survey.iterrows():
        for scenario, minutes in [('이동10분', 10), ('이동60분', 60)]:
            long.append({'respondent': i, 'outdoor_minutes': minutes,
                         'carry_discomfort': row['우산짐불편도'], 'wet_discomfort': row['비맞음불편도'],
                         'choice': row[scenario], 'y': int(row[scenario] == '챙긴다')})
    long = pd.DataFrame(long)
    result['model_target'] = '챙긴다=1; 챙기지 않는다/고민된다=0 (sensitivity excludes 고민된다)'
    result['cv'] = '5-fold StratifiedGroupKFold by row-derived respondent ID; shuffle=True, random_state=42'
    result['models'] = {}
    artifacts = {}
    for name, frame, features in [('time_only', long, ['outdoor_minutes']),
                                   ('personalized', long, FEATURES),
                                   ('decisive_only', long[long.choice.ne('고민된다')].reset_index(drop=True), FEATURES)]:
        metrics, artifact = grouped_model(frame, features)
        result['models'][name] = metrics
        artifacts[name] = artifact
    dump(out / 'intention_model.json', artifacts['personalized'])
    # Verify exported arithmetic against sklearn pipeline prediction, independently.
    all_fit = make_pipeline(StandardScaler(), LogisticRegression(C=1., max_iter=2000, random_state=42))
    all_fit.fit(long[FEATURES].to_numpy(dtype=float), long.y.to_numpy())
    for i, row in long.iterrows():
        exported = intention_score(artifacts['personalized'], pop=50,
                                   **{k: float(row[k]) for k in FEATURES})['score']
        expected = 100 * all_fit.predict_proba(long.loc[[i], FEATURES].to_numpy(dtype=float))[0, 1]
        if abs(exported - expected) > .005001:
            raise AssertionError('Exported score differs from fitted estimator')
    diff = survey['비맞음불편도'] - survey['우산짐불편도']
    preference = []
    for label, mask in [('wet_more', diff > 0), ('equal', diff == 0), ('carry_more', diff < 0)]:
        group = survey[mask]
        for scenario in ['이동10분', '이동60분']:
            carry = int(group[scenario].eq('챙긴다').sum())
            preference.append({'preference_group': label, 'scenario': scenario, 'n': len(group),
                               'carry': carry, 'carry_rate': ratio(carry, len(group))})
    pd.DataFrame(preference).to_csv(out / 'survey_preference_groups.csv', index=False, encoding='utf-8-sig')
    result['preference_groups'] = preference
    result['notes'] = ['80 row IDs are proxies for respondents, not verified unique people.',
                       'No join key between weather observations and survey responses. No arbitrary cross-join.',
                       'Q1/Q2 hold POP constant at 50%; effect of POP or PCP cannot be learned.',
                       'Group CV is internal validation, not independent external validation.',
                       'Personalized vs time-only comparison is exploratory; no tuning/model selection claim.']
    dump(out / 'survey_metrics.json', result)
    return result


def charts(weather, survey, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'Malgun Gothic', 'axes.unicode_minus': False, 'font.size': 11})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    matrix = np.array(weather['confusion_matrix'])
    axes[0].imshow(matrix, cmap='Blues', vmin=0, vmax=max(1, matrix.max()))
    axes[0].set(xticks=[0, 1], xticklabels=['예상 강수 없음', '예상 강수 있음'],
                yticks=[0, 1], yticklabels=['실측 강수 없음', '실측 강수 있음'],
                title=f'예보 오차행렬 · POP ≥ 50%\n선택된 관측 시간 {weather["evaluated_rows"]}개')
    for i in range(2):
        for j in range(2):
            axes[0].text(j, i, f'{matrix[i,j]}', ha='center', va='center', fontsize=22,
                         color='white' if matrix[i,j] > matrix.max() / 2 else '#152c45')
    tests = survey['paired_tests']
    vals = [100 * tests[0][k] / survey['n_respondents'] for k in ['before_carry', 'after_carry']]
    bars = axes[1].bar(['10분', '60분'], vals, color=['#aac5e3', '#346eab'], edgecolor='#23496d')
    axes[1].set(ylim=(0, 100), ylabel='챙긴다 응답 비율 (%)', title=f'POP 50% 가상 상황의 우산 지참 의향\n동일 설문 응답 {survey["n_respondents"]}행의 대응 비교')
    axes[1].bar_label(bars, labels=[f'{v:g}%' for v in vals], padding=5)
    axes[1].spines[['top', 'right']].set_visible(False)
    fig.savefig(out / 'analysis_summary.png', dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weather', type=Path, required=True)
    parser.add_argument('--survey', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=Path('results/sohyun'))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    w = weather_analysis(args.weather, args.out)
    s = survey_analysis(args.survey, args.out)
    charts(w, s, args.out)
    dump(args.out / 'provenance.json', {
        'weather': {'file': args.weather.name, 'sha256': hashlib.sha256(args.weather.read_bytes()).hexdigest(),
                    'source': 'https://app.notion.com/p/7-3f9234f14c7d80e3b958fc365d0de062'},
        'survey': {'file': args.survey.name, 'sha256': hashlib.sha256(args.survey.read_bytes()).hexdigest(),
                   'source': 'https://github.com/shuu-uz/UmbrellAI/blob/88d24e2636e76b2c5506c40185409efb22f068a5/survey_analysis/cleaned_responses.csv'},
        'versions': {p: importlib.metadata.version(p) for p in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'matplotlib']},
        'analysis_date': '2026-11-02', 'raw_data_committed': False})
    print(json.dumps({'weather_counts': w['counts'], 'survey_personalized_oof': s['models']['personalized']['oof']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
