"""Synthetic correctness cases, not evidence of empirical model performance."""
import unittest
import pandas as pd
from src.run_analysis import select_forecasts, paired_test, survey_validation
from src.intention_model import FEATURES, intention_score
from src.recommendation import recommend_from_travel_forecasts
from src.run_analysis import preference_association


def weather_rows():
    return pd.DataFrame([
        dict(date='20261011', time='1500', nx='58', ny='121', baseDate='20261010',
             baseTime='1400', lead_hours='25', POP='0', real_rain='0', rain_missing='0'),
        dict(date='20261011', time='1500', nx='58', ny='121', baseDate='20261011',
             baseTime='1400', lead_hours='1', POP='60', real_rain='0', rain_missing='0')])


class WorkflowChecks(unittest.TestCase):
    def test_latest_available_issuance_and_audit(self):
        selected, audit = select_forecasts(weather_rows())
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected.iloc[0].POP, '60')
        self.assertEqual(audit['superseded_rows'], 1)

    def test_late_issuance_not_used(self):
        rows = weather_rows()
        rows.loc[1, ['baseTime', 'lead_hours']] = ['1500', '0']
        selected, audit = select_forecasts(rows)
        self.assertEqual(selected.iloc[0].POP, '0')
        self.assertEqual(audit['ineligible_issuance_rows'], 1)

    def test_duplicate_issuance_rejected(self):
        rows = weather_rows()
        with self.assertRaisesRegex(ValueError, 'Duplicate issuance'):
            select_forecasts(pd.concat([rows, rows.iloc[:1]]))

    def test_conflicting_observation_rejected(self):
        rows = weather_rows()
        rows.loc[1, 'real_rain'] = '1'
        with self.assertRaisesRegex(ValueError, 'Conflicting observation'):
            select_forecasts(rows)

    def test_malformed_time_and_inconsistent_lead_rejected(self):
        rows = weather_rows()
        rows.loc[0, 'time'] = '9999'
        with self.assertRaises(ValueError):
            select_forecasts(rows)
        rows = weather_rows()
        rows.loc[0, 'lead_hours'] = '1'
        with self.assertRaisesRegex(ValueError, 'lead_hours conflicts'):
            select_forecasts(rows)

    def test_paired_exact_test(self):
        survey = pd.DataFrame({'a': ['챙기지 않는다'] * 5 + ['챙긴다'],
                               'b': ['챙긴다'] * 6})
        result = paired_test(survey, 'a', 'b')
        self.assertEqual((result['gain'], result['loss']), (5, 0))
        self.assertEqual(result['exact_mcnemar_p'], .0625)

    def test_score_domain_and_probability(self):
        model = dict(feature_names=FEATURES, target='stated_carry_intention',
                     coefficients=[0, 0, 0], intercept=0, mean=[0, 0, 0], scale=[1, 1, 1])
        kwargs = dict(pop=50, outdoor_minutes=10, carry_discomfort=3, wet_discomfort=4)
        self.assertEqual(intention_score(model, **kwargs)['score'], 50)
        self.assertIsNone(intention_score(model, **{**kwargs, 'pop': 60})['score'])
        self.assertIsNone(intention_score(model, **{**kwargs, 'outdoor_minutes': 30})['score'])
        with self.assertRaises(ValueError):
            intention_score(model, **{**kwargs, 'wet_discomfort': float('nan')})
        with self.assertRaises(ValueError):
            intention_score(model, **{**kwargs, 'carry_discomfort': 6})

    def test_travel_forecasts_require_window_and_supported_weather(self):
        model = dict(feature_names=FEATURES, target='stated_carry_intention',
                     coefficients=[0, 0, 0], intercept=0, mean=[0, 0, 0], scale=[1, 1, 1])
        kwargs = dict(outdoor_minutes=10, carry_discomfort=4, wet_discomfort=4)
        f = recommend_from_travel_forecasts
        self.assertEqual(f([{'POP': 50}], model, **kwargs)['status'], 'travel_window_required')
        kwargs['travel_window_confirmed'] = True
        self.assertEqual(f([{'POP': 50}], model, **kwargs)['score'], 50)
        self.assertIsNone(f([{'POP': 0}, {'POP': 50}], model, **kwargs)['score'])
        self.assertIsNone(f([{'POP': float('nan')}], model, **kwargs)['score'])
        self.assertIsNone(f([], model, **kwargs)['score'])
        self.assertEqual(f([{'POP': 50, 'PTY': 2}], model, **kwargs)['status'],
                         'unsupported_precipitation_type')

    def test_zero_exposure_and_unmeasured_duration(self):
        kwargs = dict(carry_discomfort=4, wet_discomfort=4, travel_window_confirmed=True)
        f = recommend_from_travel_forecasts
        self.assertEqual(f([], {}, outdoor_minutes=0, **kwargs)['status'], 'no_outdoor_exposure')
        self.assertIsNone(f([{'POP': 50}], {}, outdoor_minutes=30, **kwargs)['score'])
        with self.assertRaises(ValueError):
            f([], {}, outdoor_minutes=-1, **kwargs)

    def test_respondent_preference_association_and_constant_case(self):
        data = pd.DataFrame({'우산짐불편도': [5]*5+[1]*5,
                             '비맞음불편도': [1]*5+[5]*5,
                             '이동10분': ['챙기지 않는다']*5+['챙긴다']*5,
                             '이동60분': ['챙기지 않는다']*5+['챙긴다']*5})
        r = preference_association(data, permutations=1000)
        self.assertAlmostEqual(r['spearman_rho'], 1)
        self.assertLess(r['permutation_p_two_sided'], .05)
        data['비맞음불편도'] = data['우산짐불편도']
        self.assertIsNone(preference_association(data)['spearman_rho'])


if __name__ == '__main__':
    unittest.main()
