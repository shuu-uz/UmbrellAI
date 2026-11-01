"""Synthetic correctness cases, not evidence of empirical model performance."""
import unittest
import pandas as pd
from src.run_analysis import select_forecasts, paired_test, survey_validation
from src.intention_model import FEATURES, intention_score


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


if __name__ == '__main__':
    unittest.main()
