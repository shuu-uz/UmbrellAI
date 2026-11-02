import unittest
from src.analysis import evaluate, recommend_umbrella


def samples():
    return [dict(fcstDate='20261008', fcstTime=f'{i:02}00', nx=60, ny=127,
                 POP=pop, actual_mm=rain, rain_missing=missing)
            for i, (pop, rain, missing) in enumerate([
                (80, 2, 0), (80, 0, 0), (10, 2, 0), (10, 0, 0), (80, None, 1)])]


class Checks(unittest.TestCase):
    def test_known_confusion_and_exclusion(self):
        r = evaluate(samples(), rain_column='actual_mm', synthetic=True)
        self.assertEqual(r['confusion_matrix'], [[1, 1], [1, 1]])
        self.assertEqual(r['excluded']['observation_missing'], 1)
        self.assertEqual(r['miss_rate'], .5)
        self.assertEqual(r['false_alarm_ratio'], .5)
        self.assertEqual(r['data_kind'], 'SYNTHETIC_ONLY')

    def test_duplicates_rejected(self):
        with self.assertRaises(ValueError):
            evaluate(samples()+samples()[:1], rain_column='actual_mm')

    def test_current_collector_and_observation_columns(self):
        row = dict(date='20261008', time='0900', nx=58, ny=121,
                   POP=80, PCP=2, PTY=1, real_rain=1, rain_missing=0,
                   base_date='20261008', base_time='0800')
        self.assertEqual(evaluate([row], synthetic=True)['counts']['TP'], 1)
        self.assertEqual(recommend_umbrella([row], 20)['level'], 'recommended')

    def test_undefined_ratio(self):
        self.assertIsNone(evaluate([], rain_column='actual_mm')['miss_rate'])

    def test_invalid_and_null_excluded(self):
        rows = samples()[:2]
        rows[0]['POP'] = float('nan')
        rows[1]['actual_mm'] = None
        self.assertEqual(evaluate(rows, rain_column='actual_mm')['evaluated_rows'], 0)

    def test_recommendations(self):
        self.assertEqual(recommend_umbrella([dict(POP=80, pcp_mm=0)], 20)['level'], 'recommended')
        self.assertEqual(recommend_umbrella([dict(POP=80, pcp_mm=0)], 5)['level'], 'consider')
        self.assertEqual(recommend_umbrella([dict(POP=None, pcp_mm=0)], 20)['level'], 'insufficient_data')


if __name__ == '__main__':
    unittest.main()
