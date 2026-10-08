import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('summary_under_test', ROOT / 'scripts/summarize_pair.py')
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


class SummaryTests(unittest.TestCase):
    def test_preference_orientation(self):
        record = {'helpful_id': '1', 'safer_id': '2', 'preference_likelihood': [
            {'mean_logp': -2, 'sum_logp': -20}, {'mean_logp': -3, 'sum_logp': -15}]}
        self.assertEqual(summary.agreement(record, 'helpful_id', 'mean_logp'), 1)
        self.assertEqual(summary.agreement(record, 'safer_id', 'mean_logp'), 0)
        self.assertEqual(summary.agreement(record, 'safer_id', 'sum_logp'), 1)
        record['safer_id'] = '0'
        self.assertIsNone(summary.agreement(record, 'safer_id', 'mean_logp'))

    def test_ci_and_repetition(self):
        low, high = summary.wilson(5, 10)
        self.assertLess(low, 0.5)
        self.assertGreater(high, 0.5)
        self.assertEqual(summary.repetition('a b c d'), 0)
        self.assertGreater(summary.repetition('a b c a b c a b c'), 0)

    def test_refusal_is_only_marker(self):
        self.assertTrue(summary.REFUSAL.search('I cannot provide instructions.'))
        self.assertFalse(summary.REFUSAL.search('Consider wearing gloves.'))


if __name__ == '__main__':
    unittest.main()
