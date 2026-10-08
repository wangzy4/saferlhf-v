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

    def test_batch_time_is_not_counted_per_sample(self):
        records = []
        for tokens, size, seconds in [(2, 2, 4), (4, 2, 4), (3, 1, 1)]:
            records.append({'generated_tokens': tokens, 'batch_size': size,
                            'batch_generation_seconds': seconds, 'hit_token_limit': False,
                            'response': 'Normal response.', 'helpful_id': '1', 'safer_id': '2',
                            'preference_likelihood': None})
        result = summary.summarize(records)
        self.assertEqual(result['summed_batch_generation_seconds'], 5)
        self.assertAlmostEqual(result['amortized_generation_seconds_per_sample'], 5/3)
        self.assertAlmostEqual(result['generated_tokens_per_replica_second'], 9/5)

    def test_compact_loop(self):
        self.assertTrue(summary.repeated_span('The' * 30))
        self.assertTrue(summary.repeated_span('In ' * 30))
        self.assertFalse(summary.repeated_span('A normal descriptive response.'))

    def test_refusal_is_only_marker(self):
        self.assertTrue(summary.REFUSAL.search('I cannot provide instructions.'))
        self.assertFalse(summary.REFUSAL.search('Consider wearing gloves.'))


if __name__ == '__main__':
    unittest.main()
