"""Detached score loss decomposition cannot change optimization or create model grads."""
import unittest

try:
    import torch
    from safe_rlhf_v.utils.preference_training import preference_loss, preference_score_diagnostics
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'torch unavailable')
class ScoreDiagnosticsTests(unittest.TestCase):
    def test_components_match_native_loss_and_leave_original_gradient_untouched(self):
        for dtype in (torch.float32, torch.bfloat16):
            for kind in ('rm', 'cm'):
                original = torch.tensor([[0.5, -0.4], [-0.2, -0.1]], dtype=dtype, requires_grad=True)
                ratings = original.new_tensor([[-2, 3], [0, -1]])
                expected, _ = preference_loss(original, [1, 2], kind, ratings)
                expected_gradient = torch.autograd.grad(expected, original, retain_graph=True)[0]
                result = preference_score_diagnostics(original, [1, 2], kind, ratings)
                torch.testing.assert_close(result['loss_total'], expected.detach(), atol=0, rtol=0)
                torch.testing.assert_close(result['score_gradient_l2_total'], expected_gradient.float().norm(),
                                           atol=0, rtol=0)
                self.assertIsNone(original.grad)
                self.assertTrue(all(not value.requires_grad for value in result.values()))
                expected.backward()
                torch.testing.assert_close(original.grad, expected_gradient, atol=0, rtol=0)

    def test_zero_rating_absolute_gradient_and_evaluation_no_grad_context(self):
        scores = torch.tensor([[-0.5, -0.25], [-0.75, -1.0]], requires_grad=True)
        with torch.no_grad():
            result = preference_score_diagnostics(scores, [1, 2], 'cm', torch.zeros_like(scores))
        self.assertEqual(float(result['score_gradient_l2_absolute']), 0)
        self.assertEqual(float(result['rating_zero_fraction']), 1)
        self.assertEqual(float(result['score_positive_fraction']), 0)
        self.assertGreater(float(result['score_gradient_l2_pairwise']), 0)
        self.assertIsNone(scores.grad)

    def test_nonfinite_diagnostic_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Non-finite'):
            preference_score_diagnostics(torch.tensor([[float('nan'), 1.0]]), [1], 'rm')


if __name__ == '__main__':
    unittest.main()
