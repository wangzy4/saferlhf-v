"""CPU-only release privacy and public-fallback contracts."""
import unittest
from types import SimpleNamespace
from scripts.export_score_artifacts import PRIVATE, private_quota_error, sanitize_config


class ArtifactExportTests(unittest.TestCase):
    def test_only_model_path_fields_are_rewritten(self):
        original = {'base_model_name_or_path': '/data/private/base',
                    'nested': {'_name_or_path': '/home/private/model', 'safe': 'sdpa'},
                    'unexpected': '/data/private/secret'}
        sanitized = sanitize_config(original)
        self.assertEqual(sanitized['base_model_name_or_path'], 'llava-hf/llava-1.5-7b-hf')
        self.assertFalse(PRIVATE.search(sanitized['nested']['_name_or_path']))
        # Unknown private fields fail the subsequent scan; never silently discard them.
        self.assertTrue(PRIVATE.search(sanitized['unexpected']))
        self.assertEqual(original['nested']['_name_or_path'], '/home/private/model')

    def test_public_fallback_requires_private_storage_quota(self):
        class Failure(Exception):
            def __init__(self, status, message):
                super().__init__(message)
                self.response = SimpleNamespace(status_code=status)
        self.assertTrue(private_quota_error(Failure(403, 'Private storage quota exceeded')))
        for status, message in ((403, 'Permission denied'), (500, 'Private storage quota exceeded'),
                                (429, 'Rate limit'), (403, 'Public storage quota exceeded')):
            self.assertFalse(private_quota_error(Failure(status, message)))


if __name__ == '__main__':
    unittest.main()
