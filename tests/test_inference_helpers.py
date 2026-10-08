import ast
import unittest
from pathlib import Path

# Extract pure helpers only, so source tests don't import torch/transformers.
path = Path(__file__).resolve().parents[1] / 'scripts/infer_llava_pair.py'
tree = ast.parse(path.read_text())
helpers = ast.Module(body=[node for node in tree.body if isinstance(node, ast.FunctionDef)
                          and node.name in {'prompt_for', 'validate_transformers_version'}], type_ignores=[])
namespace = {}
exec(compile(helpers, str(path), 'exec'), namespace)


class InferenceHelpersTests(unittest.TestCase):
    def test_version_guard(self):
        check = namespace['validate_transformers_version']
        for version in ['4.36.0', '4.47.1']:
            with self.assertRaisesRegex(RuntimeError, 'cached left-padded'):
                check(version)
        check('4.48.3')

    def test_manual_prompt(self):
        self.assertEqual(namespace['prompt_for']('Question?'), 'USER: <image>\nQuestion?\nASSISTANT:')

    def test_hf_template_not_silent_fallback(self):
        with self.assertRaises(RuntimeError):
            namespace['prompt_for']('Question?', None, 'hf')
        class Processor:
            chat_template = 'present'
            def apply_chat_template(self, messages, **kwargs):
                self.messages, self.kwargs = messages, kwargs
                return 'official prompt'
        processor = Processor()
        self.assertEqual(namespace['prompt_for']('Question?', processor, 'hf'), 'official prompt')
        self.assertEqual(processor.messages[0]['content'][0]['type'], 'image')
        self.assertTrue(processor.kwargs['add_generation_prompt'])
        self.assertFalse(processor.kwargs['tokenize'])


if __name__ == '__main__':
    unittest.main()
