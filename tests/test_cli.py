import ast
import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('cli_under_test', ROOT / 'safe_rlhf_v/utils/cli.py')
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)
apply_cli_overrides = cli.apply_cli_overrides
parse_cli_overrides = cli.parse_cli_overrides


class CliTests(unittest.TestCase):
    def setUp(self):
        self.config = {'train_cfgs': {'epochs': 3, 'bf16': True, 'threshold': -0.5,
                                      'adam_betas': [0.9, 0.95]},
                       'model_cfgs': {'model_name_or_path': None}}

    def test_first_flag_and_types(self):
        result = apply_cli_overrides(self.config, ['--model_name_or_path', 'base', '--epochs', '2',
                                                   '--threshold', '-1.5', '--bf16', 'false'])
        self.assertEqual(result['model_cfgs']['model_name_or_path'], 'base')
        self.assertIs(type(result['train_cfgs']['epochs']), int)
        self.assertEqual(result['train_cfgs']['threshold'], -1.5)
        self.assertIs(result['train_cfgs']['bf16'], False)
        self.assertEqual(self.config['train_cfgs']['epochs'], 3)

    def test_launcher_and_equals(self):
        self.assertEqual(parse_cli_overrides(['--local_rank=0', '--epochs=2']), {'epochs': '2'})
        self.assertEqual(parse_cli_overrides(['--local-rank', '0', '--epochs', '2']), {'epochs': '2'})

    def test_empty(self):
        self.assertEqual(apply_cli_overrides(self.config, []), self.config)

    def test_nested_and_list(self):
        result = apply_cli_overrides(self.config, ['--train-cfgs:epochs=4', '--adam_betas', '[0.8, 0.9]'])
        self.assertEqual(result['train_cfgs']['epochs'], 4)
        self.assertEqual(result['train_cfgs']['adam_betas'], [0.8, 0.9])

    def test_bad_values(self):
        for tokens in [['--epochs'], ['--epochs', '--bf16', 'True'], ['--epochs', '2.5'],
                       ['--bf16', 'maybe'], ['--unknown', '1'], ['--epochs', '2', '--epochs', '3'],
                       ['--model_cfgs:unknown=1'], ['positional'], ['--threshold', 'NaN']]:
            with self.subTest(tokens=tokens), self.assertRaises(ValueError):
                apply_cli_overrides(self.config, tokens)

    def test_ambiguous(self):
        with self.assertRaises(ValueError):
            apply_cli_overrides({'a': {'epochs': 1}, 'b': {'epochs': 2}}, ['--epochs', '3'])

    def test_all_entrypoints_use_shared_parser(self):
        mains = 0
        for path in (ROOT / 'safe_rlhf_v/trainers').rglob('*.py'):
            tree = ast.parse(path.read_text())
            for node in tree.body:
                if isinstance(node, ast.FunctionDef) and node.name == 'main':
                    mains += 1
                    calls = [n for n in ast.walk(node) if isinstance(n, ast.Call)
                             and isinstance(n.func, ast.Name) and n.func.id == 'apply_cli_overrides']
                    self.assertEqual(len(calls), 1, str(path))
        self.assertEqual(mains, 9)


if __name__ == '__main__':
    unittest.main()
