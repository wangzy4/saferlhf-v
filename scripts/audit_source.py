#!/usr/bin/env python3
"""Dependency-free source audit; does not import training code or access GPUs."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    files = sorted((ROOT / 'safe_rlhf_v').rglob('*.py'))
    for path in files:
        ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    print(f'PASS: {len(files)} Python files parse successfully (not a runtime check).')
    failures = []
    for path in sorted((ROOT / 'safe_rlhf_v/trainers').rglob('*.py')):
        tree = ast.parse(path.read_text(encoding='utf-8'))
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef) or node.name != 'main':
                continue
            # Select keys and values, plus dict(zip(...)), never argparse/distributed setup.
            assignments = [
                statement for statement in node.body
                if isinstance(statement, ast.Assign)
                and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)
                and (statement.targets[0].id in {'keys', 'values'}
                     or (statement.targets[0].id == 'unparsed_args'
                         and isinstance(statement.value, ast.Call)
                         and isinstance(statement.value.func, ast.Name)
                         and statement.value.func.id == 'dict'))
            ]
            if not assignments:
                continue
            env = {'unparsed_args': ['--model_name_or_path', 'base', '--epochs', '3']}
            module = ast.Module(body=assignments, type_ignores=[])
            exec(compile(module, str(path), 'exec'), env)
            result = env['unparsed_args']
            expected = {'model_name_or_path': 'base', 'epochs': '3'}
            if result != expected:
                failures.append(path.relative_to(ROOT))
                print(f'FAIL: {path.relative_to(ROOT)}: {result!r}; expected {expected!r}')
    print(f'CLI parsing failures: {len(failures)}. Upstream defects; no training executed.')
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
