#!/usr/bin/env python3
"""Dependency-free AST and CLI regression audit; no training code is imported."""
import ast
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    files = sorted((ROOT / 'safe_rlhf_v').rglob('*.py'))
    for path in files:
        ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    print(f'PASS: {len(files)} Python files parse successfully (not a runtime check).', flush=True)
    return subprocess.call([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'], cwd=ROOT)


if __name__ == '__main__':
    raise SystemExit(main())
