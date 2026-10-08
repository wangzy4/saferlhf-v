"""Strict, dependency-free configuration overrides for training entrypoints."""
import ast
import copy
import math


def parse_cli_overrides(tokens):
    """Parse --key value / --key=value, tolerating DeepSpeed's local rank flag."""
    result = {}
    index = 0
    while index < len(tokens):
        flag = tokens[index]
        if not flag.startswith('--') or len(flag) == 2:
            raise ValueError(f'Expected --key, got {flag!r}')
        key, sep, value = flag[2:].partition('=')
        if not sep:
            index += 1
            if index == len(tokens) or tokens[index].startswith('--'):
                raise ValueError(f'Missing value for --{key}')
            value = tokens[index]
        key = key.replace('-', '_')
        if key not in {'local_rank'}:
            if key in result:
                raise ValueError(f'Duplicate override --{key}')
            result[key] = value
        else:
            int(value)  # validate launcher metadata, never insert it into the config
        index += 1
    return result


def _paths(config, key, prefix=()):
    result = []
    for name, value in config.items():
        path = prefix + (name,)
        if name == key:
            result.append(path)
        if isinstance(value, dict):
            result.extend(_paths(value, key, path))
    return result


def _convert(raw, old):
    if isinstance(old, str):
        return raw
    if isinstance(old, bool):
        if raw.lower() not in {'true', 'false'}:
            raise ValueError('Expected true or false')
        return raw.lower() == 'true'
    if isinstance(old, int):
        return int(raw)
    if isinstance(old, float):
        value = float(raw)
        if not math.isfinite(value):
            raise ValueError('Expected a finite float')
        return value
    if isinstance(old, list):
        if raw.startswith('['):
            value = ast.literal_eval(raw)
            if not isinstance(value, list):
                raise ValueError('Expected a list')
            return value
        return [item for item in raw.split(',') if item]
    if old is None:
        if raw.lower() in {'null', 'none'}:
            return None
        # Null model/dataset fields are identifiers, not implicitly numeric.
        return raw
    raise ValueError(f'Unsupported configuration type: {type(old).__name__}')


def apply_cli_overrides(config, tokens):
    """Apply only known, unambiguous keys; input config is never mutated."""
    result = copy.deepcopy(config)
    for key, raw in parse_cli_overrides(tokens).items():
        paths = [tuple(key.split(':'))] if ':' in key else _paths(result, key)
        if len(paths) != 1:
            raise ValueError(f'Unknown or ambiguous configuration key --{key}')
        node = result
        try:
            for part in paths[0][:-1]:
                node = node[part]
            leaf = paths[0][-1]
            old = node[leaf]
        except (KeyError, TypeError) as exc:
            raise ValueError(f'Unknown configuration key --{key}') from exc
        try:
            node[leaf] = _convert(raw, old)
        except (ValueError, SyntaxError) as exc:
            raise ValueError(f'Invalid value for --{key}: {raw!r}') from exc
    return result
