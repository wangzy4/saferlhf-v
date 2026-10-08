#!/usr/bin/env python3
"""Download only the LLaVA policy pair and BeaverTails-V evaluation parquets."""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download

SAFE_REV = '9fa9092c3d09484039e299cb92dcf9fe945075fa'
DATA_REV = 'ee19041205c720c0faea575de563de8a6a8f9094'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    manifest_path = root / 'assets.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        manifest = {'base_repo': 'llava-hf/llava-1.5-7b-hf',
                    'base_revision': api.model_info('llava-hf/llava-1.5-7b-hf').sha,
                    'safe_repo': 'saferlhf-v/SafeRLHF-V', 'safe_revision': SAFE_REV,
                    'data_repo': 'saferlhf-v/BeaverTails-V', 'data_revision': DATA_REV}
        manifest_path.write_text(json.dumps(manifest, indent=2))
    common = dict(cache_dir=str(root / 'hf-cache'), max_workers=4)
    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs = [
            pool.submit(snapshot_download, manifest['base_repo'], revision=manifest['base_revision'],
                        local_dir=root / 'models/base',
                        allow_patterns=['*.json', '*.model', '*.safetensors'], **common),
            pool.submit(snapshot_download, manifest['safe_repo'], revision=SAFE_REV,
                        local_dir=root / 'models/safe',
                        allow_patterns=['LLaVA_Safe_RLHF-V/*'], **common),
            pool.submit(snapshot_download, manifest['data_repo'], repo_type='dataset', revision=DATA_REV,
                        local_dir=root / 'datasets/beavertails-v',
                        allow_patterns=['README.md', 'data/*/evaluation*'], **common),
        ]
        for job in jobs:
            job.result()
    (root / 'logs/assets-ready').touch()
    print('Assets ready', flush=True)


if __name__ == '__main__':
    main()
