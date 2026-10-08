#!/usr/bin/env python3
"""Fallback resumable HTTP range downloads, with mandatory LFS SHA256 checks.

Use when the HF CDN is slow or hf_transfer/Xet repeatedly fails. No weights are
written outside --root. A completed range is recorded only after the data is fsynced.
"""
import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

CHUNK = 32 * 1024 * 1024


def download(root, repo, revision, filename, target_dir, workers, repo_type='model'):
    if repo_type not in {'model', 'dataset'}:
        raise ValueError('repo_type must be model or dataset')
    collection = 'models' if repo_type == 'model' else 'datasets'
    metadata = requests.get(f'https://huggingface.co/api/{collection}/{repo}/revision/{revision}',
                            params={'blobs': 'true'}, timeout=30)
    metadata.raise_for_status()
    info = next(s for s in metadata.json()['siblings'] if s['rfilename'] == filename)
    size, digest = info['lfs']['size'], info['lfs']['sha256']
    target = root / target_dir / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + '.range-partial')
    markers = target.with_name(target.name + '.range-chunks')
    markers.mkdir(exist_ok=True)
    if target.exists():
        with target.open('rb') as handle:
            actual = hashlib.file_digest(handle, 'sha256').hexdigest()
        if actual == digest:
            print('Verified existing', filename, flush=True)
            return
        raise RuntimeError(f'Existing file checksum mismatch: {target}')
    fd = os.open(partial, os.O_CREAT | os.O_RDWR, 0o600)
    if os.fstat(fd).st_size != size:
        os.ftruncate(fd, size)
        for marker in markers.iterdir():
            marker.unlink()
    prefix = '' if repo_type == 'model' else 'datasets/'
    resolve = f'https://huggingface.co/{prefix}{repo}/resolve/{revision}/{filename}'
    chunks = list(range((size + CHUNK - 1) // CHUNK))

    def worker(index):
        if (markers / str(index)).exists():
            return
        start, end = index * CHUNK, min((index + 1) * CHUNK, size) - 1
        for attempt in range(8):
            try:
                with requests.get(resolve, params={'download': 'true', 'range_chunk': index},
                                  headers={'Range': f'bytes={start}-{end}', 'Accept-Encoding': 'identity'},
                                  stream=True, timeout=(20, 60)) as response:
                    response.raise_for_status()
                    expected = f'bytes {start}-{end}/{size}'
                    if response.status_code != 206 or response.headers.get('Content-Range') != expected:
                        raise RuntimeError(f'Range mismatch: {response.status_code} '
                                           f'{response.headers.get("Content-Range")} != {expected}')
                    offset = start
                    for data in response.iter_content(1024 * 1024):
                        if offset + len(data) > end + 1:
                            raise RuntimeError('Range overrun')
                        view = memoryview(data)
                        while view:
                            written = os.pwrite(fd, view, offset)
                            if not written:
                                raise OSError('Short file write')
                            offset += written
                            view = view[written:]
                    if offset != end + 1:
                        raise RuntimeError('Incomplete range')
                os.fsync(fd)
                (markers / str(index)).touch()
                if index % 10 == 0:
                    print(f'{filename}: {len(list(markers.iterdir()))}/{len(chunks)} chunks', flush=True)
                return
            except Exception as exc:
                print(f'{filename} chunk {index}, retry {attempt+1}: {exc}', flush=True)
                if attempt == 7:
                    raise
                time.sleep(min(2**attempt, 20))

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(worker, chunks))
    finally:
        os.close(fd)
    with partial.open('rb') as handle:
        actual = hashlib.file_digest(handle, 'sha256').hexdigest()
    if actual != digest:
        raise RuntimeError(f'SHA256 mismatch: {filename}: {actual} != {digest}')
    partial.replace(target)
    (target.with_name(target.name + '.sha256')).write_text(digest + '\n')
    print('VERIFIED', filename, digest, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--workers-per-file', type=int, default=8)
    parser.add_argument('--safe-workers', type=int, default=0,
                        help='Optional separate concurrency for the single large policy file')
    args = parser.parse_args()
    manifest = json.loads((args.root / 'assets.json').read_text())
    jobs = [(manifest['base_repo'], manifest['base_revision'],
             f'model-{index:05d}-of-00003.safetensors', 'models/base') for index in range(1, 4)]
    jobs.append((manifest['safe_repo'], manifest['safe_revision'],
                 'LLaVA_Safe_RLHF-V/pytorch_model.bin', 'models/safe'))
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(download, args.root, *job,
                               args.safe_workers if job[3] == 'models/safe' and args.safe_workers
                               else args.workers_per_file) for job in jobs]
        for future in futures:
            future.result()
    (args.root / 'logs/assets-ready').touch()
    print('All weights downloaded and SHA256 verified', flush=True)


if __name__ == '__main__':
    main()
