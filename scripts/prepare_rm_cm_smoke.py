#!/usr/bin/env python3
"""Prepare private small train/internal-validation sets; never train on evaluation."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoConfig, AutoProcessor
from download_weights_ranges import download
from safe_rlhf_v.utils.preference_training import image_from_value, image_key, response_text
from safe_rlhf_v.utils.processors import configure_llava_processor

REVISION = 'ee19041205c720c0faea575de563de8a6a8f9094'
CATEGORIES = ['animal_abuse', 'dangerous_behavior', 'false_information',
              'financial_and_academic_fraud', 'privacy_invasion_and_surveillance']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--name', default='rm-cm-smoke-128-32')
    p.add_argument('--train-size', type=int, default=128)
    p.add_argument('--validation-size', type=int, default=32)
    p.add_argument('--max-length', type=int, default=2048)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--all-categories', action='store_true',
                   help='Use all 20 categories identified by the fixed evaluation file layout')
    p.add_argument('--exclude-dataset', action='append', default=[],
                   help='Exclude all images used by a previous prepared dataset (repeatable)')
    args = p.parse_args()
    root = args.root.resolve()
    evaluation_files = sorted((root / 'datasets/beavertails-v/data').glob('*/evaluation*.parquet'))
    if len(evaluation_files) != 20 or len({f.parent.name for f in evaluation_files}) != 20:
        raise RuntimeError('Need all 20 fixed evaluation files for overlap exclusion')
    categories = [f.parent.name for f in evaluation_files] if args.all_categories else CATEGORIES
    if min(args.train_size, args.validation_size) < len(categories):
        p.error('Both splits must contain all selected categories')
    if not 576 < args.max_length <= 4096:
        p.error('max-length must include visual tokens and stay within base context')
    target = root / 'datasets' / args.name
    if target.exists():
        raise RuntimeError('Dataset target exists; use a new name instead of overwriting')
    with ThreadPoolExecutor(max_workers=5) as pool:
        jobs = [pool.submit(download, root, 'saferlhf-v/BeaverTails-V', REVISION,
                            f'data/{cat}/train.parquet', 'datasets/beavertails-v', 8,
                            repo_type='dataset') for cat in categories]
        for job in jobs:
            job.result()
    evaluation_images = {image_key(r['image']) for f in evaluation_files
                         for r in pq.read_table(f, columns=['image']).to_pylist()}
    previous_images, previous_hashes = set(), {}
    for name in args.exclude_dataset:
        previous = root / 'datasets' / name
        if not (previous / 'READY').exists():
            raise RuntimeError(f'Excluded dataset not ready: {name}')
        prior_manifest = json.loads((previous / 'manifest.json').read_text())
        previous_hashes[name] = prior_manifest['file_sha256']
        for split in ('train', 'validation'):
            path = previous / f'{split}.parquet'
            with path.open('rb') as handle:
                if hashlib.file_digest(handle, 'sha256').hexdigest() != prior_manifest['file_sha256'][split]:
                    raise RuntimeError(f'Excluded dataset checksum mismatch: {name}/{split}')
            previous_images.update(image_key(r['image']) for r in
                                   pq.read_table(path, columns=['image']).to_pylist())
    pools, excluded, source_hashes = {}, Counter(), {}
    for cat in categories:
        source = root / f'datasets/beavertails-v/data/{cat}/train.parquet'
        source_hashes[cat] = source.with_name('train.parquet.sha256').read_text().strip()
        rows = pq.read_table(source).to_pylist()
        candidates = []
        for i, row in enumerate(rows):
            if not all(int(row[k]) in (1, 2) for k in ['more_helpful_response_id', 'safer_response_id']):
                excluded['invalid_preference_id'] += 1
                continue
            if row['response_1'].strip() == row['response_2'].strip():
                excluded['identical_responses'] += 1
                continue
            key = image_key(row['image'])
            if key in evaluation_images:
                excluded['evaluation_image_overlap_rows'] += 1
                continue
            for response in (1, 2):
                rating = row[f'response_{response}_harmless_rate']
                safe = row[f'is_response_{response}_safe']
                if safe not in ('yes', 'no') or not -3 <= rating <= 3:
                    raise RuntimeError(f'Invalid safety annotation: {cat}/{i}')
                if (rating >= 0) != (safe == 'yes'):
                    raise RuntimeError(f'Rating/safety annotation conflict: {cat}/{i}')
            if key in previous_images:
                excluded['previous_dataset_image_overlap_rows'] += 1
                continue
            candidates.append({**row, 'category': cat, 'source_id': f'{cat}/train/{i}',
                               'image_hash': key})
        random.Random(args.seed + categories.index(cat)).shuffle(candidates)
        pools[cat] = candidates
        print('Eligible candidates', cat, len(candidates), flush=True)
    base = root / 'models/base'
    processor = AutoProcessor.from_pretrained(base, use_fast=False, local_files_only=True)
    configure_llava_processor(processor, AutoConfig.from_pretrained(base, local_files_only=True))
    processor.tokenizer.padding_side = 'right'
    processor.tokenizer.model_max_length = args.max_length
    selected_images, splits, lengths = set(), {}, []
    for split, size in [('validation', args.validation_size), ('train', args.train_size)]:
        selected = []
        for index in range(size):
            cat = categories[index % len(categories)]
            while pools[cat]:
                row = pools[cat].pop()
                if row['image_hash'] in selected_images:
                    excluded['duplicate_selected_image'] += 1
                    continue
                texts = [response_text(processor, row['question'], row[f'response_{i}'])
                         for i in (1, 2)]
                image = image_from_value(row['image'])
                inputs = processor(text=texts, images=[image, image], padding=True, return_tensors='pt')
                width = inputs.input_ids.shape[1]
                if width > args.max_length:
                    excluded['over_length_candidates'] += 1
                    continue
                if not (inputs.input_ids == 32000).sum(1).eq(576).all():
                    raise RuntimeError('Unexpected LLaVA image token expansion')
                selected_images.add(row['image_hash'])
                selected.append(row)
                lengths.append(width)
                break
            else:
                raise RuntimeError(f'Not enough eligible image-disjoint rows for {cat}')
        splits[split] = selected
    target.mkdir(parents=True)
    file_hashes = {}
    for split, rows in splits.items():
        path = target / f'{split}.parquet'
        pq.write_table(pa.Table.from_pylist(rows), path)
        with path.open('rb') as f:
            file_hashes[split] = hashlib.file_digest(f, 'sha256').hexdigest()
    manifest = {'data_repo': 'saferlhf-v/BeaverTails-V', 'revision': REVISION,
                'source_split': 'train', 'categories': categories, 'seed': args.seed,
                'n_train': len(splits['train']), 'n_validation': len(splits['validation']),
                'categories_by_split': {s: dict(Counter(r['category'] for r in rows))
                                        for s, rows in splits.items()},
                'evaluation_images_checked': len(evaluation_images),
                'selected_evaluation_image_overlap': 0, 'train_validation_image_overlap': 0,
                'excluded': dict(excluded), 'max_selected_tokens': max(lengths),
                'max_length': args.max_length, 'file_sha256': file_hashes,
                'source_train_sha256': source_hashes,
                'excluded_datasets': previous_hashes,
                'previous_dataset_images_checked': len(previous_images),
                'selected_previous_dataset_image_overlap': 0,
                'scope': f'{len(categories)} source categories; exact decoded-image overlap, not semantic dedup'}
    (target / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    (target / 'READY').touch()
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()
