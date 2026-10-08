#!/usr/bin/env python3
"""Actual small-data LLaVA RM/CM training with LoRA, not author full-finetuning.

Uses the project's real score architecture and matching losses, no DeepSpeed or
judge. Private datasets/checkpoints/logs stay under --root. CUDA must be explicit.
"""
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pyarrow.parquet as pq
import torch
from peft import LoraConfig, PeftModel, get_peft_model
import peft
import transformers
from transformers import AutoProcessor
from safe_rlhf_v.models.llava import AccustomedLlavaRewardModel
from safe_rlhf_v.utils.preference_training import (
    binary_cost_diagnostics, image_from_value, preference_loss, ranking_diagnostics, response_text,
)
from safe_rlhf_v.utils.processors import configure_llava_processor

WEIGHT_HASHES = [
    'c11dbf016ee7d35ee130c19b67e20eb04873996006f424b7bcbd453b6517ee66',
    '46df6c6e5fad297fe7fbca4963dcf180cb1d0528cabc884d1f603311fed01328',
    '4f06177c37ca13944e73b57dd6d051767b950dffc12400c700dfa2509c24c697',
]


def guard_gpu():
    gpu = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not gpu.isdigit():
        raise RuntimeError('Set CUDA_VISIBLE_DEVICES to exactly one explicit physical GPU')
    used = int(subprocess.check_output(['nvidia-smi', f'--id={gpu}', '--query-gpu=memory.used',
                                       '--format=csv,noheader,nounits'], text=True).strip())
    if used > 100:
        raise RuntimeError('Selected GPU is occupied; refusing to start')


def load_base(base):
    model, info = AccustomedLlavaRewardModel.from_pretrained(
        base, torch_dtype=torch.bfloat16, attn_implementation='sdpa',
        low_cpu_mem_usage=True, local_files_only=True, output_loading_info=True)
    # HF may load the prefix-remapped backbone without reporting the outside head.
    # Independently check the backbone inventory and tensor samples, then create a
    # fresh native Linear head explicitly (also avoids an uninitialized meta head).
    if set(info['missing_keys']) - {'score_head.weight'} or any(
            info.get(k) for k in ('unexpected_keys', 'mismatched_keys', 'error_msgs')):
        raise RuntimeError(f'Unexpected score initialization keys: {info}')
    index = json.loads((base / 'model.safetensors.index.json').read_text())['weight_map']
    backbone = model.model.state_dict()
    if set(backbone) != set(index) or any(t.is_meta for t in backbone.values()):
        raise RuntimeError('Score backbone does not match the complete base weight inventory')
    from safetensors import safe_open
    probes = ['language_model.model.embed_tokens.weight',
              'language_model.model.layers.0.self_attn.q_proj.weight',
              'multi_modal_projector.linear_1.weight',
              'vision_tower.vision_model.embeddings.patch_embedding.weight']
    for name in probes:
        with safe_open(base / index[name], framework='pt', device='cpu') as handle:
            expected = handle.get_tensor(name).to(backbone[name].dtype)
        if not torch.equal(expected, backbone[name].cpu()):
            raise RuntimeError(f'Base tensor mismatch: {name}')
    model.score_head = torch.nn.Linear(model.model.language_model.lm_head.in_features,
                                       1, bias=False, dtype=torch.float32)
    info['score_head_initialized_explicitly'] = True
    info['backbone_tensor_probes_verified'] = probes
    return model, {k: sorted(v) if isinstance(v, set) else v for k, v in info.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--kind', choices=['rm', 'cm'], required=True)
    p.add_argument('--run-name', required=True)
    p.add_argument('--dataset-name', default='rm-cm-smoke-128-32')
    p.add_argument('--epochs', type=int, default=2)
    p.add_argument('--batch-pairs', type=int, default=1)
    p.add_argument('--accumulation', type=int, default=4)
    p.add_argument('--lr', type=float, default=3e-5)
    p.add_argument('--rank', type=int, default=8)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    if min(args.epochs, args.batch_pairs, args.accumulation, args.rank) < 1 or not 0 < args.lr < 1:
        p.error('Invalid training hyperparameters')
    if transformers.__version__ != '4.48.3':
        raise RuntimeError('This smoke runner has been validated only against Transformers 4.48.3')
    root = args.root.resolve()
    run = root / 'runs' / args.run_name / args.kind
    if run.exists():
        raise RuntimeError('Run exists: do not overwrite or mix settings')
    data = root / 'datasets' / args.dataset_name
    if not (data / 'READY').exists():
        raise RuntimeError('Prepared dataset is not ready')
    manifest = json.loads((data / 'manifest.json').read_text())
    rows = {}
    for split in ('train', 'validation'):
        path = data / f'{split}.parquet'
        with path.open('rb') as f:
            if hashlib.file_digest(f, 'sha256').hexdigest() != manifest['file_sha256'][split]:
                raise RuntimeError('Prepared dataset checksum mismatch')
        rows[split] = pq.read_table(path).to_pylist()
    base = root / 'models/base'
    for i, digest in enumerate(WEIGHT_HASHES, 1):
        with (base / f'model-{i:05d}-of-00003.safetensors').open('rb') as f:
            if hashlib.file_digest(f, 'sha256').hexdigest() != digest:
                raise RuntimeError('Base weight checksum mismatch')
    guard_gpu()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Exactly one visible CUDA device is required')
    torch.cuda.set_device(0)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    run.mkdir(parents=True)
    started = time.perf_counter()
    model, loading = load_base(base)
    processor = AutoProcessor.from_pretrained(base, use_fast=False, local_files_only=True)
    configure_llava_processor(processor, model.config)
    processor.tokenizer.padding_side = 'right'
    processor.tokenizer.model_max_length = manifest['max_length']
    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05,
        target_modules=r'.*language_model.*\.(q_proj|v_proj)',
        modules_to_save=['score_head', 'multi_modal_projector']))
    # FP32 optimizer states/parameters for trainable modules; BF16 frozen backbone.
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.float()
    if any(p.requires_grad for n, p in model.named_parameters() if 'vision_tower' in n):
        raise RuntimeError('Vision tower must stay frozen')
    model.get_base_model().model.language_model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={'use_reentrant': False})
    model.to('cuda')
    trainable = [p for p in model.parameters() if p.requires_grad]
    trainable_count = sum(p.numel() for p in trainable)
    total_count = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    effective_pairs = args.batch_pairs * args.accumulation
    total_steps = args.epochs * math.ceil(len(rows['train']) / effective_pairs)
    warmup = max(1, math.ceil(total_steps * 0.03))
    def schedule(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1, (step - warmup) / max(1, total_steps - warmup))
        return 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)

    def scores_for(batch):
        images, texts = [], []
        for row in batch:
            image = image_from_value(row['image'])
            images.extend([image, image])
            texts.extend(response_text(processor, row['question'], row[f'response_{i}']) for i in (1, 2))
        inputs = processor(text=texts, images=images, padding=True, return_tensors='pt')
        if inputs.input_ids.shape[1] > manifest['max_length']:
            raise RuntimeError('Over-length training batch; refusing to truncate responses')
        inputs = {k: v.to('cuda') for k, v in inputs.items()}
        with torch.autocast('cuda', dtype=torch.bfloat16):
            return model(**inputs, use_cache=False).end_scores.reshape(len(batch), 2)

    def loss_for(scores, batch):
        field = 'more_helpful_response_id' if args.kind == 'rm' else 'safer_response_id'
        ids = [int(r[field]) for r in batch]
        ratings = torch.tensor([[r[f'response_{i}_harmless_rate'] for i in (1, 2)] for r in batch],
                               device='cuda')
        return preference_loss(scores, ids, args.kind, ratings)

    @torch.no_grad()
    def evaluate(split_rows):
        model.eval()
        losses, correct, all_scores = [], 0.0, []
        for offset in range(0, len(split_rows), args.batch_pairs):
            batch = split_rows[offset:offset + args.batch_pairs]
            scores = scores_for(batch)
            if not torch.isfinite(scores).all():
                raise RuntimeError('Non-finite evaluation scores')
            loss, accuracy = loss_for(scores, batch)
            losses.append(loss.item() * len(batch))
            correct += accuracy.item() * len(batch)
            all_scores.append(scores.cpu())
        values = torch.cat(all_scores)
        result = {'n': len(split_rows), 'loss': sum(losses) / len(split_rows),
                  'pairwise_accuracy': correct / len(split_rows),
                  'score_mean': values.mean().item(), 'score_std': values.std().item()}
        field = 'more_helpful_response_id' if args.kind == 'rm' else 'safer_response_id'
        result.update(ranking_diagnostics(values, [int(r[field]) for r in split_rows],
                                         args.kind, [r['category'] for r in split_rows]))
        if args.kind == 'cm':
            labels = [[r[f'is_response_{i}_safe'] for i in (1, 2)] for r in split_rows]
            if any(label not in ('yes', 'no') for pair in labels for label in pair):
                raise RuntimeError('Unknown safety label')
            unsafe = torch.tensor([[label == 'no' for label in pair] for pair in labels])
            result.update(binary_cost_diagnostics(values, unsafe))
        return result

    probe = rows['train'][:32]
    history = [{'epoch': 0, 'train_probe': evaluate(probe), 'validation': evaluate(rows['validation'])}]
    (run / 'history.json').write_text(json.dumps(history, indent=2))
    print(json.dumps(history[-1]), flush=True)
    step = 0
    train_seconds = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        shuffled = list(rows['train'])
        random.Random(args.seed + epoch).shuffle(shuffled)
        for offset in range(0, len(shuffled), effective_pairs):
            group = shuffled[offset:offset + effective_pairs]
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            tick = time.perf_counter()
            group_loss = 0.0
            for index in range(0, len(group), args.batch_pairs):
                batch = group[index:index + args.batch_pairs]
                scores = scores_for(batch)
                loss, _ = loss_for(scores, batch)
                if not torch.isfinite(loss):
                    raise RuntimeError('Non-finite training loss')
                (loss * len(batch) / len(group)).backward()
                group_loss += loss.item() * len(batch) / len(group)
            norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
            if not norm > 0:
                raise RuntimeError('No trainable gradient')
            optimizer.step()
            scheduler.step()
            torch.cuda.synchronize()
            train_seconds += time.perf_counter() - tick
            step += 1
            with (run / 'steps.jsonl').open('a') as f:
                f.write(json.dumps({'epoch': epoch, 'step': step, 'loss': group_loss,
                                    'grad_norm': float(norm), 'lr': scheduler.get_last_lr()[0]}) + '\n')
            if step == 1 or step % 8 == 0:
                print(json.dumps({'epoch': epoch, 'step': step, 'loss': group_loss,
                                  'train_seconds': train_seconds}), flush=True)
        history.append({'epoch': epoch, 'train_probe': evaluate(probe),
                        'validation': evaluate(rows['validation'])})
        (run / 'history.json').write_text(json.dumps(history, indent=2))
        print(json.dumps(history[-1]), flush=True)
    train_metrics = evaluate(rows['train'])
    model.eval()
    with torch.no_grad():
        expected = scores_for(rows['validation'][:2]).cpu()
    adapter = run / 'adapter'
    model.save_pretrained(adapter, safe_serialization=True)
    processor.save_pretrained(adapter)
    peak_allocated = torch.cuda.max_memory_allocated() / 2**30
    peak_reserved = torch.cuda.max_memory_reserved() / 2**30
    # Drop optimizer/model before loading a fresh base and saved adapter.
    del optimizer, scheduler, trainable, model, scores, loss
    gc.collect()
    torch.cuda.empty_cache()
    fresh, _ = load_base(base)
    model = PeftModel.from_pretrained(fresh, adapter, is_trainable=False).to('cuda')
    del fresh
    model.eval()
    with torch.no_grad():
        actual = scores_for(rows['validation'][:2]).cpu()
    difference = (expected - actual).abs().max().item()
    if not torch.allclose(expected, actual, atol=1e-3, rtol=1e-3):
        raise RuntimeError(f'Saved adapter reload mismatch: {difference}')
    summary = {'kind': args.kind, 'mode': 'lora_language_qv_plus_projector_and_score_head',
               'author_checkpoint': False, 'full_parameter_reproduction': False,
               'n_train': len(rows['train']), 'n_validation': len(rows['validation']),
               'categories': manifest['categories'], 'data_revision': manifest['revision'],
               'base_revision': 'b234b804b114d9e37bb655e11cbbb5f5e971b7a9',
               'epochs': args.epochs, 'optimizer_steps': step, 'lr': args.lr,
               'batch_pairs': args.batch_pairs, 'gradient_accumulation': args.accumulation,
               'lora_rank': args.rank, 'seed': args.seed,
               'trainable_parameters': trainable_count, 'total_parameters': total_count,
               'initial_validation': history[0]['validation'],
               'final_validation': history[-1]['validation'], 'final_train': train_metrics,
               'training_seconds': train_seconds, 'total_seconds': time.perf_counter() - started,
               'peak_allocated_gib': peak_allocated, 'peak_reserved_gib': peak_reserved,
               'reload_max_score_difference': difference, 'torch': torch.__version__,
               'transformers': transformers.__version__, 'peft': peft.__version__,
               'n_categories': len(manifest['categories']),
               'code_sha256': {path: hashlib.sha256((Path(__file__).resolve().parents[1] / path)
                                                    .read_bytes()).hexdigest()
                               for path in ['scripts/train_preference_smoke.py',
                                            'safe_rlhf_v/utils/preference_training.py',
                                            'safe_rlhf_v/models/llava.py']},
               'dataset_sha256': manifest['file_sha256'],
               'limitations': f'Small internal validation, {len(manifest["categories"])} categories, '
                              'one seed; LoRA, not full-parameter paper reproduction'}
    (run / 'summary.json').write_text(json.dumps(summary, indent=2))
    (run / 'loading.json').write_text(json.dumps(loading, indent=2))
    (run / 'args-private.json').write_text(json.dumps(vars(args), default=str, indent=2))
    (run / 'DONE').touch()
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
