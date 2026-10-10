#!/usr/bin/env python3
"""Full-language-backbone native LLaVA RM/CM training, DeepSpeed ZeRO-2.

Uses published optimizer settings and native score architecture/loss. Dataset
size/provenance are explicit; this is not the author's recovered experiment.
No LoRA, RL or judge. Launch with explicit idle physical GPUs.
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
import deepspeed
from deepspeed.ops.adam import FusedAdam
import numpy as np
import pyarrow.parquet as pq
import torch
import torch.distributed as dist
import transformers
from transformers import AutoProcessor, CLIPVisionConfig, LlamaConfig, LlavaConfig, get_scheduler
from safe_rlhf_v.models.llava import AccustomedLlavaRewardModel
from safe_rlhf_v.utils.preference_training import (
    binary_cost_diagnostics, image_from_value, preference_loss, preference_score_diagnostics,
    ranking_diagnostics, response_text,
)
from safe_rlhf_v.utils.preference_distributed import (
    configure_full_score_training, score_deepspeed_config, restore_llama_rope_fp32,
)
from safe_rlhf_v.utils.processors import configure_llava_processor
from train_preference_smoke import load_base, WEIGHT_HASHES


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--run-name', required=True)
    p.add_argument('--dataset-name', required=True)
    p.add_argument('--kind', choices=('rm', 'cm'), required=True)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--batch-pairs', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--tiny', action='store_true', help='Random small architecture engine test, not 7B training')
    p.add_argument('--limit-train', type=int)
    p.add_argument('--limit-validation', type=int)
    p.add_argument('--checkpoint-interval', type=int, default=0,
                   help='Save additional optimizer snapshots every N updates; 0 disables')
    p.add_argument('--epoch-checkpoints', action='store_true')
    p.add_argument('--native-loss-precision', action='store_true',
                   help='Use upstream integer-sign CM arithmetic; historical default used FP32 absolute terms')
    args = p.parse_args()
    rank, local, world = (int(os.environ[k]) for k in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE'))
    devices = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    if len(devices) != world or not all(s.isdigit() for s in devices) or len(set(devices)) != world:
        p.error('CUDA_VISIBLE_DEVICES must explicitly list distinct physical GPUs, one per rank')
    if min(args.epochs, args.batch_pairs) < 1 or args.checkpoint_interval < 0 or any(
            v is not None and v < world for v in (args.limit_train, args.limit_validation)):
        p.error('Invalid epochs/batch/sample limits')
    if transformers.__version__ != '4.48.3' or deepspeed.__version__ != '0.16.2':
        raise RuntimeError('Preflight is pinned to Transformers 4.48.3 / DeepSpeed 0.16.2')
    root = args.root.resolve()
    run = root / 'runs' / args.run_name / args.kind
    base, data = root / 'models/base', root / 'datasets' / args.dataset_name
    launcher_id = os.environ['TORCHELASTIC_RUN_ID']
    if rank == 0:
        if run.exists():
            raise RuntimeError('Run already exists; never overwrite/mix settings')
        used = subprocess.check_output(['nvidia-smi', '--id=' + ','.join(devices),
                                        '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True)
        if any(int(v) > 100 for v in used.split()):
            raise RuntimeError('Selected GPU occupied; refusing to start')
        if not args.tiny:
            for index, digest in enumerate(WEIGHT_HASHES, 1):
                with (base / f'model-{index:05d}-of-00003.safetensors').open('rb') as handle:
                    if hashlib.file_digest(handle, 'sha256').hexdigest() != digest:
                        raise RuntimeError('Base checksum mismatch')
        run.mkdir(parents=True)
        temporary_marker = run / 'preflight-ready.tmp'
        temporary_marker.write_text(launcher_id)
        temporary_marker.replace(run / 'preflight-ready')
    else:
        deadline = time.monotonic() + 240
        while True:
            marker = run / 'preflight-ready'
            if marker.exists() and marker.read_text() == launcher_id:
                break
            if marker.exists() and marker.read_text():
                raise RuntimeError('Existing run belongs to a different launcher; refusing reuse')
            if time.monotonic() > deadline:
                raise RuntimeError('Rank zero preflight did not finish')
            time.sleep(0.25)
    if not (data / 'READY').exists():
        raise RuntimeError('Dataset not ready')
    manifest = json.loads((data / 'manifest.json').read_text())
    rows = {}
    for split, limit in [('train', args.limit_train), ('validation', args.limit_validation)]:
        path = data / f'{split}.parquet'
        with path.open('rb') as handle:
            if hashlib.file_digest(handle, 'sha256').hexdigest() != manifest['file_sha256'][split]:
                raise RuntimeError('Dataset checksum mismatch')
        rows[split] = pq.read_table(path).to_pylist()[:limit]
    if len(rows['train']) % (world * args.batch_pairs) or not rows['validation']:
        raise RuntimeError('Training must be evenly divisible; validation is exact, unpadded')
    torch.cuda.set_device(local)
    score_device = torch.device('cuda', local)
    deepspeed.init_distributed(dist_backend='nccl')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    started = time.perf_counter()
    if args.tiny:
        config = LlavaConfig(
            vision_config=CLIPVisionConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                                           num_attention_heads=4, image_size=336, patch_size=14),
            text_config=LlamaConfig(vocab_size=32064, hidden_size=32, intermediate_size=64,
                                    num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4,
                                    pad_token_id=32001, bos_token_id=1, eos_token_id=2),
            image_token_index=32000, image_seq_length=576, vision_feature_layer=-1)
        model, loading = AccustomedLlavaRewardModel(config), {'random_tiny_engine_test': True}
    else:
        model, loading = load_base(base)
    processor = AutoProcessor.from_pretrained(base, use_fast=False, local_files_only=True)
    configure_llava_processor(processor, model.config)
    processor.tokenizer.padding_side = 'right'
    processor.tokenizer.model_max_length = manifest['max_length']
    trainable = configure_full_score_training(model)
    trainable_count = sum(v.numel() for v in trainable)
    total_count = sum(v.numel() for v in model.parameters())
    optimizer = FusedAdam(trainable, lr=3e-5, betas=(0.9, 0.95), weight_decay=0.0)
    total_steps = args.epochs * len(rows['train']) // (world * args.batch_pairs)
    scheduler_name = 'constant_with_warmup' if args.kind == 'rm' else 'cosine'
    scheduler = get_scheduler(scheduler_name, optimizer=optimizer,
                              num_warmup_steps=int(total_steps * 0.03), num_training_steps=total_steps)
    ds_config = score_deepspeed_config(world, args.batch_pairs)
    engine, optimizer, _, scheduler = deepspeed.initialize(model=model, optimizer=optimizer,
                                                          lr_scheduler=scheduler, config=ds_config)
    rotary_precision = restore_llama_rope_fp32(engine.module)
    del model, trainable
    if rank == 0:
        (run / 'loading.json').write_text(json.dumps(loading, indent=2))
        (run / 'deepspeed.json').write_text(json.dumps(ds_config, indent=2))
        (run / 'args-private.json').write_text(json.dumps(vars(args), default=str, indent=2))
        print(json.dumps({'event': 'engine_ready', 'tiny': args.tiny,
                          'trainable_parameters': trainable_count, 'total_parameters': total_count}), flush=True)

    def inputs_for(batch):
        images, texts = [], []
        for row in batch:
            image = image_from_value(row['image'])
            images += [image, image]
            texts.extend(response_text(processor, row['question'], row[f'response_{i}']) for i in (1, 2))
        inputs = processor(text=texts, images=images, padding=True, return_tensors='pt')
        if inputs.input_ids.shape[1] > manifest['max_length']:
            raise RuntimeError('Over-length batch; no truncation permitted')
        return {k: v.to(score_device, dtype=torch.bfloat16 if v.is_floating_point() else v.dtype)
                for k, v in inputs.items()}

    def scores_for(batch, scoring_model=None):
        inputs = inputs_for(batch)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            return (scoring_model or engine)(**inputs, use_cache=False,
                                             num_logits_to_keep=1).end_scores.reshape(len(batch), 2)

    def loss_for(scores, batch):
        field = 'more_helpful_response_id' if args.kind == 'rm' else 'safer_response_id'
        ratings = scores.new_tensor([[r[f'response_{i}_harmless_rate'] for i in (1, 2)] for r in batch])
        return preference_loss(scores, [int(r[field]) for r in batch], args.kind, ratings,
                               native_precision=args.native_loss_precision)[0]

    def score_diagnostics(scores, batch):
        field = 'more_helpful_response_id' if args.kind == 'rm' else 'safer_response_id'
        ratings = scores.new_tensor([[r[f'response_{i}_harmless_rate'] for i in (1, 2)] for r in batch])
        return preference_score_diagnostics(scores, [int(r[field]) for r in batch], args.kind, ratings,
                                            native_precision=args.native_loss_precision)

    @torch.no_grad()
    def evaluate(all_rows, return_scores=False):
        engine.eval()
        local_results = []
        indices = list(range(rank, len(all_rows), world))
        for offset in range(0, len(indices), args.batch_pairs):
            chosen = indices[offset:offset + args.batch_pairs]
            batch = [all_rows[i] for i in chosen]
            scores = scores_for(batch)
            loss = loss_for(scores, batch)
            if not torch.isfinite(scores).all() or not torch.isfinite(loss):
                raise RuntimeError('Non-finite evaluation')
            local_results.append((chosen, scores.cpu().tolist(), loss.item() * len(batch)))
        gathered = [None] * world
        dist.all_gather_object(gathered, local_results)
        values = torch.empty(len(all_rows), 2)
        loss_sum, seen = 0.0, set()
        for group in gathered:
            for chosen, scores, value in group:
                if seen.intersection(chosen):
                    raise RuntimeError('Duplicate distributed evaluation indices')
                seen.update(chosen)
                values[chosen] = torch.tensor(scores)
                loss_sum += value
        if len(seen) != len(all_rows):
            raise RuntimeError('Missing distributed evaluation indices')
        field = 'more_helpful_response_id' if args.kind == 'rm' else 'safer_response_id'
        result = {'n': len(all_rows), 'loss': loss_sum / len(all_rows)}
        result.update(ranking_diagnostics(values, [int(r[field]) for r in all_rows], args.kind,
                                         [r['category'] for r in all_rows]))
        result['pairwise_accuracy'] = result['pairwise_correct'] / len(all_rows)
        if args.kind == 'cm':
            unsafe = torch.tensor([[r[f'is_response_{i}_safe'] == 'no' for i in (1, 2)] for r in all_rows])
            result.update(binary_cost_diagnostics(values, unsafe))
        result['fp32_score_loss_diagnostics'] = {
            name: value.item() for name, value in score_diagnostics(values, all_rows).items()}
        return (result, values) if return_scores else result

    history = [{'epoch': 0, 'validation': evaluate(rows['validation'])}]
    def save_history():
        if rank == 0:
            (run / 'history.json').write_text(json.dumps(history, indent=2))
            print(json.dumps(history[-1]), flush=True)
    save_history()
    step, training_seconds = 0, 0.0
    checkpoint_tags = set()

    def save_checkpoint(epoch, next_batch_offset):
        tag = f'step-{step}'
        if tag in checkpoint_tags:
            return
        rng = {'python': random.getstate(), 'numpy': np.random.get_state(),
               'torch_cpu': torch.get_rng_state(), 'torch_cuda': torch.cuda.get_rng_state()}
        rng_states = [None] * world
        dist.all_gather_object(rng_states, rng)
        engine.save_checkpoint(str(run / 'checkpoint'), tag=tag, client_state={
            'optimizer_steps': step, 'seed': args.seed, 'epoch': epoch,
            'next_local_batch_offset': next_batch_offset, 'world_size': world,
            'batch_pairs_per_gpu': args.batch_pairs, 'planned_optimizer_steps': total_steps,
            'dataset_sha256': manifest['file_sha256'], 'kind': args.kind,
            'native_loss_precision': args.native_loss_precision,
            'rng_states_by_rank': rng_states, 'history': history,
        })
        checkpoint_tags.add(tag)
        if rank == 0:
            print(json.dumps({'event': 'checkpoint_saved', 'tag': tag,
                              'epoch': epoch, 'next_local_batch_offset': next_batch_offset}), flush=True)

    for epoch in range(1, args.epochs + 1):
        engine.train()
        order = list(range(len(rows['train'])))
        random.Random(args.seed + epoch).shuffle(order)
        order = order[rank::world]
        for offset in range(0, len(order), args.batch_pairs):
            torch.cuda.synchronize()
            tick = time.perf_counter()
            batch = [rows['train'][i] for i in order[offset:offset + args.batch_pairs]]
            scores = scores_for(batch)
            loss = loss_for(scores, batch)
            diagnostics = score_diagnostics(scores, batch)
            diagnostic_names = sorted(diagnostics)
            diagnostic_values = torch.stack([diagnostics[name].float() for name in diagnostic_names])
            dist.all_reduce(diagnostic_values)
            diagnostic_values /= world
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite training loss')
            engine.backward(loss)
            engine.step()
            norm = engine.get_global_grad_norm()
            if norm is None or not math.isfinite(float(norm)) or float(norm) <= 0:
                raise RuntimeError('Missing/non-finite/zero global gradient norm')
            if not torch.isfinite(engine.module.score_head.weight).all():
                raise RuntimeError('Non-finite updated score head')
            average_loss = loss.detach().clone()
            dist.all_reduce(average_loss)
            torch.cuda.synchronize()
            training_seconds += time.perf_counter() - tick
            step += 1
            if rank == 0:
                event = {'epoch': epoch, 'step': step, 'loss': average_loss.item() / world,
                         'grad_norm': float(norm), 'lr': scheduler.get_last_lr()[0],
                         'training_seconds': training_seconds,
                         'diagnostic_mean_over_ranks': dict(zip(diagnostic_names, diagnostic_values.tolist()))}
                with (run / 'steps.jsonl').open('a') as handle:
                    handle.write(json.dumps(event) + '\n')
                print(json.dumps(event), flush=True)
            if args.checkpoint_interval and step % args.checkpoint_interval == 0:
                save_checkpoint(epoch, offset + args.batch_pairs)
        validation, expected_validation_scores = evaluate(rows['validation'], return_scores=True)
        history.append({'epoch': epoch, 'validation': validation})
        save_history()
        if args.epoch_checkpoints:
            save_checkpoint(epoch + 1, 0)
    final_train = evaluate(rows['train'])
    engine.eval()
    reload_rows = rows['validation'][:1]
    with torch.no_grad():
        expected = scores_for(reload_rows).cpu()
    # ZeRO-2 has replicated complete BF16 parameters; save full safetensors on rank zero.
    model_dir = run / 'model'
    if rank == 0:
        engine.module.save_pretrained(model_dir, safe_serialization=True, max_shard_size='5GB')
        processor.save_pretrained(model_dir)
    dist.barrier()
    save_checkpoint(args.epochs + 1, 0)
    # Test real optimizer/scheduler checkpoint restore, not only adapter weights.
    # Probe and deliberately perturb local optimizer moments before restoration.
    # A successful load with weights alone must not pass this contract.
    def optimizer_probes():
        probes = []
        for state in engine.optimizer.optimizer.state.values():
            if 'exp_avg' in state:
                probes.append({key: state[key].flatten()[:64].detach().cpu().clone()
                               for key in ('exp_avg', 'exp_avg_sq')})
        if not probes:
            raise RuntimeError('Optimizer has no trained moment states')
        return probes
    expected_optimizer = optimizer_probes()
    with torch.no_grad():
        engine.module.score_head.weight.add_(1)
        for state in engine.optimizer.optimizer.state.values():
            if 'exp_avg' in state:
                state['exp_avg'].flatten()[:64].add_(1)
    checkpoint, client = engine.load_checkpoint(str(run / 'checkpoint'), tag=f'step-{step}',
                                                load_optimizer_states=True, load_lr_scheduler_states=True)
    if not checkpoint or client.get('optimizer_steps') != step or engine.global_steps != step:
        raise RuntimeError('Checkpoint/client/step restoration failed')
    actual_optimizer = optimizer_probes()
    if len(actual_optimizer) != len(expected_optimizer) or any(
            not torch.equal(before[key], after[key])
            for before, after in zip(expected_optimizer, actual_optimizer)
            for key in ('exp_avg', 'exp_avg_sq')):
        raise RuntimeError('Optimizer moment restoration mismatch')
    with torch.no_grad():
        restored = scores_for(reload_rows).cpu()
    checkpoint_difference = (expected - restored).abs().max().item()
    if not torch.equal(expected, restored):
        raise RuntimeError('Checkpoint restore score mismatch')
    # Exercise a post-restore optimizer step without overwriting final artifacts.
    # A terminal cosine LR may be zero: report it, not a claim of parameter change.
    engine.train()
    post_restore_lr = scheduler.get_last_lr()[0]
    resume_batch = rows['train'][rank * args.batch_pairs:(rank + 1) * args.batch_pairs]
    resume_loss = loss_for(scores_for(resume_batch), resume_batch)
    if not torch.isfinite(resume_loss):
        raise RuntimeError('Non-finite resumed training loss')
    engine.backward(resume_loss)
    engine.step()
    if engine.global_steps != step + 1 or not math.isfinite(float(engine.get_global_grad_norm())):
        raise RuntimeError('Resumed optimizer update failed')
    # Release engine/optimizer and autograd graph references BEFORE independent
    # loading. Full-context verification must not require two 7B models plus Adam.
    del engine, optimizer, scheduler, scores, loss, resume_loss
    gc.collect()
    torch.cuda.empty_cache()
    dist.barrier()
    # Fresh independent score architecture reload on rank zero, while other ranks wait.
    fresh_difference, fresh_info, validation_reload_difference = None, None, None
    if rank == 0:
        fresh, fresh_info = AccustomedLlavaRewardModel.from_pretrained(
            model_dir, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
            local_files_only=True, attn_implementation='sdpa', output_loading_info=True)
        if any(fresh_info[k] for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
            raise RuntimeError(f'Fresh checkpoint loading keys: {fresh_info}')
        if any(v.is_meta for v in fresh.parameters()):
            raise RuntimeError('Fresh checkpoint contains meta parameters')
        fresh.to(score_device).eval()
        with torch.no_grad():
            actual = scores_for(reload_rows, fresh).cpu()
        fresh_difference = (expected - actual).abs().max().item()
        if not torch.allclose(expected, actual, atol=1e-3, rtol=1e-3):
            raise RuntimeError('Fresh full-model reload score mismatch')
        # Same batch composition as the distributed evaluator, not different padding.
        actual_validation_scores = torch.empty_like(expected_validation_scores)
        with torch.no_grad():
            for partition in range(world):
                indices = list(range(partition, len(rows['validation']), world))
                for offset in range(0, len(indices), args.batch_pairs):
                    chosen = indices[offset:offset + args.batch_pairs]
                    batch = [rows['validation'][i] for i in chosen]
                    actual_validation_scores[chosen] = scores_for(batch, fresh).cpu()
        validation_reload_difference = (expected_validation_scores - actual_validation_scores).abs().max().item()
        if not torch.allclose(expected_validation_scores, actual_validation_scores, atol=1e-3, rtol=1e-3):
            raise RuntimeError('Fresh full-validation reload score mismatch')
        if not torch.equal(expected_validation_scores[:, 0] > expected_validation_scores[:, 1],
                           actual_validation_scores[:, 0] > actual_validation_scores[:, 1]):
            raise RuntimeError('Fresh full-validation reload ranking mismatch')
        if args.kind == 'cm' and not torch.equal(expected_validation_scores > 0, actual_validation_scores > 0):
            raise RuntimeError('Fresh full-validation reload safety threshold mismatch')
        del fresh
        gc.collect()
        torch.cuda.empty_cache()
    dist.barrier()
    torch.cuda.synchronize()
    metrics = [None] * world
    dist.all_gather_object(metrics, {'rank': rank, 'allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
                                    'reserved_gib': torch.cuda.max_memory_reserved() / 2**30,
                                    'training_seconds': training_seconds})
    if rank == 0:
        summary = {'kind': args.kind, 'mode': 'random_tiny_engine_test' if args.tiny else 'full_language_backbone_projector_score',
                   'author_checkpoint': False, 'paper_reproduction_complete': False,
                   'n_train': len(rows['train']), 'n_validation': len(rows['validation']),
                   'epochs': args.epochs, 'optimizer_steps': step, 'world_size': world,
                   'batch_pairs_per_gpu': args.batch_pairs, 'global_batch_pairs': world * args.batch_pairs,
                   'lr': 3e-5, 'weight_decay': 0.0, 'betas': [0.9, 0.95], 'seed': args.seed,
                   'scheduler': scheduler_name, 'warmup_steps': int(total_steps * 0.03),
                   'gradient_checkpointing': 'non_reentrant', 'zero_stage': 2,
                   'trainable_parameters': trainable_count, 'total_parameters': total_count,
                   'initial_validation': history[0]['validation'], 'final_validation': history[-1]['validation'],
                   'final_train': final_train, 'per_rank_resources': metrics,
                   'total_seconds': time.perf_counter() - started,
                   'checkpoint_restore_score_difference': checkpoint_difference,
                   'fresh_model_reload_score_difference': fresh_difference,
                   'fresh_reload_validation_max_score_difference': validation_reload_difference,
                   'fresh_reload_validation_pairs': len(rows['validation']),
                   'fresh_reload_validation_ranking_equal': True,
                   'rotary_buffer_precision': rotary_precision,
                   'score_loss_diagnostics': 'detached selected-precision training loss parts and score gradients; '
                                             'per-step mean of rank-local statistics; eval recomputed FP32',
                   'native_loss_precision': args.native_loss_precision,
                   'auc_algorithm': 'Mann–Whitney searchsorted, tie weight 0.5',
                   'optimizer_moment_restore_verified': True, 'post_restore_update_verified': True,
                   'resume_check_updates_not_saved': 1, 'post_restore_learning_rate': post_restore_lr,
                   'checkpoint_tags': sorted(checkpoint_tags, key=lambda tag: int(tag.split('-')[1])),
                   'checkpoint_rng_saved': True, 'full_cursor_rng_resume_verified': False,
                   'max_length': manifest['max_length'],
                   'validation_source_split': manifest.get('validation_source_split', 'train'),
                   'dataset_sha256': manifest['file_sha256'], 'data_revision': manifest['revision'],
                   'base_revision': None if args.tiny else 'b234b804b114d9e37bb655e11cbbb5f5e971b7a9',
                   'processor_revision': 'b234b804b114d9e37bb655e11cbbb5f5e971b7a9',
                   'torch': torch.__version__, 'transformers': transformers.__version__, 'deepspeed': deepspeed.__version__,
                   'code_sha256': {path: hashlib.sha256((Path(__file__).resolve().parents[1] / path).read_bytes()).hexdigest()
                                   for path in ['scripts/train_preference_full.py', 'safe_rlhf_v/utils/preference_distributed.py',
                                                'safe_rlhf_v/utils/preference_training.py', 'safe_rlhf_v/models/llava.py']},
                   'limitations': 'One seed; not recovered author run or final-policy evaluation; '
                                  'dataset split/filters are defined by its manifest, no best-epoch selection; '
                                  'ZeRO-2, frozen unused vocabulary head; logits_to_keep=1 leaves hidden-state scoring unchanged; '
                                  'resource peaks include independent reload; cursor/RNG snapshots do not certify full resume'}
        (run / 'summary.json').write_text(json.dumps(summary, indent=2))
        (run / 'DONE').touch()
        print(json.dumps(summary, indent=2), flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
