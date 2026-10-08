#!/usr/bin/env python3
"""Matched LLaVA generation and response-likelihood preference evaluation.

Uses native Transformers inference, not the upstream training wrappers. No judge API.
"""
import argparse
import hashlib
import io
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer, LlavaForConditionalGeneration


def load_examples(root, per_category):
    examples = []
    for path in sorted((root / 'datasets/beavertails-v/data').glob('*/evaluation*.parquet')):
        dataset = load_dataset('parquet', data_files=str(path), split='train')
        category = path.parent.name
        for index, item in enumerate(dataset):
            if per_category and index >= per_category:
                break
            raw = item['image']
            if isinstance(raw, dict):
                raw = raw.get('bytes') or raw.get('path')
            if isinstance(raw, bytes):
                with Image.open(io.BytesIO(raw)) as image:
                    item['image'] = image.convert('RGB')
            elif isinstance(raw, (str, Path)):
                with Image.open(raw) as image:
                    item['image'] = image.convert('RGB')
            elif isinstance(raw, Image.Image):
                item['image'] = raw.convert('RGB')
            else:
                raise TypeError(f'Unsupported parquet image type: {type(raw).__name__}')
            examples.append({'id': f'{category}/{index}', **item, 'category': category})
    if not examples:
        raise RuntimeError('No evaluation parquet files found')
    return examples


def prompt_for(question, processor=None, style='manual'):
    if style == 'hf':
        if processor is None or not processor.chat_template:
            raise RuntimeError('HF prompt style requires the fixed base processor chat template')
        messages = [{'role': 'user', 'content': [{'type': 'image'},
                                               {'type': 'text', 'text': question}]}]
        return processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return f'USER: <image>\n{question}\nASSISTANT:'


def likelihoods(model, processor, batch, prompt_style='manual'):
    """Return per-response summed and mean conditional token log probabilities."""
    prompts = [prompt_for(item['question'], processor, prompt_style) for item in batch for _ in range(2)]
    images = [item['image'].convert('RGB') for item in batch for _ in range(2)]
    responses = [item[f'response_{r}'] for item in batch for r in (1, 2)]
    texts = [p + ' ' + r + processor.tokenizer.eos_token for p, r in zip(prompts, responses)]
    inputs = processor(text=texts, images=images, return_tensors='pt', padding=True)
    prompt_inputs = processor(text=prompts, images=images, return_tensors='pt', padding=True)
    if inputs.input_ids.shape[1] > 4096:
        raise RuntimeError('Response exceeds model context; do not silently truncate preference scoring')
    mask = torch.zeros_like(inputs.attention_mask, dtype=torch.bool)
    for index in range(len(texts)):
        full = inputs.input_ids[index][inputs.attention_mask[index].bool()].tolist()
        prefix = prompt_inputs.input_ids[index][prompt_inputs.attention_mask[index].bool()].tolist()
        shared = 0
        for a, b in zip(full, prefix):
            if a != b:
                break
            shared += 1
        # Tokenization can merge the last prompt token with the first response token.
        if shared < len(prefix) - 1:
            raise RuntimeError('Prompt/response token boundary mismatch')
        offset = inputs.input_ids.shape[1] - len(full)
        mask[index, offset + shared:] = True
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.inference_mode():
        logits = model(**inputs, use_cache=False).logits[:, :-1].float()
        targets = inputs['input_ids'][:, 1:]
        logp = logits.log_softmax(-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        shifted_mask = mask[:, 1:].to(model.device)
        sums = (logp * shifted_mask).sum(-1)
        counts = shifted_mask.sum(-1)
        if (counts == 0).any():
            raise RuntimeError('No response tokens to score')
        means = sums / counts
    return [{'sum_logp': s, 'mean_logp': m, 'tokens': n}
            for s, m, n in zip(sums.cpu().tolist(), means.cpu().tolist(), counts.cpu().tolist())]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--run', required=True)
    parser.add_argument('--model', choices=['base', 'safe'], required=True)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--per-category', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--max-new-tokens', type=int, default=256)
    parser.add_argument('--prompt-style', choices=['manual', 'hf'], default='manual')
    parser.add_argument('--skip-likelihood', action='store_true')
    args = parser.parse_args()
    if not 0 <= args.shard < args.num_shards:
        parser.error('Invalid shard')
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    root = args.root.resolve()
    run_dir = root / 'runs' / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    output = run_dir / f'{args.model}-{args.shard}.jsonl'
    metadata_file = run_dir / f'{args.model}-{args.shard}.metadata.json'
    model_path = root / ('models/base' if args.model == 'base' else 'models/safe/LLaVA_Safe_RLHF-V')
    processor = AutoProcessor.from_pretrained(root / 'models/base', local_files_only=True)
    # CLIP has a CLS token, removed by LLaVA's default vision feature selection.
    processor.patch_size = 14
    processor.num_additional_image_tokens = 1
    processor.vision_feature_select_strategy = 'default'
    processor.tokenizer.padding_side = 'left'
    other_tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if other_tokenizer.get_vocab() != processor.tokenizer.get_vocab():
        raise RuntimeError('Tokenizer vocab differs; shared processor is not valid')
    model, loading_info = LlavaForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        attn_implementation='sdpa', local_files_only=True, output_loading_info=True,
        use_safetensors=args.model == 'base', weights_only=True)
    if any(loading_info.get(k) for k in ['missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs']):
        raise RuntimeError(f'Checkpoint did not load exactly: {loading_info}')
    model = model.to('cuda').eval()
    selected = load_examples(root, args.per_category)[args.shard::args.num_shards]
    metadata = {'args': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                'assets': json.loads((root / 'assets.json').read_text()),
                'torch': torch.__version__, 'transformers': __import__('transformers').__version__,
                'gpu': torch.cuda.get_device_name(0), 'processor': {'source': 'base', 'patch_size': 14,
                'num_additional_image_tokens': 1, 'vision_feature_select_strategy': 'default'},
                'decoding': {'do_sample': False, 'repetition_penalty': 1.0,
                             'max_new_tokens': args.max_new_tokens, 'seed': 42},
                'examples': len(selected), 'loading_info': loading_info}
    if metadata_file.exists() and json.loads(metadata_file.read_text()) != metadata:
        raise RuntimeError('Cannot resume with different settings')
    metadata_file.write_text(json.dumps(metadata, indent=2))
    finished = set()
    if output.exists():
        for line in output.read_text().splitlines():
            finished.add(json.loads(line)['id'])
    selected = [item for item in selected if item['id'] not in finished]
    print(f'Loaded {args.model}; remaining {len(selected)} examples', flush=True)
    with output.open('a', buffering=1) as handle:
        for start in range(0, len(selected), args.batch_size):
            batch = selected[start:start + args.batch_size]
            images = [item['image'].convert('RGB') for item in batch]
            inputs = processor(text=[prompt_for(item['question'], processor, args.prompt_style) for item in batch],
                               images=images, return_tensors='pt', padding=True)
            if (inputs.input_ids == model.config.image_token_index).sum(1).tolist() != [576] * len(batch):
                raise RuntimeError('Expected exactly 576 image tokens per example')
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            torch.cuda.synchronize()
            begin = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                           do_sample=False, repetition_penalty=1.0, use_cache=True,
                                           pad_token_id=processor.tokenizer.pad_token_id)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - begin
            new_tokens = generated[:, inputs['input_ids'].shape[1]:]
            texts = processor.batch_decode(new_tokens, skip_special_tokens=True)
            scores = None if args.skip_likelihood else likelihoods(model, processor, batch, args.prompt_style)
            for index, (item, image, text, tokens) in enumerate(zip(batch, images, texts, new_tokens)):
                eos = processor.tokenizer.eos_token_id
                ids = tokens.tolist()
                stop = ids.index(eos) if eos in ids else len(ids)
                record = {'id': item['id'], 'category': item['category'], 'question': item['question'],
                          'image_sha256': hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest(),
                          'model': args.model, 'response': text.strip(), 'generated_tokens': stop,
                          'hit_token_limit': stop == args.max_new_tokens,
                          'batch_generation_seconds': elapsed, 'batch_size': len(batch),
                          'helpful_id': item['more_helpful_response_id'], 'safer_id': item['safer_response_id'],
                          'preference_likelihood': scores[2 * index:2 * index + 2] if scores else None}
                handle.write(json.dumps(record, ensure_ascii=False) + '\n')
            print(f'{args.model}/{args.shard}: {start + len(batch)}/{len(selected)} '
                  f'generation {elapsed:.2f}s, peak {torch.cuda.max_memory_allocated()/2**30:.2f} GiB', flush=True)
    (run_dir / f'{args.model}-{args.shard}.done').touch()


if __name__ == '__main__':
    main()
