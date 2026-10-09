#!/usr/bin/env python3
"""Export only validated score weights/processor and sanitized aggregate metadata.

No samples, pickle optimizer states, private arguments, cache or credentials.
Uploads are opt-in; private by default. Public fallback requires an explicit
flag and a confirmed private-storage quota error (not an auth/network failure).
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys

BASE = 'llava-hf/llava-1.5-7b-hf'
BASE_REVISION = 'b234b804b114d9e37bb655e11cbbb5f5e971b7a9'
PROCESSOR_FILES = {
    'added_tokens.json', 'chat_template.json', 'preprocessor_config.json', 'processor_config.json',
    'special_tokens_map.json', 'tokenizer_config.json', 'tokenizer.model', 'tokenizer.json',
}
PRIVATE = re.compile(r'/data(?:\d+)?/|/home/|file://|172\.16\.|hf_[A-Za-z0-9]{20,}')


def sanitize_config(value):
    if isinstance(value, dict):
        return {k: (BASE if k in {'_name_or_path', 'name_or_path', 'base_model_name_or_path'}
                    and isinstance(v, str) and (v.startswith('/') or v.startswith('file:'))
                    else sanitize_config(v)) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize_config(v) for v in value]
    return value


def private_quota_error(error):
    status = getattr(getattr(error, 'response', None), 'status_code', None)
    message = str(error).lower()
    return status in {400, 402, 403, 413, 507} and 'private' in message and (
        'storage' in message or 'quota' in message) and (
        'limit' in message or 'exceed' in message or 'quota' in message)


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def model_card(kind, mode, summary, repo_id, training_commit):
    metric = summary['final_validation']
    code = 'https://github.com/wangzy4/saferlhf-v/tree/reproduction/fix-and-inference'
    loading = (f'''from pathlib import Path
from huggingface_hub import snapshot_download
from peft import PeftModel
from scripts.train_preference_smoke import load_base
base_path = Path(snapshot_download("{BASE}", revision="{BASE_REVISION}"))
base, _ = load_base(base_path)  # strict backbone inventory/probes; explicit FP32 head
model = PeftModel.from_pretrained(base, REPO, revision=REVISION).eval().to("cuda")
''' if mode == 'lora' else '''from safe_rlhf_v.models.llava import AccustomedLlavaRewardModel
model = AccustomedLlavaRewardModel.from_pretrained(
    REPO, revision=REVISION, torch_dtype=torch.bfloat16,
    attn_implementation="sdpa", low_cpu_mem_usage=True,
    trust_remote_code=False).eval().to("cuda")
''')
    safety = (f"Safety-label accuracy at cost > 0: {metric['safety_label_accuracy_at_zero']:.6f}; "
              f"balanced accuracy: {metric['balanced_safety_label_accuracy_at_zero']:.6f}; "
              f"majority baseline: {metric['majority_class_baseline']:.6f}; "
              f"response AUC: {metric['response_safety_auc']:.6f}.\n" if kind == 'cm' else '')
    if kind == 'cm' and metric['balanced_safety_label_accuracy_at_zero'] <= 0.5:
        safety += ('\n**QUALITY FAILED: zero-threshold balanced accuracy is at chance. '
                   'Do not use this checkpoint for RL, safety gating or deployment. '
                   'It is archived for research diagnostics only.**\n')
    return f'''---
license: other
license_name: llama2-with-training-data-terms
license_link: https://huggingface.co/{repo_id}/blob/main/README.md#license-and-use
base_model: {BASE}
language:
- en
tags:
- llava
- {'reward-model' if kind == 'rm' else 'cost-model'}
- safety
- research
{'- lora' if mode == 'lora' else '- full-backbone'}
datasets:
- saferlhf-v/BeaverTails-V
---

# LLaVA-1.5-7B {'reward' if kind == 'rm' else 'cost'} score model ({mode})

Self-trained research checkpoint, **not the author's checkpoint, a chat policy,
or a completed paper reproduction**. {'Higher reward means more helpful.' if kind == 'rm' else 'Higher cost means more harmful; lower cost means safer.'}

## Training and validation

- Fixed base revision: `{BASE_REVISION}`.
- Source data revision: `ee19041205c720c0faea575de563de8a6a8f9094`.
- {summary['n_train']} training pairs, {summary['n_validation']} internal validation pairs,
  {summary['epochs']} epochs, seed {summary['seed']}; validation comes from source **train split**.
- Training source commit: `{training_commit}`; exact source/data hashes in `results.json`.
- {'Language q/v LoRA rank 8 plus full projector/head, frozen vision; not full finetuning.' if mode == 'lora' else 'Full language backbone, projector and score head; frozen vision and unused vocabulary head. BF16, ZeRO-2.'}
- Strict validation preference accuracy: {metric['pairwise_correct']}/{summary['n_validation']}
  ({metric['pairwise_accuracy']:.6f}); ties count as incorrect.
{safety}
Exact RGB-image overlap with previously used evaluation and internal train/validation
was excluded. This is **not semantic/near-duplicate deduplication**. Small internal,
single-seed results do not establish generalization, deployment safety, or policy
helpfulness/safety win rates. Do not compare this accuracy directly with paper tables
or a different internal validation set. No RL or judge evaluation is included.

## Load and score (no remote model code)

Use the fork's native score wrapper, not `AutoModelForCausalLM` or a chat pipeline.
Clone [the source repository]({code}); use its documented pinned environment
(torch 2.5.1, Transformers 4.48.3, PEFT 0.14.0 for adapters). Run from the clone.
`REVISION` below must be the immutable HF commit returned by upload/model_info.

```python
import torch
from PIL import Image
from transformers import AutoProcessor
from safe_rlhf_v.utils.processors import configure_llava_processor
from safe_rlhf_v.utils.preference_training import response_text
REPO = "{repo_id}"
REVISION = "<immutable HF commit>"
{loading}
processor = AutoProcessor.from_pretrained(REPO, revision=REVISION, use_fast=False)
configure_llava_processor(processor, model.config)
processor.tokenizer.padding_side = "right"
text = response_text(processor, "Describe this picture.", "This picture shows a landscape.")
image = Image.open("example.jpg").convert("RGB")
inputs = processor(text=[text], images=[image], return_tensors="pt")
inputs = {{k: v.to("cuda") for k, v in inputs.items()}}
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    score = model(**inputs, use_cache=False, num_logits_to_keep=1).end_scores
print(score)
```

Move adapters by device only: do not blanket-cast their FP32 trained modules or
nonpersistent RoPE buffers with `.bfloat16()` / `.to(dtype=...)`. Processor expands
576 image tokens; do not silently truncate responses. Use matching native architecture.

## Artifacts and scope

Savetensors {'adapter, projector and head (requires pinned base)' if mode == 'lora' else 'complete model weights'},
processor/tokenizer, aggregate results and SHA256 file manifest. No raw data,
per-sample risk texts, machine paths, secrets or pickle optimizer checkpoints are
uploaded. DeepSpeed optimizer checkpoints remain on the training data disk; this
repository is an inference-weight archive, not a complete restart/RNG backup.

## License and use

The LLaVA base derives from Llama 2: see the included `LICENSE.Llama2.txt`,
`LLAMA2_USE_POLICY.md` and `NOTICE`. BeaverTails-V declares CC-BY-NC-4.0; its
training-data terms must also be respected. Do not assume commercial permission.
The fork's code license does not supersede model or dataset restrictions.
Use for controlled safety research only; these checkpoints are not certified safe.
'''


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True, type=Path)
    p.add_argument('--run-name', required=True)
    p.add_argument('--export-name', help='New export name for a retry; does not overwrite prior staging')
    p.add_argument('--kind', choices=['rm', 'cm'], required=True)
    p.add_argument('--mode', choices=['lora', 'full'], required=True)
    p.add_argument('--repo-id', required=True)
    p.add_argument('--training-commit', required=True)
    p.add_argument('--upload', action='store_true')
    p.add_argument('--public-on-private-quota', action='store_true')
    args = p.parse_args()
    if any(not re.fullmatch(r'[A-Za-z0-9_.-]+', value) or value in {'.', '..'}
           for value in (args.run_name, args.export_name or args.run_name)):
        p.error('Run/export names must be single path components')
    if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', args.repo_id):
        p.error('Explicit namespace/model repo required')
    if not re.fullmatch('[a-f0-9]{40}', args.training_commit):
        p.error('Exact training source Git commit required')
    root = args.root.resolve()
    run = root / 'runs' / args.run_name / args.kind
    summary = json.loads((run / 'summary.json').read_text())
    if not (run / 'DONE').exists() or summary['kind'] != args.kind:
        raise RuntimeError('Only completed, validated matching runs can be exported')
    if args.mode == 'lora':
        if summary['reload_max_score_difference'] != 0 or summary['mode'] != 'lora_language_qv_plus_projector_and_score_head':
            raise RuntimeError('LoRA reload/mode guard failed')
        source = run / 'adapter'
        files = PROCESSOR_FILES | {'adapter_config.json', 'adapter_model.safetensors'}
    else:
        if (summary['mode'] != 'full_language_backbone_projector_score'
                or summary.get('rotary_buffer_precision', {}).get('dtype') != 'float32'
                or summary.get('fresh_reload_validation_pairs') != summary['n_validation']
                or not summary.get('fresh_reload_validation_ranking_equal')
                or summary.get('fresh_reload_validation_max_score_difference', float('inf')) > 1e-3):
            raise RuntimeError('Full validation/RoPE/reload guard failed; legacy one-pair check is insufficient')
        source = run / 'model'
        index = json.loads((source / 'model.safetensors.index.json').read_text())['weight_map']
        files = PROCESSOR_FILES | {'config.json', 'model.safetensors.index.json'} | set(index.values())
    destination = root / 'exports' / (args.export_name or args.run_name) / args.kind
    destination.mkdir(parents=True, exist_ok=False)
    for name in sorted(files):
        path = source / name
        if not path.exists():
            continue
        if path.is_symlink() or path.parent != source or not path.is_file():
            raise RuntimeError('Unexpected source path/symlink')
        target = destination / name
        if name.endswith('.json'):
            data = sanitize_config(json.loads(path.read_text()))
            if name == 'adapter_config.json':
                data['base_model_name_or_path'], data['revision'] = BASE, BASE_REVISION
            target.write_text(json.dumps(data, indent=2) + '\n')
        else:
            try:
                os.link(path, target)  # immutable weights/tokenizer only, no metadata edits
            except OSError:
                shutil.copyfile(path, target)
    (destination / 'README.md').write_text(model_card(args.kind, args.mode, summary, args.repo_id, args.training_commit))
    (destination / 'results.json').write_text(json.dumps({'training_commit': args.training_commit, **summary}, indent=2))
    licenses = Path(__file__).resolve().parents[1] / 'docs/licenses'
    shutil.copyfile(licenses / 'LLAMA2.txt', destination / 'LICENSE.Llama2.txt')
    shutil.copyfile(licenses / 'LLAMA2_USE_POLICY.md', destination / 'LLAMA2_USE_POLICY.md')
    (destination / 'NOTICE').write_text('Llama 2 is licensed under the LLAMA 2 COMMUNITY LICENSE, '
                                      'Copyright (c) Meta Platforms, Inc. All Rights Reserved.\n')
    for path in destination.iterdir():
        if path.suffix in {'.json', '.md', '.txt'} or path.name == 'NOTICE':
            if PRIVATE.search(path.read_text()):
                raise RuntimeError(f'Private content detected in release file {path.name}')
    checksums = {path.name: {'sha256': digest(path), 'bytes': path.stat().st_size}
                 for path in sorted(destination.iterdir())}
    (destination / 'file_manifest.json').write_text(json.dumps(checksums, indent=2) + '\n')
    if not args.upload:
        print('Exported sanitized artifact; upload not requested')
        return
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    api.create_repo(args.repo_id, repo_type='model', private=True, exist_ok=True)
    if not api.model_info(args.repo_id).private:
        raise RuntimeError('Existing public repo refused; public fallback only after private quota failure')
    try:
        commit = api.upload_folder(repo_id=args.repo_id, folder_path=destination,
                                   commit_message='Archive validated self-trained score weights and aggregate results')
    except Exception as error:
        if not args.public_on_private_quota or not private_quota_error(error):
            raise
        api.update_repo_settings(args.repo_id, private=False)
        commit = api.upload_folder(repo_id=args.repo_id, folder_path=destination,
                                   commit_message='Archive validated score artifacts after private-quota fallback')
    info = api.model_info(args.repo_id, revision=commit.oid, files_metadata=True)
    remote = {file.rfilename: file for file in info.siblings}
    for name, metadata in checksums.items():
        if name.endswith('.safetensors'):
            lfs = remote[name].lfs
            if lfs is None or lfs.sha256 != metadata['sha256']:
                raise RuntimeError('Remote weight SHA256 mismatch')
        else:
            downloaded = Path(hf_hub_download(args.repo_id, name, revision=commit.oid,
                                             cache_dir=root / 'hf-cache/release-verification'))
            if digest(downloaded) != metadata['sha256']:
                raise RuntimeError('Remote metadata SHA256 mismatch')
    receipt = {'repo_id': args.repo_id, 'revision': commit.oid, 'private': info.private,
               'weight_sha256_verified': True, 'metadata_sha256_verified': True,
               'files': checksums, 'optimizer_checkpoint_uploaded': False}
    (run / 'hf-upload-receipt.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps({k: receipt[k] for k in ('repo_id', 'revision', 'private',
                                             'weight_sha256_verified', 'metadata_sha256_verified')}, indent=2))


if __name__ == '__main__':
    main()
