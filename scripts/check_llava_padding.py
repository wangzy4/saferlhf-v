#!/usr/bin/env python3
"""GPU regression test for LLaVA cached decoding with left-padded batches.

Run only on a free GPU: CUDA_VISIBLE_DEVICES=<gpu> python scripts/check_llava_padding.py --root <data-root>
Uses one benign evaluation question that exposed the Transformers 4.47 failure.
"""
import argparse
import json
import re
from pathlib import Path

import torch
import transformers
from transformers import AutoProcessor, LlavaForConditionalGeneration

from infer_llava_pair import load_examples, prompt_for, validate_transformers_version


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=Path)
    args = parser.parse_args()
    validate_transformers_version(transformers.__version__)
    root = args.root
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    processor = AutoProcessor.from_pretrained(root / 'models/base', local_files_only=True, use_fast=False)
    processor.patch_size, processor.num_additional_image_tokens = 14, 1
    processor.vision_feature_select_strategy = 'default'
    processor.tokenizer.padding_side = 'left'
    examples = load_examples(root, 0)[1::3]
    index = next(i for i, item in enumerate(examples) if item['id'] == 'false_information/0')
    batch = examples[index // 4 * 4:index // 4 * 4 + 4]
    for name, suffix in [('base', 'models/base'), ('safe', 'models/safe/LLaVA_Safe_RLHF-V')]:
        model = LlavaForConditionalGeneration.from_pretrained(
            root / suffix, torch_dtype=torch.bfloat16, attn_implementation='sdpa',
            low_cpu_mem_usage=True, local_files_only=True, weights_only=True).to('cuda').eval()
        for items in [[examples[index]], batch]:
            inputs = processor(text=[prompt_for(x['question'], processor, 'hf') for x in items],
                               images=[x['image'] for x in items], return_tensors='pt', padding=True).to('cuda')
            if len(items) > 1 and not (inputs.attention_mask == 0).any():
                raise RuntimeError('Regression batch must contain padding')
            with torch.inference_mode():
                output = model.generate(**inputs, max_new_tokens=64, do_sample=False,
                                        repetition_penalty=1.0, pad_token_id=processor.tokenizer.pad_token_id)
            responses = processor.batch_decode(output[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
            text = responses[next(i for i, x in enumerate(items) if x['id'] == examples[index]['id'])]
            if re.search(r'(.{2,24}?)\1{9,}', text, re.S):
                raise RuntimeError(f'Early cached-generation loop: {name}/{len(items)}')
            print(json.dumps({'model': name, 'batch_size': len(items),
                              'padding_tokens': (inputs.attention_mask == 0).sum(1).tolist(),
                              'response': text}), flush=True)
        del model
        torch.cuda.empty_cache()
    print('PASS: benign cached left-padding regression for both checkpoints')


if __name__ == '__main__':
    main()
