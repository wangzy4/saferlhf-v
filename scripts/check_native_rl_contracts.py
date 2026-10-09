#!/usr/bin/env python3
"""Read-only, real-weight probe of the upstream SafeRLHF-V rollout/loss contract.

This is NOT RL training: actor/reference and scorer/critic share frozen models.
No optimizer, parameter update, API request, or sample export is performed.
"""
import argparse
import collections
import hashlib
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from datasets import load_dataset
from transformers import GenerationConfig

from safe_rlhf_v.models.pretrained_model import load_pretrained_models
from safe_rlhf_v.trainers.text_image_to_text.safe_rlhf_v import SafeRLHFVTrainer
from safe_rlhf_v.utils.masking import response_mask_from_lengths
from safe_rlhf_v.utils.preference_training import (image_from_value, response_text,
    ranking_diagnostics, binary_cost_diagnostics)
from safe_rlhf_v.utils.tools import is_same_tokenizer, seed_everything


class ReadOnlyEngine:
    def __init__(self, model):
        self.module = model
    def __call__(self, *args, **kwargs):
        return self.module(*args, **kwargs)
    def eval(self):
        self.module.eval()
    def train(self):
        self.module.train()


def load_checked(path, score):
    index = path / 'model.safetensors.index.json'
    if not index.is_file():
        raise RuntimeError('Probe requires indexed safetensors for strict key inventory')
    expected = set(json.loads(index.read_text())['weight_map'])
    model, tokenizer, processor = load_pretrained_models(
        path, model_max_length=2048, padding_side='left',
        trust_remote_code=False, dtype=torch.bfloat16, is_reward_model=score,
        auto_model_kwargs={'local_files_only': True, 'use_safetensors': True,
                           'weights_only': True, 'low_cpu_mem_usage': False,
                           'attn_implementation': 'sdpa'})
    if set(model.state_dict()) != expected or any(p.is_meta for p in model.parameters()):
        raise RuntimeError('RL loader inventory/meta check failed')
    buffers = [b for n,b in model.named_buffers() if n.endswith('rotary_emb.inv_freq')]
    if len(buffers) != 1 or buffers[0].dtype != torch.float32:
        raise RuntimeError('Native RL loader must preserve canonical FP32 RoPE')
    if processor is None or tokenizer.padding_side != 'left':
        raise RuntimeError('Expected native expanded left-padded multimodal processor')
    model.requires_grad_(False)
    return model.to('cuda').eval(), tokenizer, processor


def encode(processor, rows, responses=False):
    texts, images = [], []
    for row in rows:
        image = image_from_value(row['image'])
        if responses:
            for i in (1,2):
                texts.append(response_text(processor, row['question'], row[f'response_{i}']))
                images.append(image)
        else:
            texts.append(processor.apply_chat_template([{'role':'user','content':[
                {'type':'image'}, {'type':'text','text':row['question']}]}],
                tokenize=False, add_generation_prompt=True))
            images.append(image)
    inputs = processor(text=texts, images=images, return_tensors='pt', padding=True,
                       truncation=False)
    if inputs.input_ids.shape[1] > 2048:
        raise RuntimeError('Do not truncate contract probe')
    if not torch.all((inputs.input_ids == 32000).sum(1) == 576):
        raise RuntimeError('Expected 576 expanded image tokens')
    return {k:v.to('cuda') for k,v in inputs.items()}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--dataset',default='score-validation-20cat-80-v1')
    ap.add_argument('--port',type=int,default=29577)
    args=ap.parse_args()
    if args.output.exists():
        raise RuntimeError('Refuse to overwrite an existing probe')
    args.output.mkdir(parents=True)
    seed_everything(42)
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    dist.init_process_group('nccl',init_method=f'tcp://127.0.0.1:{args.port}',rank=0,world_size=1)
    root=args.root.resolve()
    data=root/'datasets'/args.dataset
    if not (data/'READY').is_file():raise RuntimeError('Dataset not READY')
    manifest=json.loads((data/'manifest.json').read_text())
    rows=list(load_dataset('parquet',data_files=str(data/'validation.parquet'),split='train'))
    run=root/'runs/rm-cm-full-20cat-1024-256-zero2-v1'
    for kind in ('rm','cm'):
        summary=json.loads((run/kind/'summary.json').read_text())
        if not (run/kind/'DONE').is_file() or summary['fresh_reload_validation_max_score_difference']!=0:
            raise RuntimeError('Only verified full score checkpoints may be probed')
    actor,tok,proc=load_checked(root/'models/base',False)
    reward,rtok,rproc=load_checked(run/'rm/model',True)
    cost,ctok,cproc=load_checked(run/'cm/model',True)
    if not all(is_same_tokenizer(tok,t) for t in (rtok,ctok)):
        raise RuntimeError('Critic tokenizers must match actor')
    report={'scope':'read_only_upstream_rollout_and_loss_probe_not_rl_training',
        'n_independent_pairs':len(rows),'shared_actor_reference_and_scorer_critic':True,
        'optimizer_updates':0,'parameters_frozen':True,'judge_api_calls':0,
        'dataset_manifest_sha256':hashlib.sha256((data/'manifest.json').read_bytes()).hexdigest(),
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'source_commit':os.environ.get('PROBE_SOURCE_COMMIT'),
        'strict_rl_loader_key_inventory':True,'canonical_rope_fp32':True}
    for kind,model,processor in [('rm',reward,rproc),('cm',cost,cproc)]:
        values=[]
        for start in range(0,len(rows),2):
            with torch.inference_mode():
                values.append(model(**encode(processor,rows[start:start+2],True),
                                    use_cache=False,num_logits_to_keep=1).end_scores.reshape(-1,2).cpu())
        scores=torch.cat(values)
        ids=[int(r['more_helpful_response_id' if kind=='rm' else 'safer_response_id']) for r in rows]
        result=ranking_diagnostics(scores,ids,kind,[r['category'] for r in rows])
        result['pairwise_accuracy']=result['pairwise_correct']/len(rows)
        if kind=='cm':
            unsafe=torch.tensor([[r[f'is_response_{i}_safe']=='no' for i in (1,2)] for r in rows])
            result.update(binary_cost_diagnostics(scores,unsafe))
        report[kind+'_independent_response_validation']=result
        print(kind,'independent pair accuracy',result['pairwise_accuracy'],flush=True)
    trainer=SafeRLHFVTrainer.__new__(SafeRLHFVTrainer)
    trainer.actor_model=ReadOnlyEngine(actor)
    trainer.actor_reference_model=trainer.actor_model
    trainer.reward_model=ReadOnlyEngine(reward)
    trainer.reward_critic_model=trainer.reward_model
    trainer.cost_model=ReadOnlyEngine(cost)
    trainer.cost_critic_model=trainer.cost_model
    trainer.tokenizer=trainer.reward_tokenizer=trainer.cost_tokenizer=tok
    trainer.infer_batch=trainer.reward_infer_batch=lambda batch:batch
    trainer.cfgs=SimpleNamespace(train_cfgs=SimpleNamespace(actor_gradient_checkpointing=False))
    trainer.lora_enabled=False
    trainer.generation_config=GenerationConfig(max_new_tokens=128,temperature=1.,top_p=1.,
        repetition_penalty=1.,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id,
        bos_token_id=tok.bos_token_id)
    trainer.episode_costs=collections.deque(maxlen=128)
    trainer.kl_coeff=.02;trainer.clip_range_score=50.;trainer.clip_range_value=5.
    trainer.clip_range_ratio=.2;trainer.gamma=1.;trainer.gae_lambda=.95
    trainer.log_lambda=torch.tensor(math.log(10.),device='cuda')
    reports=[]
    for start in (0,2):
        inf,train=trainer.rollout(encode(proc,rows[start:start+2]))
        t=train[0]; mask=t['response_mask']
        assert torch.equal(mask,response_mask_from_lengths(t['response_lens'],mask.shape[1],mask.device))
        for key in ('log_probs','ref_log_probs','reward_values','cost_values'):
            assert t[key].shape==mask.shape and torch.isfinite(t[key]).all()
        assert torch.equal(t['log_probs'],t['ref_log_probs'])
        rewards,costs=trainer.add_kl_divergence_regularization_with_cost(
            t['reward'],t['cost'],t['log_probs'],t['ref_log_probs'],mask)
        ra,rr=trainer.get_advantages_and_returns(t['reward_values'],rewards,mask,0)
        ca,cr=trainer.get_advantages_and_returns(t['cost_values'],costs,mask,0)
        losses=[trainer.actor_loss_fn_with_cost(t['log_probs'],t['log_probs'],ra,ca,mask),
                trainer.critic_loss_fn(t['reward_values'],t['reward_values'],rr,mask),
                trainer.critic_loss_fn(t['cost_values'],t['cost_values'],cr,mask)]
        assert all(torch.isfinite(x).all() for x in (ra,rr,ca,cr,*losses))
        reports.append({'response_lengths':t['response_lens'], 'all_shapes_and_mask_valid':True,
            'identical_actor_reference_kl_zero':True,'reward':t['reward'].cpu().tolist(),
            'cost':t['cost'].cpu().tolist(),'native_losses':[float(x) for x in losses]})
        print('native rollout/loss probe',start+2,'/4 valid',flush=True)
    assert all(p.grad is None for m in (actor,reward,cost) for p in m.parameters())
    report['four_generated_rollout_probes']=reports
    report['model_gradients_not_created']=True
    report['peak_allocated_gib']=torch.cuda.max_memory_allocated()/2**30
    report['peak_reserved_gib']=torch.cuda.max_memory_reserved()/2**30
    (args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    (args.output/'DONE').write_text('Read-only native contract probe; no optimizer updates.\n')
    dist.destroy_process_group()

if __name__=='__main__':
    try:
        main()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
