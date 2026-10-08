"""Contracts for small full-backbone RM/CM DeepSpeed preflights (not RL)."""


def score_deepspeed_config(world_size, batch_pairs, accumulation=1):
    if any(type(v) is not int or v < 1 for v in (world_size, batch_pairs, accumulation)):
        raise ValueError('Positive integer world/batch/accumulation required')
    return {
        'train_micro_batch_size_per_gpu': batch_pairs,
        'gradient_accumulation_steps': accumulation,
        'train_batch_size': world_size * batch_pairs * accumulation,
        'bf16': {'enabled': True},
        'zero_optimization': {'stage': 2, 'overlap_comm': False,
                              'contiguous_gradients': True, 'reduce_bucket_size': 50000000,
                              'allgather_bucket_size': 50000000, 'ignore_unused_parameters': True},
        'gradient_clipping': 1.0,
        'steps_per_print': 1000000,
        'wall_clock_breakdown': False,
    }


def configure_full_score_training(model):
    """Full language backbone/projector/head, frozen vision and unused LM head.

    The vocabulary output head is disconnected from the scalar score loss. With
    the published zero weight decay it cannot update anyway; exclude its unused
    optimizer states explicitly. This is not language-backbone LoRA.
    """
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    for parameter in model.model.vision_tower.parameters():
        parameter.requires_grad_(False)
    for parameter in model.model.language_model.lm_head.parameters():
        parameter.requires_grad_(False)
    model.model.language_model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={'use_reentrant': False})
    model.model.config.use_cache = False
    model.model.language_model.config.use_cache = False
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def restore_llama_rope_fp32(model):
    """Undo DeepSpeed's blanket BF16 cast of nonpersistent rotary frequencies.

    ``inv_freq`` is absent from state_dict/safetensors. A BF16 conversion rounds
    its values permanently; simply calling .float() cannot recover them. Rebuild
    on CPU with the native Transformers function, as in a fresh HF load, then
    move the FP32 buffer to its original device. Only default Llama RoPE is
    certified here; extended/dynamic schemes need their own validation.
    """
    import torch
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

    count = 0
    for module in model.modules():
        if isinstance(module, LlamaRotaryEmbedding):
            if module.rope_type != 'default':
                raise ValueError('Only default Llama RoPE has been certified')
            device = module.inv_freq.device
            inv_freq, scaling = module.rope_init_fn(module.config, device=torch.device('cpu'))
            module.register_buffer('inv_freq', inv_freq.to(device=device, dtype=torch.float32),
                                   persistent=False)
            module.original_inv_freq = module.inv_freq
            module.attention_scaling = scaling
            count += 1
    if count != 1:
        raise ValueError('Expected exactly one native Llama rotary module')
    return {'modules': count, 'dtype': 'float32', 'reconstructed_on': 'cpu', 'persistent': False}
