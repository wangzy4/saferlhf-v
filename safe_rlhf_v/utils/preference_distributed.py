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
