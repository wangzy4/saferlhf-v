"""Native processor configuration needed for aligned LLaVA token positions."""


def configure_llava_processor(processor, config):
    if config.model_type != 'llava':
        return
    processor.patch_size = config.vision_config.patch_size
    processor.num_additional_image_tokens = (1 if config.vision_config.model_type == 'clip_vision_model' else 0)
    processor.vision_feature_select_strategy = config.vision_feature_select_strategy
