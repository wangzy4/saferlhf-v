"""CPU contracts; real multi-GPU DeepSpeed preflight is a separate recorded run."""
import unittest

from safe_rlhf_v.utils.preference_distributed import score_deepspeed_config

try:
    import torch
    from transformers import CLIPVisionConfig, LlamaConfig, LlavaConfig
    from safe_rlhf_v.models.llava import AccustomedLlavaRewardModel
    from safe_rlhf_v.utils.preference_distributed import configure_full_score_training
except ImportError:
    torch = None


class DistributedConfigTests(unittest.TestCase):
    def test_global_batch_and_no_offload(self):
        cfg = score_deepspeed_config(4, 4)
        self.assertEqual(cfg['train_batch_size'], 16)
        self.assertEqual(cfg['zero_optimization']['stage'], 2)
        self.assertTrue(cfg['bf16']['enabled'])
        self.assertNotIn('offload_optimizer', cfg['zero_optimization'])
        self.assertEqual(cfg['gradient_clipping'], 1.0)

    def test_invalid_batch_contracts(self):
        for args in [(0, 4), (4, 0), (True, 4), (4, 1.5), (4, 4, -1)]:
            with self.assertRaises(ValueError):
                score_deepspeed_config(*args)

    @unittest.skipIf(torch is None, 'torch/transformers unavailable')
    def test_full_score_gradients_and_logits_optimization(self):
        config = LlavaConfig(
            vision_config=CLIPVisionConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                                           num_attention_heads=4, image_size=4, patch_size=2),
            text_config=LlamaConfig(vocab_size=64, hidden_size=16, intermediate_size=32,
                                    num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4,
                                    pad_token_id=0, bos_token_id=1, eos_token_id=2),
            image_token_index=63, image_seq_length=4, vision_feature_layer=-1)
        model = AccustomedLlavaRewardModel(config)
        parameters = configure_full_score_training(model)
        self.assertTrue(parameters)
        self.assertFalse(any(p.requires_grad for p in model.model.vision_tower.parameters()))
        self.assertFalse(model.model.language_model.lm_head.weight.requires_grad)
        self.assertTrue(model.model.language_model.model.embed_tokens.weight.requires_grad)
        ids = torch.tensor([[1, 63, 63, 63, 63, 5, 2], [1, 63, 63, 63, 63, 6, 2]])
        inputs = dict(input_ids=ids, attention_mask=torch.ones_like(ids), pixel_values=torch.randn(2, 3, 4, 4),
                      use_cache=False)
        model.eval()
        with torch.no_grad():
            full = model(**inputs).end_scores
            reduced = model(**inputs, num_logits_to_keep=1).end_scores
        torch.testing.assert_close(full, reduced, atol=0, rtol=0)
        model.train()
        model(**inputs, num_logits_to_keep=1).end_scores.square().mean().backward()
        for prefix in ('score_head', 'model.multi_modal_projector', 'model.language_model.model'):
            grads = [p.grad for n, p in model.named_parameters() if n.startswith(prefix) and p.grad is not None]
            self.assertTrue(grads, prefix)
            self.assertGreater(sum(float(g.abs().sum()) for g in grads), 0, prefix)


if __name__ == '__main__':
    unittest.main()
