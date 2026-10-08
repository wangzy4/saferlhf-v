"""Contracts for real small-data RM/CM training; GPU/model assets not required."""
import copy
import importlib.util
import io
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest

from test_training_contracts import source_method

ROOT = Path(__file__).resolve().parents[1]
try:
    import torch
except ImportError:
    torch = None


class ImportContracts(unittest.TestCase):
    def test_package_import_does_not_load_audio_templates(self):
        subprocess.run([sys.executable, '-c', "import sys, safe_rlhf_v; "
                        "assert 'safe_rlhf_v.configs.format_dataset' not in sys.modules"],
                       cwd=ROOT, check=True)


@unittest.skipUnless(torch is not None, 'torch is not installed')
class SmokeContracts(unittest.TestCase):
    def setUp(self):
        from safe_rlhf_v.utils import preference_training
        self.helpers = preference_training

    def test_equivalent_image_encodings_share_group(self):
        from PIL import Image
        image = Image.new('RGB', (4, 4), (24, 28, 32))
        encodings = []
        for fmt in ('PNG', 'BMP'):
            stream = io.BytesIO()
            image.save(stream, format=fmt)
            encodings.append(stream.getvalue())
        self.assertNotEqual(*encodings)
        self.assertEqual(*(self.helpers.image_key({'bytes': value}) for value in encodings))

    def test_eos_appended_once(self):
        for text in ['ASSISTANT: response', 'ASSISTANT: response</s>']:
            processor = NS(apply_chat_template=lambda *a, **kw: text,
                           tokenizer=NS(eos_token='</s>'))
            self.assertEqual(self.helpers.response_text(processor, 'Question', 'Response'),
                             'ASSISTANT: response</s>')

    def test_loss_and_gradient_match_actual_rm_cm_methods(self):
        raw = torch.tensor([[0.3, -0.2], [-0.1, 0.8], [0.4, 0.5]])
        ids = torch.tensor([1, 2, 1])
        ratings = torch.tensor([[3, -2], [0, 1], [-1, -3]])
        for kind in ('rm', 'cm'):
            for scale in (0.0, 1.0):
                current = raw.clone().requires_grad_()
                loss, accuracy = self.helpers.preference_loss(current, ids, kind, ratings,
                                                               regularization=0.01, scale=scale)
                high_ids = ids - 1 if kind == 'rm' else 2 - ids
                rows = torch.arange(3)
                high, low = current[rows, high_ids], current[rows, 1 - high_ids]
                end = torch.cat([high, low]).unsqueeze(-1)
                trainer = NS(model=lambda **kw: NS(scores=end.unsqueeze(1), end_scores=end),
                             infer_batch=lambda b: {}, scale_coeff=scale,
                             cfgs=NS(train_cfgs=NS(regularization=0.01)))
                batch = {'input_ids': torch.ones(6, 2), 'meta_info': {
                    'is_better_safe': (-ratings[rows, high_ids]).tolist(),
                    'is_worse_safe': (-ratings[rows, 1 - high_ids]).tolist()}}
                fn = source_method(f'safe_rlhf_v/trainers/text_to_text/{kind}.py',
                                   kind.upper() + 'Trainer', 'loss',
                                   {'torch': torch, 'F': torch.nn.functional})
                actual = fn(trainer, batch)
                torch.testing.assert_close(loss, actual['loss'])
                torch.testing.assert_close(accuracy, actual['accuracy'])
                torch.testing.assert_close(torch.autograd.grad(loss, current, retain_graph=True)[0],
                                           torch.autograd.grad(actual['loss'], current)[0])

    def test_invalid_preference_contracts_fail(self):
        with self.assertRaises(ValueError):
            self.helpers.preference_loss(torch.zeros(2, 3), [1, 2], 'rm')
        with self.assertRaises(ValueError):
            self.helpers.preference_loss(torch.zeros(2, 2), [1, 0], 'cm')
        with self.assertRaises(ValueError):
            self.helpers.preference_loss(torch.zeros(2, 2), [1, 2], 'cm')

    def test_zero_cm_rating_adds_constant_not_anchor_gradient(self):
        gradients = []
        losses = []
        for scale in (0.0, 1.0):
            scores = torch.zeros(1, 2, requires_grad=True)
            loss, _ = self.helpers.preference_loss(scores, [1], 'cm', torch.zeros_like(scores),
                                                   regularization=0, scale=scale)
            gradients.append(torch.autograd.grad(loss, scores)[0])
            losses.append(loss.item())
        torch.testing.assert_close(*gradients)
        self.assertAlmostEqual(losses[1] - losses[0], 2 * __import__('math').log(2), places=6)

    def test_ranking_diagnostics_follow_cost_direction_and_count_ties(self):
        scores = torch.tensor([[1., 0.], [2., 2.], [-1., 2.]])
        rm = self.helpers.ranking_diagnostics(scores, [1, 2, 2], 'rm', ['a', 'a', 'b'])
        cm = self.helpers.ranking_diagnostics(scores, [2, 1, 1], 'cm', ['a', 'a', 'b'])
        self.assertEqual(rm, cm)
        self.assertEqual(rm['pairwise_correct'], 2)
        self.assertEqual(rm['pairwise_ties'], 1)
        self.assertEqual(rm['by_category']['a']['accuracy'], 0.5)
        with self.assertRaises(ValueError):
            self.helpers.ranking_diagnostics(scores, [1, 2, 2], 'unknown', ['a', 'a', 'b'])

    def test_binary_cost_auc_confusion_and_single_class(self):
        scores = torch.tensor([[-2., -1.], [0., 2.]])
        unsafe = torch.tensor([[False, True], [False, True]])
        result = self.helpers.binary_cost_diagnostics(scores, unsafe)
        self.assertEqual(result['true_unsafe'], 1)
        self.assertEqual(result['false_safe'], 1)
        self.assertEqual(result['true_safe'], 2)
        self.assertEqual(result['false_unsafe'], 0)
        self.assertEqual(result['safety_label_accuracy_at_zero'], 0.75)
        self.assertEqual(result['response_safety_auc'], 0.75)
        self.assertEqual(result['balanced_safety_label_accuracy_at_zero'], 0.75)
        tied = self.helpers.binary_cost_diagnostics(torch.zeros(2), torch.tensor([True, False]))
        self.assertEqual(tied['response_safety_auc'], 0.5)
        one_class = self.helpers.binary_cost_diagnostics(torch.zeros(2), torch.zeros(2, dtype=torch.bool))
        self.assertIsNone(one_class['response_safety_auc'])
        with self.assertRaises(ValueError):
            self.helpers.binary_cost_diagnostics(torch.tensor([float('nan')]), torch.tensor([False]))

    @unittest.skipUnless(importlib.util.find_spec('peft'), 'peft is not installed')
    def test_native_score_lora_training_and_adapter_reload(self):
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import CLIPVisionConfig, LlamaConfig, LlavaConfig
        from safe_rlhf_v.models.llava import AccustomedLlavaRewardModel
        config = LlavaConfig(vision_config=CLIPVisionConfig(hidden_size=16, intermediate_size=32,
                             num_hidden_layers=1, num_attention_heads=4, image_size=4, patch_size=2),
                             text_config=LlamaConfig(vocab_size=64, hidden_size=16,
                             intermediate_size=32, num_hidden_layers=1, num_attention_heads=4,
                             num_key_value_heads=4, pad_token_id=0, bos_token_id=1, eos_token_id=2),
                             image_token_index=63, image_seq_length=4, vision_feature_layer=-1)
        torch.manual_seed(42)
        base = AccustomedLlavaRewardModel(config)
        initial_state = copy.deepcopy(base.state_dict())
        model = get_peft_model(base, LoraConfig(r=2, lora_alpha=4,
                     target_modules=r'.*language_model.*\.(q_proj|v_proj)',
                     modules_to_save=['score_head', 'multi_modal_projector']))
        self.assertFalse(any(p.requires_grad for n, p in model.named_parameters() if 'vision_tower' in n))
        inputs = {'input_ids': torch.tensor([[1, 63, 63, 63, 63, 3, 5, 2],
                                             [1, 63, 63, 63, 63, 4, 6, 2]]),
                  'attention_mask': torch.ones(2, 8, dtype=torch.long),
                  'pixel_values': torch.randn(2, 3, 4, 4), 'use_cache': False}
        optimizer = torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=0.01)
        output = model(**inputs).end_scores.reshape(1, 2)
        loss, _ = self.helpers.preference_loss(output, [1], 'rm')
        loss.backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                            for n, p in model.named_parameters() if 'score_head' in n))
        optimizer.step()
        model.eval()
        with torch.no_grad():
            expected = model(**inputs).end_scores
        with tempfile.TemporaryDirectory() as path:
            model.save_pretrained(path, safe_serialization=True)
            fresh = AccustomedLlavaRewardModel(config)
            fresh.load_state_dict(initial_state)
            restored = PeftModel.from_pretrained(fresh, path).eval()
            with torch.no_grad():
                torch.testing.assert_close(expected, restored(**inputs).end_scores)


if __name__ == '__main__':
    unittest.main()
