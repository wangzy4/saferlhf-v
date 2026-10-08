"""Training contracts, isolated from DeepSpeed and optional multimodal imports.

AST loading executes the actual source methods, not reimplemented loss functions.
Torch cases use CPU tensors/fake engines; this is NOT full training validation.
"""
import ast
import importlib.util
import math
from pathlib import Path
from types import MethodType, SimpleNamespace as NS
import unittest

ROOT = Path(__file__).resolve().parents[1]
TRAINER = 'safe_rlhf_v/trainers/text_image_to_text/safe_rlhf_v.py'
try:
    import torch
except ImportError:
    torch = None


def source_method(path, class_name, method_name, namespace):
    tree = ast.parse((ROOT / path).read_text())
    body = tree.body
    if class_name is not None:
        cls = next(n for n in body if isinstance(n, ast.ClassDef) and n.name == class_name)
        body = cls.body
    method = next(n for n in body if isinstance(n, ast.FunctionDef) and n.name == method_name)
    method.decorator_list = []
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, method], type_ignores=[]))
    exec(compile(module, str(ROOT / path), 'exec'), namespace)
    return namespace[method_name]


def load_helper(path):
    spec = importlib.util.spec_from_file_location(Path(path).stem, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DatasetContracts(unittest.TestCase):
    def sample(self):
        return {'question': 'Synthetic question', 'image': object(), 'response_1': 'First',
                'response_2': 'Second', 'safer_response_id': 1, 'more_helpful_response_id': 2,
                'response_1_harmless_rate': 3, 'response_2_harmless_rate': -2}

    def formatter(self, name, sample):
        fn = source_method('safe_rlhf_v/configs/format_dataset.py', name,
                           'format_preference_sample', {})
        return fn(None, sample)

    def test_rm_uses_helpfulness_not_safety(self):
        better, worse, _ = self.formatter('RM_V', self.sample())
        self.assertEqual(better[-1]['content'][0]['text'], 'Second')
        self.assertEqual(worse[-1]['content'][0]['text'], 'First')

    def test_cm_higher_score_means_less_safe(self):
        for safer in [1, 2]:
            sample = self.sample()
            sample['safer_response_id'] = safer
            better, worse, meta = self.formatter('CM_V', sample)
            unsafe = 3 - safer
            self.assertEqual(better[-1]['content'][0]['text'], sample[f'response_{unsafe}'])
            self.assertEqual(worse[-1]['content'][0]['text'], sample[f'response_{safer}'])
            self.assertEqual(meta['is_better_safe'], -sample[f'response_{unsafe}_harmless_rate'])
            self.assertEqual(meta['is_worse_safe'], -sample[f'response_{safer}_harmless_rate'])

    def test_zero_rating_remains_neutral_not_negative_one(self):
        sample = self.sample()
        sample['response_1_harmless_rate'] = 0
        self.assertEqual(self.formatter('CM_V', sample)[2]['is_worse_safe'], 0)

    def test_collator_pair_and_image_order(self):
        calls = []
        def processor(**kwargs):
            calls.append(kwargs)
            return {}
        fn = source_method('safe_rlhf_v/datasets/text_image_to_text/preference.py',
                           'PreferenceCollator_cm', '__call__',
                           {'get_current_device': lambda: 'cpu', 'torch': NS(Tensor=object)})
        samples = [{'image': f'image{i}', 'better_conversation': f'cost-high{i}',
                    'worse_conversation': f'cost-low{i}', 'better_response_lens': i+1,
                    'worse_response_lens': i+3, 'is_better_safe': 2, 'is_worse_safe': -3}
                   for i in range(2)]
        result = fn(NS(processor=processor, padding_side='right'), samples)
        self.assertEqual(calls[0]['images'], ['image0', 'image1', 'image0', 'image1'])
        self.assertEqual(calls[0]['text'], ['cost-high0', 'cost-high1', 'cost-low0', 'cost-low1'])
        self.assertEqual(result['meta_info']['response_lens'], [1, 2, 3, 4])
        self.assertEqual(result['meta_info']['is_better_safe'], [2, 2])
        self.assertEqual(result['meta_info']['is_worse_safe'], [-3, -3])

    def test_llava_processor_expands_image_tokens(self):
        configure = load_helper('safe_rlhf_v/utils/processors.py').configure_llava_processor
        processor = NS()
        config = NS(model_type='llava', vision_config=NS(model_type='clip_vision_model', patch_size=14),
                    vision_feature_select_strategy='default')
        configure(processor, config)
        self.assertEqual((processor.patch_size, processor.num_additional_image_tokens,
                          processor.vision_feature_select_strategy), (14, 1, 'default'))
        other = NS()
        configure(other, NS(model_type='qwen2_vl'))
        self.assertEqual(vars(other), {})


class ModelRoutingContracts(unittest.TestCase):
    def initialize(self, critic_path='cost-critic', vocab_groups=None):
        calls = []
        groups = vocab_groups or {}
        class Tokenizer(NS):
            def __len__(self):
                return 100
        def load(path, **kwargs):
            calls.append((path, kwargs))
            return NS(), Tokenizer(group=groups.get(path, 'actor')), NS()
        namespace = {'load_pretrained_models': load,
                     'is_same_tokenizer': lambda a, b: a.group == b.group,
                     'GenerationConfig': lambda **kwargs: NS(**kwargs)}
        fn = source_method(TRAINER, 'SafeRLHFVTrainer', 'init_models_with_cost_model', namespace)
        model_cfg = NS(actor_model_name_or_path='actor', reward_model_name_or_path='reward',
                       reward_critic_model_name_or_path='reward-critic', cost_model_name_or_path='cost',
                       cost_critic_model_name_or_path=critic_path, model_max_length=4096,
                       trust_remote_code=False, max_new_tokens=8, temperature=1, top_p=1,
                       repetition_penalty=1)
        train_cfg = NS(freeze_mm_proj=False, freeze_vision_tower=True,
                       freeze_language_model=False, processor_kwargs={})
        trainer = NS(cfgs=NS(model_cfgs=model_cfg, train_cfgs=train_cfg),
                     ds_train_cfgs={'zero_optimization': {'stage': 0}},
                     ds_eval_cfgs={'zero_optimization': {'stage': 0}})
        fn(trainer)
        return trainer, calls

    def test_explicit_cost_critic_checkpoint_and_freeze_flags(self):
        _, calls = self.initialize()
        self.assertEqual(calls[-1][0], 'cost-critic')
        self.assertFalse(calls[-1][1]['freeze_mm_proj'])
        self.assertTrue(calls[-1][1]['freeze_vision_tower'])
        self.assertFalse(calls[-1][1]['freeze_language_model'])

    def test_unspecified_cost_critic_falls_back_to_cm(self):
        _, calls = self.initialize(critic_path=None)
        self.assertEqual(calls[-1][0], 'cost')

    def test_scalar_model_tokenizers_not_overwritten(self):
        trainer, _ = self.initialize(vocab_groups={'reward': 'foreign-reward'})
        self.assertEqual(trainer.reward_tokenizer.group, 'foreign-reward')
        self.assertIs(trainer.cost_tokenizer, trainer.tokenizer)
        trainer, _ = self.initialize(vocab_groups={'cost': 'foreign-cost'})
        self.assertEqual(trainer.cost_tokenizer.group, 'foreign-cost')
        self.assertIs(trainer.reward_tokenizer, trainer.tokenizer)

    def test_both_tokenwise_critics_must_match_actor(self):
        for path in ['reward-critic', 'cost-critic']:
            with self.assertRaisesRegex(ValueError, 'critic tokenizer'):
                self.initialize(vocab_groups={path: 'foreign'})


@unittest.skipIf(torch is None, 'PyTorch is required; run also in the inference environment')
class TensorContracts(unittest.TestCase):
    def setUp(self):
        self.masking = load_helper('safe_rlhf_v/utils/masking.py')
        self.namespace = {'torch': torch, 'last_valid_indices': self.masking.last_valid_indices,
                          'response_mask_from_lengths': self.masking.response_mask_from_lengths,
                          'masked_mean': source_method('safe_rlhf_v/utils/tools.py', None,
                                                       'masked_mean', {'torch': torch})}

    def method(self, name):
        return source_method(TRAINER, 'SafeRLHFVTrainer', name, self.namespace)

    def test_last_indices_both_padding_sides_and_invalid_masks(self):
        mask = torch.tensor([[1, 1, 0, 0], [0, 0, 1, 1]])
        indices = self.masking.last_valid_indices(mask)
        self.assertEqual(indices.tolist(), [1, 3])
        self.assertEqual(indices.dtype, torch.long)
        self.assertEqual(indices.device, mask.device)
        for invalid in [torch.zeros(1, 2), torch.tensor([[1, 2]]), torch.zeros(1, 0), torch.ones(2)]:
            with self.assertRaises(ValueError):
                self.masking.last_valid_indices(invalid)

    def test_response_lengths_include_one_token_and_reject_invalid(self):
        mask = self.masking.response_mask_from_lengths([1, 3], 3)
        self.assertEqual(mask.tolist(), [[True, False, False], [True, True, True]])
        self.assertEqual(self.masking.response_mask_from_lengths([1], 1).shape, (1, 1))
        for lengths in [[0], [4], [1.5], [], [True]]:
            with self.assertRaises(ValueError):
                self.masking.response_mask_from_lengths(lengths, 3)

    def test_reward_head_gathers_real_end_and_backpropagates(self):
        fn = source_method('safe_rlhf_v/models/llava.py', 'AccustomedLlavaRewardModel', 'forward',
                           dict(self.namespace, ScoreModelOutput=lambda **kwargs: NS(**kwargs)))
        hidden = torch.tensor([[[1.], [2.], [3.], [900.]],
                               [[4.], [5.], [6.], [7.]]], requires_grad=True)
        head = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            head.weight.fill_(1)
        model = NS(model=lambda *a, **k: NS(hidden_states=[hidden]), score_head=head)
        mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]])
        result = fn(model, input_ids=torch.ones(2, 4, dtype=torch.long), attention_mask=mask)
        self.assertEqual(result.end_scores.squeeze(-1).tolist(), [3, 7])
        self.assertEqual(result.end_index.tolist(), [2, 3])
        result.end_scores.sum().backward()
        self.assertEqual(hidden.grad.squeeze(-1).tolist(), [[0, 0, 1, 0], [0, 0, 0, 1]])
        with self.assertRaisesRegex(ValueError, 'expand LLaVA image tokens'):
            fn(model, input_ids=torch.ones(2, 3), attention_mask=torch.ones(2, 3))

    @unittest.skipUnless(importlib.util.find_spec('transformers'), 'Native Transformers needed')
    def test_native_tiny_llava_right_padded_batch_matches_single_and_backward(self):
        from transformers import LlavaConfig, LlavaForConditionalGeneration
        from transformers.models.llava.modeling_llava import LlavaPreTrainedModel
        tree = ast.parse((ROOT / 'safe_rlhf_v/models/llava.py').read_text())
        classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
        future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
        module = ast.fix_missing_locations(ast.Module(body=[future] + classes, type_ignores=[]))
        namespace = dict(self.namespace, nn=torch.nn, LlavaPreTrainedModel=LlavaPreTrainedModel,
                         LlavaForConditionalGeneration=LlavaForConditionalGeneration,
                         ScoreModelOutput=lambda **kwargs: NS(**kwargs))
        exec(compile(module, 'safe_rlhf_v/models/llava.py', 'exec'), namespace)
        config = LlavaConfig(
            vision_config={'model_type': 'clip_vision_model', 'hidden_size': 16,
                           'intermediate_size': 32, 'num_hidden_layers': 1,
                           'num_attention_heads': 4, 'image_size': 8, 'patch_size': 4},
            text_config={'model_type': 'llama', 'vocab_size': 40, 'hidden_size': 16,
                         'intermediate_size': 32, 'num_hidden_layers': 1,
                         'num_attention_heads': 4, 'num_key_value_heads': 4,
                         'max_position_embeddings': 128, 'pad_token_id': 0},
            image_token_index=39, image_seq_length=4, pad_token_id=0,
            vision_feature_layer=-1, vision_feature_select_strategy='default')
        torch.manual_seed(7)
        model = namespace['AccustomedLlavaRewardModel'](config).eval()
        ids = torch.tensor([[1, 39, 39, 39, 39, 5, 2, 0, 0],
                            [1, 39, 39, 39, 39, 6, 7, 8, 2]])
        pixels = torch.randn(2, 3, 8, 8)
        batch = model(input_ids=ids, attention_mask=ids.ne(0), pixel_values=pixels)
        self.assertEqual(batch.end_index.tolist(), [6, 8])
        with torch.no_grad():
            for row, length in enumerate([7, 9]):
                single = model(input_ids=ids[row:row+1, :length],
                               attention_mask=ids[row:row+1, :length].ne(0),
                               pixel_values=pixels[row:row+1])
                torch.testing.assert_close(batch.end_scores[row:row+1], single.end_scores,
                                           rtol=1e-5, atol=1e-6)
        batch.end_scores.sum().backward()
        self.assertIsNotNone(model.score_head.weight.grad)
        self.assertTrue(torch.isfinite(model.score_head.weight.grad).all())
        self.assertGreater(model.score_head.weight.grad.abs().sum().item(), 0.)

    def test_cm_loss_gradient_direction_and_zero_anchor(self):
        fn = source_method('safe_rlhf_v/trainers/text_to_text/cm.py', 'CMTrainer', 'loss',
                           {'torch': torch, 'F': torch.nn.functional})
        for scale in [0., 1.]:
            scores = torch.zeros(2, 1, requires_grad=True)
            trainer = NS(model=lambda **k: NS(scores=scores.unsqueeze(1), end_scores=scores),
                         infer_batch=lambda b: {}, scale_coeff=scale,
                         cfgs=NS(train_cfgs=NS(regularization=0)))
            batch = {'input_ids': torch.ones(2, 2),
                     'meta_info': {'is_better_safe': [3], 'is_worse_safe': [-3]}}
            fn(trainer, batch)['loss'].backward()
            self.assertLess(scores.grad[0].item(), 0)  # descent raises harmful cost
            self.assertGreater(scores.grad[1].item(), 0)  # descent lowers safe cost
        # With both ratings zero the absolute term adds a constant, not an anchor gradient.
        grads = []
        for scale in [0., 1.]:
            scores = torch.zeros(2, 1, requires_grad=True)
            trainer.scale_coeff = scale
            trainer.model = lambda **k: NS(scores=scores.unsqueeze(1), end_scores=scores)
            batch['meta_info'] = {'is_better_safe': [0], 'is_worse_safe': [0]}
            fn(trainer, batch)['loss'].backward()
            grads.append(scores.grad)
        torch.testing.assert_close(grads[0], grads[1])

    def test_terminal_rewards_costs_and_kl_exclude_padding(self):
        trainer = NS(kl_coeff=.1, clip_range_score=50.)
        mask = self.masking.response_mask_from_lengths([3, 1], 3)
        logp = torch.tensor([[-1., -.5, -.2], [-.4, 99., 99.]])
        ref = torch.full_like(logp, -1.)
        reward, cost = self.method('add_kl_divergence_regularization_with_cost')(
            trainer, torch.tensor([2., 4.]), torch.tensor([3., 5.]), logp, ref, mask)
        self.assertEqual(reward[1, 1:].tolist(), [0, 0])
        self.assertEqual(cost[1, 1:].tolist(), [0, 0])
        torch.testing.assert_close(reward[:, 0], torch.tensor([0., 3.94]))
        torch.testing.assert_close(cost[:, 0], torch.tensor([0., 5.06]))
        # Opposite KL signs cancel the lambda normalization: same KL strength for any lambda.
        for multiplier in [0., 2., 20.]:
            terminal = torch.zeros_like(logp)
            terminal[0, 2] = (2 - multiplier * 3) / (1 + multiplier)
            terminal[1, 0] = (4 - multiplier * 5) / (1 + multiplier)
            combined = (reward - multiplier * cost) / (1 + multiplier)
            torch.testing.assert_close(combined, terminal - .1 * (logp-ref).masked_fill(~mask, 0))

    def test_gae_padded_batch_matches_unpadded_single(self):
        trainer = NS(gamma=.9, gae_lambda=.95)
        mask = self.masking.response_mask_from_lengths([1, 3], 3)
        values = torch.tensor([[.5, 900, 900], [.5, .5, .5]])
        rewards = torch.tensor([[2., 900, 900], [0., 0., 3.]])
        fn = self.method('get_advantages_and_returns')
        advantages, returns = fn(trainer, values, rewards, mask, 0)
        for row, length in enumerate([1, 3]):
            single_a, single_r = fn(trainer, values[row:row+1, :length],
                                   rewards[row:row+1, :length], mask[row:row+1, :length], 0)
            torch.testing.assert_close(advantages[row, :length], single_a[0])
            torch.testing.assert_close(returns[row, :length], single_r[0])
        self.assertEqual(advantages[0].tolist(), [1.5, 0., 0.])

    def test_cost_advantage_penalizes_unsafe_actions_and_ignores_padding(self):
        trainer = NS(log_lambda=torch.tensor(math.log(2.)), clip_range_ratio=.2)
        for advantage, sign in [(2., 1), (-2., -1)]:
            logp = torch.zeros(1, 2, requires_grad=True)
            loss = self.method('actor_loss_fn_with_cost')(
                trainer, logp, torch.zeros_like(logp), torch.zeros_like(logp),
                torch.tensor([[advantage, 999.]]), torch.tensor([[True, False]]))
            loss.backward()
            self.assertGreater(sign * logp.grad[0, 0].item(), 0)
            self.assertEqual(logp.grad[0, 1].item(), 0)

    def test_cpu_rl_step_updates_actor_critics_and_lambda_with_real_mask(self):
        class Engine:
            def __init__(self, tensor, output_name):
                self.parameter = torch.nn.Parameter(tensor)
                self.optimizer = torch.optim.SGD([self.parameter], lr=.01)
                self.output_name = output_name
            def __call__(self, **kwargs):
                return NS(**{self.output_name: self.parameter})
            def backward(self, loss):
                self.optimizer.zero_grad()
                loss.backward()
                self.last_gradient = self.parameter.grad.clone()
            def step(self):
                self.optimizer.step()

        self.namespace.update(
            gather_log_probabilities=lambda logits, ids:
                logits.log_softmax(-1).gather(-1, ids.unsqueeze(-1)).squeeze(-1),
            get_current_device=lambda: 'cpu', is_main_process=lambda: True,
            get_all_reduce_mean=lambda value: value, get_all_reduce_max=lambda value: value,
            dist=NS(reduce=lambda *a, **k: None, broadcast=lambda *a, **k: None,
                    barrier=lambda: None, ReduceOp=NS(AVG=None)))
        for episode_cost, cap, step in [(2., None, 1), (-2., None, 1), (2., 2.1, 1), (2., None, 0)]:
            trainer = NS(episode_costs=[episode_cost], global_step=step, lambda_update_delay_steps=1,
                         threshold=0., log_lambda=torch.nn.Parameter(torch.tensor(math.log(2.))),
                         log_lambda_max=math.log(cap) if cap else None, kl_coeff=.1,
                         clip_range_score=50., gamma=1., gae_lambda=.95,
                         clip_range_ratio=.2, clip_range_value=.2)
            trainer.log_lambda_optimizer = torch.optim.SGD([trainer.log_lambda], lr=.1)
            trainer.actor_model = Engine(torch.zeros(2, 6, 10), 'logits')
            trainer.reward_critic_model = Engine(torch.zeros(2, 6, 1), 'scores')
            trainer.cost_critic_model = Engine(torch.zeros(2, 6, 1), 'scores')
            for name in ['add_kl_divergence_regularization_with_cost', 'get_advantages_and_returns',
                         'actor_loss_fn_with_cost']:
                setattr(trainer, name, MethodType(self.method(name), trainer))
            critic_loss = source_method('safe_rlhf_v/trainers/text_to_text/ppo.py', 'PPOTrainer',
                                        'critic_loss_fn', self.namespace)
            trainer.critic_loss_fn = MethodType(critic_loss, trainer)
            mask = self.masking.response_mask_from_lengths([1, 3], 3)
            old_logp = torch.full((2, 3), -math.log(10.)).masked_fill(~mask, 0)
            inference = {'input_ids': torch.full((2, 6), 2, dtype=torch.long),
                         'attention_mask': torch.ones(2, 6, dtype=torch.long)}
            batch = {'response_lens': [1, 3], 'response_mask': mask,
                     'log_probs': old_logp, 'ref_log_probs': old_logp.clone(),
                     'reward': torch.ones(2), 'cost': torch.full((2,), 2.),
                     'reward_values': torch.zeros(2, 3), 'cost_values': torch.zeros(2, 3)}
            metrics = self.method('rl_step')(trainer, inference, batch)
            expected = 2. if step == 0 else 2. * math.exp(.2 * episode_cost)
            if cap:
                expected = min(expected, cap)
            self.assertAlmostEqual(metrics['train/lambda'], expected, places=5)
            self.assertEqual(metrics['train/mean_generated_length'], 2.)
            self.assertEqual(metrics['train/max_generated_length'], 3.)
            for engine in [trainer.actor_model, trainer.reward_critic_model, trainer.cost_critic_model]:
                self.assertEqual(engine.last_gradient[0, :4].abs().sum().item(), 0.)
                self.assertGreater(engine.last_gradient[0, 4].abs().sum().item(), 0.)
                self.assertEqual(engine.last_gradient[0, 5].abs().sum().item(), 0.)

    def test_rollout_keeps_actor_ids_zero_logp_valid_and_one_token_shape(self):
        def gather(logits, ids):
            return logits.log_softmax(-1).gather(-1, ids.unsqueeze(-1)).squeeze(-1)
        self.namespace['gather_log_probabilities'] = gather
        fn = self.method('rollout')
        for lengths in [[1], [1, 3]]:
            batch, width = len(lengths), 6
            ids = torch.full((batch, width), 2, dtype=torch.long)
            actor = {'input_ids': ids, 'attention_mask': torch.ones_like(ids)}
            logits = torch.zeros(batch, width, 10)
            logits[:, :, 2] = 80  # float32 logp is exactly zero, still a valid response
            values = torch.arange(batch * (width-1)).reshape(batch, width-1).float()
            trainer = NS(set_train=lambda **k: None, actor_step=lambda b: (actor, lengths),
                         actor_model=lambda **k: NS(logits=logits),
                         actor_reference_model=lambda **k: NS(logits=logits),
                         reward_model_step=lambda b: dict(input_ids=ids+1, reward=torch.zeros(batch),
                                                          reward_values=values),
                         cost_model_step=lambda b: dict(cost=torch.zeros(batch), cost_values=values))
            inference, training = fn(trainer, {})
            self.assertIs(inference[0]['input_ids'], ids)
            self.assertEqual(training[0]['log_probs'].shape, (batch, max(lengths)))
            self.assertTrue((training[0]['log_probs'] == 0).all())
            self.assertEqual(training[0]['response_mask'].sum(1).tolist(), lengths)


if __name__ == '__main__':
    unittest.main()
