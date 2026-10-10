"""PPO adapter for v2.1; explicitly not the single-step SGD theorem."""
import torch
from safe_rlhf_v.trainers.text_image_to_text.safe_rlhf_v import SafeRLHFVTrainer
from safe_rlhf_v.utils.masking import last_valid_indices
from safe_rlhf_v.utils.tools import masked_mean


class ActiveSafetyPPOTrainer(SafeRLHFVTrainer):
    """Loss adapter, not a standalone active-learning training entrypoint.

    The round orchestrator must initialize ``active_multiplier``, supply pooled
    corrected costs, disable the inherited moving-average/log-space dual update,
    and call the projected linear dual update only AFTER the fixed pool update.
    """

    def actor_loss_fn_with_cost(self, log_probs, old_log_probs, reward_advantages, cost_advantages, mask):
        # No batch-label-dependent normalization; use the fixed round multiplier.
        advantages = reward_advantages - self.active_multiplier * cost_advantages
        ratios = torch.exp(log_probs - old_log_probs)
        return -masked_mean(torch.minimum(advantages * ratios,
            advantages * ratios.clamp(1 - self.clip_range_ratio, 1 + self.clip_range_ratio)), mask)

    def add_kl_divergence_regularization_with_cost(self, reward, cost, log_probs, ref_log_probs, sequence_mask):
        # Corrected costs can be negative or >1. NO score clipping here.
        end = last_valid_indices(sequence_mask).unsqueeze(-1)
        kl = (log_probs - ref_log_probs).masked_fill(~sequence_mask.bool(), 0)
        rewards = (-self.kl_coeff * kl).scatter_add(-1, end, reward.to(kl.dtype).unsqueeze(-1))
        # KL belongs to the known reward side only, not the human-risk cost.
        costs = torch.zeros_like(kl).scatter_add(-1, end, cost.to(kl.dtype).unsqueeze(-1))
        return rewards, costs
