import torch
import torch.nn as nn
import torch.utils.checkpoint
from transformers import AutoConfig, LlavaPreTrainedModel
from transformers.models.llava.modeling_llava import LlavaForConditionalGeneration

from safe_rlhf_v.models.reward_model import ScoreModelOutput
from safe_rlhf_v.utils.masking import last_valid_indices


class AccustomedLlavaModel(LlavaForConditionalGeneration):
    """Accustomed Interface for LLaVA model"""


class AccustomedLlavaRewardModel(LlavaPreTrainedModel):
    """Accustomed Interface for LLaVA reward model"""

    supports_gradient_checkpointing = True

    def __init__(self, config: AutoConfig):
        super().__init__(config)
        setattr(self, self.base_model_prefix, AccustomedLlavaModel(config))
        self.score_head = nn.Linear(self.model.language_model.lm_head.in_features, 1, bias=False)

    @property
    def processor_available(self):
        return True

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs,
        )

        last_hidden_state = outputs.hidden_states[-1]
        scores = self.score_head(last_hidden_state).float()
        B, _, _ = scores.size()

        if attention_mask is None:
            attention_mask = torch.ones(last_hidden_state.shape[:2], device=last_hidden_state.device,
                                        dtype=torch.bool)
        if attention_mask.shape != last_hidden_state.shape[:2]:
            raise ValueError('Attention mask must match hidden states; expand LLaVA image tokens '
                             'in the processor before scoring')
        end_index = last_valid_indices(attention_mask.to(last_hidden_state.device))
        rows = torch.arange(B, device=last_hidden_state.device)
        end_last_hidden_state = last_hidden_state[rows, end_index]
        end_scores = scores[rows, end_index]

        return ScoreModelOutput(
            scores=scores,  # size = (B, L, D)
            end_scores=end_scores,  # size = (B, D)
            last_hidden_state=last_hidden_state,  # size = (B, L, E)
            end_last_hidden_state=end_last_hidden_state,  # size = (B, E)
            end_index=end_index,  # size = (B,)
        )
