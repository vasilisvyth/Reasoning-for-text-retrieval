import torch

from typing import Optional, List, Tensor
from transformers import MistralModel, MistralPreTrainedModel

def last_token_pool(last_hidden_states, attention_mask):
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

class MistralEmbedding(MistralPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = MistralModel(config)

        # Initialize weights and apply final processing
        #! not sure about it probably it is not needed because it is called in MistralModel as well
        # wasn't done in another paper here https://github.com/castorini/LiT5/blob/main/FiD/src/model.py#L25
        # self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        last_hidden_state = outputs.last_hidden_state
        embeddings = self.last_token_pool(last_hidden_state, attention_mask)

        return embeddings
#  MistralForSequenceEmbedding.from_pretrained()