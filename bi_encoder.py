from torch import nn
import torch
from transformers import AutoModel, T5EncoderModel


class DenseBiEncoder(nn.Module):
    _keys_to_ignore_on_save = None #useful when loading a model
    def __init__(self, model_name_or_dir, scale_logits, right_loss) -> None:
        super().__init__()
        # if 't5' in model_name_or_dir:
            # check https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/models/Transformer.py#L53
        T5EncoderModel._keys_to_ignore_on_load_unexpected = ["decoder.*"] #decoder and anything after it, probably not needed at all
        self.model = T5EncoderModel.from_pretrained(model_name_or_dir)
        # else:
        #     self.model = AutoModel.from_pretrained(model_name_or_dir)
        self.scale_logits, self.right_loss = scale_logits, right_loss
        self.loss = nn.CrossEntropyLoss()
        ## Count the number of parameters 
        num_parameters = sum(p.numel() for p in self.model.parameters())
        print(f"Number of parameters in {model_name_or_dir}: {num_parameters}")
        
    def encode(self, input_ids, attention_mask, **kwargs):

        hidden_states = self.model(input_ids, attention_mask,**kwargs).last_hidden_state # [batch_size, max_sentence_length, hidden_dim]
        masked_hidden_states = attention_mask.unsqueeze(dim=-1)*hidden_states # [batch_size, max_sentence_length, hidden_dim]
        masked_count = torch.sum(attention_mask,axis=1) # [batch_size]
        sum_hidden_states = torch.sum(masked_hidden_states,axis=1) # [batch_size, hidden_dim]
        mean_hidden_states = sum_hidden_states / masked_count.unsqueeze(dim=-1) # [batch_size, hidden_dim]
        return mean_hidden_states

    def score_all_combinations(self, queries, docs):

        q =self.encode(queries.input_ids, queries.attention_mask) # batch_size, hidden_dim
        v = self.encode(docs.input_ids, docs.attention_mask) # batch_size, hidden_dim
        v = torch.transpose(v,0,1) # hidden_dim, batch_size
        scores = torch.matmul(q, v) # batch_size, batch_size
        return scores
    
    def score_pairs(self, queries, docs):

        q =self.encode(queries.input_ids, queries.attention_mask) # batch_size, hidden_dim
        v = self.encode(docs.input_ids, docs.attention_mask) # batch_size, hidden_dim
        scores = (q * v).sum(dim=-1)
        return scores

    def forward(self, query_ids, doc_ids, queries, docs):

        predictions = self.score_all_combinations(queries, docs) # B, B
        if self.scale_logits:
            predictions = 100*predictions
        labels = torch.arange(predictions.shape[0], device = next(self.model.parameters()).device)
        loss = self.loss(predictions, labels)
        if self.right_loss:
            loss = loss+self.loss(predictions.t(), labels)
            loss = loss / 2
            
        return loss, predictions

    def save_pretrained(self, model_dir, state_dict=None):
        """
        Save the model's checkpoint to a directory
        Parameters
        ----------
        model_dir: str or Path
            path to save the model checkpoint to
        """
        self.model.save_pretrained(model_dir, state_dict=state_dict)

    @classmethod
    def from_pretrained(cls, model_name_or_dir, scale_logits, right_loss):
        """
        Load model checkpoint for a path or directory
        Parameters
        ----------
        model_name_or_dir: str
            a HuggingFace's model or path to a local checkpoint
        """
        return cls(model_name_or_dir, scale_logits, right_loss)