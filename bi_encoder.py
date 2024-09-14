from torch import nn
import torch
from transformers import AutoModel, T5EncoderModel, MistralConfig
from mistral_embedding import MistralEmbedding
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training
import json

def model_init(model_name_or_dir):
   if 't5' in model_name_or_dir or 'checkpoints' in model_name_or_dir:
        T5EncoderModel._keys_to_ignore_on_load_unexpected = ["decoder.*"]  # decoder and anything after it, probably not needed at all
        model = T5EncoderModel.from_pretrained(model_name_or_dir)

    elif model_name_or_dir == 'mistralai/Mistral-7B-v0.1':
        
        print('using mistral model')
        model = MistralEmbedding.from_pretrained(model_name_or_dir, load_in_8bit = True, device_map="auto")
        model = prepare_model_for_int8_training(model)
        ''' maybe also needed in the config?
        "task_type": "FEATURE_EXTRACTION"
        "base_model_name_or_path": "mistralai/Mistral-7B-v0.1"
        '''
        peft_config = LoraConfig(**json.load(open("lora.json")))
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    else:
        print('using AutoModel')
        model = AutoModel.from_pretrained(model_name_or_dir)

    return model

def print_params_counter(model, model_name_or_dir):
    num_parameters = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters in {model_name_or_dir}: {num_parameters}")

class DenseBiEncoder(nn.Module):
    _keys_to_ignore_on_save = None #useful when loading a model
    def __init__(self, model_name_or_dir, scale_logits, right_loss, aggr_type) -> None:
        super().__init__()
        self.model = model_init(model_name_or_dir)
        self.encode_fn =  self.get_encode_function(aggr_type)

        self.scale_logits, self.right_loss = scale_logits, right_loss
        self.loss = nn.CrossEntropyLoss()
        ## Count the number of parameters 
        print_params_counter(self.model, model_name_or_dir)
        
    def get_encode_function(self, aggr_type):
        functions = {
            'avg': self.encode_mean_polling,
            'cls': self.encode_cls,
            'last': self.encode_last_token_pool
        }
        if aggr_type not in functions:
            raise ValueError(f"We do not support {aggr_type}")
        return functions[aggr_type]


    def encode_last_token_pool(self, input_ids, attention_mask):
        last_hidden_states = self.model(input_ids, attention_mask)
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    def encode_mean_polling(self, input_ids, attention_mask, **kwargs):

        hidden_states = self.model(input_ids, attention_mask,**kwargs).last_hidden_state # [batch_size, max_sentence_length, hidden_dim]
        masked_hidden_states = attention_mask.unsqueeze(dim=-1)*hidden_states # [batch_size, max_sentence_length, hidden_dim]
        masked_count = torch.sum(attention_mask,axis=1) # [batch_size]
        sum_hidden_states = torch.sum(masked_hidden_states,axis=1) # [batch_size, hidden_dim]
        mean_hidden_states = sum_hidden_states / masked_count.unsqueeze(dim=-1) # [batch_size, hidden_dim]
        return mean_hidden_states

    def encode_cls(self, input_ids, attention_mask, **kwargs):
        model_output = self.model(input_ids, attention_mask)
        # Perform pooling. In this case, cls pooling.
        sentence_embeddings = model_output[0][:, 0] # batch_size, hidden_dim
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings

    def score_all_combinations(self, queries, docs):

        q =self.encode_fn(queries.input_ids, queries.attention_mask) # batch_size, hidden_dim
        v = self.encode_fn(docs.input_ids, docs.attention_mask) # batch_size, hidden_dim
        v = torch.transpose(v,0,1) # hidden_dim, batch_size
        scores = torch.matmul(q, v) # batch_size, batch_size
        return scores
    
    def score_pairs(self, queries, docs):

        q =self.encode_fn(queries.input_ids, queries.attention_mask) # batch_size, hidden_dim
        v = self.encode_fn(docs.input_ids, docs.attention_mask) # batch_size, hidden_dim
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