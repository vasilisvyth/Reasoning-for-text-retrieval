from torch.utils.data import Dataset
from quest.common import tsv_utils
import pickle
from quest.common import example_utils
from tqdm import tqdm

def tokenize(tokenizer, max_length, input_text):
    if 'mistral' in tokenizer.name_or_path:
        #511 because we also need one token for eos!
        if not isinstance(input_text, list): input_text = input_text['input_texts']
        tokenized_input = tokenizer(input_text, max_length=max_length-1, return_attention_mask=True, padding=False, truncation=True)
        # append eos_token_id to every input_ids
        tokenized_input['input_ids'] = [input_ids + [tokenizer.eos_token_id] for input_ids in tokenized_input['input_ids']]
        tokenized_input['attention_mask'] = [input_ids + [1] for input_ids in tokenized_input['attention_mask']]
        # we need to attend to the eos token 
        # tokenized_input = tokenizer.pad(tokenized_input, padding=True, return_attention_mask=True, return_tensors='pt') # this is left padding automatically

    else:
        tokenized_input = tokenizer(
            input_text,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            pad_to_multiple_of=8
        )

    return tokenized_input

class EvaluateQueryDataset(Dataset):


    def __init__(self, gold_examples, dict_query_ids_queries, tokenizer) -> None:
        super().__init__()
        self.gold_examples = gold_examples

        self.ids, self.input = list(dict_query_ids_queries.keys()), list(dict_query_ids_queries.values())
        self.dict_query_ids_queries = dict_query_ids_queries

        self.input_ids = []
        self.attention_mask = []
        # for id, input_text in zip(self.ids, self.input):
        step_size = 128
        for i in range(0,len(self.input),step_size):
            input_text = self.input[i:i+step_size]
       
            #! removed padding from here probably not a problem
            batch_dict = tokenize(tokenizer, 64, input_text)
            
            # tokenized_input = tokenizer(
            #     input_text,
            #     padding=True,
            #     truncation=True,
            #     max_length=64,
            #     return_tensors="pt",
            #     pad_to_multiple_of=8
            # )
            self.input_ids.extend(batch_dict['input_ids'])
            self.attention_mask.extend(batch_dict['attention_mask'])
            
        assert(len(self.ids) == len(self.input_ids) == len(self.attention_mask))
        a=1
        # return tokenized_data

    def __len__(self):
        """
        Return the number of inputs i.e. len(queries)
        """
        return len(self.ids)

    def __getitem__(self, idx):

        id = self.ids[idx]
        input_id = self.input_ids[idx]
        attention_mask = self.attention_mask[idx]
        
        return {'ids':id, "input_ids": input_id,
                 'attention_mask':attention_mask
                }
        # return self.tokenized_data[idx]

class EvaluateDocsDataset(Dataset):


    def __init__(self, doc_text_map, tokenizer) -> None:
        super().__init__()
        
        self.ids, self.input =  list(doc_text_map.keys()), list(doc_text_map.values())
        self.input_ids = []
        self.attention_mask = []
        # for id, input_text in zip(self.ids, self.input):
        step_size = 512
        print('Start tokenizing the documents...')
        for i in tqdm(range(0,len(self.input),step_size)):
            input_text = self.input[i:i+step_size]
            batch_dict = tokenize(tokenizer, 512, input_text)
           
            self.input_ids.extend(batch_dict['input_ids'])
            self.attention_mask.extend(batch_dict['attention_mask'])
        print('Finished tokenizing the documents...')

    def __len__(self):
        """
        Return the number of inputs i.e. len(docs)
        """
        return len(self.ids)

    def __getitem__(self, idx):

        id = self.ids[idx]
        input_id = self.input_ids[idx]
        attention_mask = self.attention_mask[idx]
        
        return {'ids':id, "input_ids": input_id,
                 'attention_mask':attention_mask
                }
