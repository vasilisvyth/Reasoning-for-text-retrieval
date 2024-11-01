'''
In this file we override many logical operators in order to be able to run our neurosymbolic algorithm
'''

import func_timeout
# import faiss
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from bi_encoder import DenseBiEncoder
from mistral_eval import last_token_pool
import torch
import pickle
import numpy as np
import torch.nn.functional as F
from copy import deepcopy
EXECUTION_TIMEOUT_TIME = 560

class Operations():
    @classmethod
    def init(cls, data, constant) -> None:
        cls.data = data
        cls.constant = constant

    @classmethod
    def calculate_score(cls,x,y, operation):
        func = getattr(cls,cls.data[operation])
        # func = cls.data[operation]
        result = func(x,y)
        return result

    @classmethod
    def avg(cls,x, y):
        return (x + y) / 2
    @classmethod
    def sum(cls,x, y):
        return (x + y)

    @classmethod
    def maximum(cls,x, y):
        return max(x, y)

    @classmethod
    def only_A(cls,x, y):
        return x

    @classmethod
    def subtract(cls,x, y):
        return x - y

    @classmethod
    def custom_score(cls, x):
        return 1 / (cls.constant + x)

class CustomDictOperations:
    def __init__(self, data):
        self.data = data

    def __and__(self, other):# __and__ intersection
        result = {}
        for key in set(self.data.keys()) & set(other.data.keys()): # intersected keys

            result[key] = Operations.calculate_score(self.data[key],other.data[key],'and')
        return CustomDictOperations(result)

    def __or__(self, other): #__or__ union
        result = {}
        for key in set(self.data.keys()) | set(other.data.keys()):
            #result[key] = max(self.data.get(key, float('-inf')), other.data.get(key, float('-inf')))
            result[key] = Operations.calculate_score(self.data.get(key, float('-inf')), other.data.get(key, float('-inf')),'or')
        return CustomDictOperations(result)

    def __sub__(self, other): #difference
        result = self.data.copy()
        for key in set(self.data.keys()) & set(other.data.keys()):
            result[key] = Operations.calculate_score(self.data.get(key, float('-inf')), other.data.get(key, float('-inf')),'diff')
      
        return CustomDictOperations(result)

    def __repr__(self):
        return repr(self.data) # still return a dict

    def items(self):
        return self.data.items()


def apply_threshold(threshold, top_k_indices, top_k_values):
    if not isinstance(threshold, str):
        top_k_indices = top_k_indices[top_k_values > threshold]
        top_k_values = top_k_values[top_k_values > threshold]
    return top_k_indices, top_k_values

def convert_to_1dlist(values):
    if isinstance(values, (torch.Tensor, np.ndarray)):
        if values.ndim > 1:
            values = values.squeeze().tolist()
        else:
            values = values.tolist()
    else:
        raise('Only support torch tensors and numpy arrays')
    
    return values

def remove_prefix(query, replace_find):
    if replace_find:
        query = query.replace('find ','')
    return query

def min_max_normalization(similarities):
    max_sim = torch.max(similarities,dim=-1).values
    min_sim = torch.min(similarities,dim=-1).values
    
    new_sim = (similarities - min_sim) / (max_sim - min_sim) 
    return new_sim

class VP_HuggingFace:

    @classmethod
    def init(cls, device, model_name, docs, tokenizer, k, method, replace_find, score, threshold='None', normalize_embeddings=False, 
            oracle_examples_dict = None, results= {}, use_cache = False):
        cls.tokenizer = tokenizer 
        cls.results = results
        cls.k = k
        cls.replace_find = replace_find
        cls.method = method
        print(f'method {method}')
        cls.device = torch.device('cpu') if use_cache else device
        cls.docs = docs.to(cls.device)
        cls.normalize_embeddings = normalize_embeddings
        cls.score = score
        cls.threshold = threshold
        cls.oracle_examples_dict = oracle_examples_dict
        cls.use_cache = use_cache
        if 'mistral' in model_name:
            cls.model = AutoModel.from_pretrained(model_name)
            cls.model.to(device)
        else:
            cls.model = DenseBiEncoder(model_name, False, False, 'avg')
            cls.model.to(device)

    @classmethod
    def find_docs(cls, original_query, query, largest=True):

        query = remove_prefix(query, cls.replace_find)
        if cls.method == 'mistral':
            TASK_DESCRIPTION = f'Given a web search query, retrieve relevant passages that answer the query'
            query = f'Instruct: {TASK_DESCRIPTION}\nQuery: {query}'
        
        if original_query not in cls.results: # if first subquestion that we encounter
            cls.results[original_query] = {}

        if query not in cls.results[original_query]:
            cls.results[original_query][query] = {}

        tokenized_query = cls.tokenizer.encode_plus(query, return_tensors='pt')
        input_ids, attention_mask = tokenized_query['input_ids'], tokenized_query['attention_mask']
        input_ids, attention_mask = input_ids.to(cls.device), attention_mask.to(cls.device)

        with torch.no_grad():
            if cls.method == 'mistral':
                if not cls.use_cache:
                    outputs = cls.model(input_ids, attention_mask)
                    embed_query = last_token_pool(outputs.last_hidden_state, attention_mask)
                    embed_query = F.normalize(embed_query, p=2, dim=1) 
                else:
                    embed_query = cls.results[original_query][query]['norm_q_embed'].cpu()
            else:
                embed_query = cls.model.encode_mean_polling(input_ids, attention_mask)

        similarities = torch.matmul(embed_query, cls.docs.t())
        new_sim = min_max_normalization(similarities)
        top_k_values, top_k_indices = torch.topk(new_sim, cls.k, dim=1, sorted=True, largest = largest)
        top_k_indices, top_k_values = apply_threshold(cls.threshold, top_k_indices, top_k_values)


        top_k_indices = convert_to_1dlist(top_k_indices)
        top_k_values = convert_to_1dlist(top_k_values)
        cls.results[original_query][query] = {
            'top_k_values': top_k_values,
            'top_k_indices': top_k_indices
        }
        cls.results[original_query][query]['norm_q_embed'] = embed_query

        map_ind_sim = {ind:val for ind, val in zip(top_k_indices, top_k_values)}
        ranked_doc = {key: Operations.custom_score(rank) for rank, key in enumerate(map_ind_sim, 1)} if cls.score != 'emb' else map_ind_sim

        return CustomDictOperations(ranked_doc)

#! use the code below
class VP_SentenceTransformer:
    @classmethod
    def init(cls, device, model_name, docs, k, method, replace_find, score, threshold='None',  normalize_embeddings=False, 
            oracle_examples_dict = None, results={}, use_cache=False):
        cls.results = results if use_cache else {}
        cls.k = k
        cls.replace_find = replace_find
        cls.method = method
        cls.use_cache = use_cache
        cls.device = torch.device('cpu') if use_cache else device
        cls.docs = docs.to(cls.device)
        cls.normalize_embeddings = normalize_embeddings
        cls.score = score
        cls.threshold = threshold
        cls.oracle_examples_dict = oracle_examples_dict
        cls.model = SentenceTransformer(model_name)
    
    @classmethod
    def replace_rows(cls, original_query):
        original_tensor = deepcopy(cls.docs)

        doc_id_text_dict = cls.oracle_examples_dict[original_query]
        # List of indices representing rows to be changed
        selected_doc_ids = list(doc_id_text_dict.keys())
        new_doc_texts = list(doc_id_text_dict.values())

        replacement_tensor = cls.model.encode(new_doc_texts, convert_to_tensor=True, normalize_embeddings = cls.normalize_embeddings)

        # Convert the list of indices to a tensor
        selected_doc_ids = torch.tensor(selected_doc_ids)

        # Replace the corresponding rows in the original tensor
        original_tensor[selected_doc_ids] = replacement_tensor
        return original_tensor
    
    @classmethod
    def find_docs(cls, original_query, query, largest=True):
        query = remove_prefix(query, cls.replace_find)

        if original_query not in cls.results: # if first subquestion that we encounter
            cls.results[original_query] = {}

        if cls.method == 'bge-large':
            INSTRUCTION = 'Represent this sentence for searching relevant passages: '#Represent this sentence for searching relevant passages: '
            query = INSTRUCTION+query
        if not cls.use_cache:
            embed_query = cls.model.encode(query,convert_to_tensor=True, normalize_embeddings = cls.normalize_embeddings)
        else:
            embed_query = cls.results[original_query][query]['norm_q_embed'].cpu()

        if cls.oracle_examples_dict is not None: # WHEN WE CHANGE EVERY TIME
            docs = cls.replace_rows(original_query)
            similarities = torch.matmul(embed_query, docs.t())
        else:
            similarities = torch.matmul(embed_query, cls.docs.t())


        new_sim = min_max_normalization(similarities)
        top_k_values, top_k_indices = torch.topk(new_sim, cls.k, dim=-1, sorted=True, largest = largest)
        top_k_indices, top_k_values = apply_threshold(cls.threshold, top_k_indices, top_k_values)

        top_k_indices = convert_to_1dlist(top_k_indices)
        top_k_values = convert_to_1dlist(top_k_values)
        map_ind_sim = {ind:val for ind, val in zip(top_k_indices, top_k_values)}
        cls.results[original_query][query] = {
            'top_k_values': top_k_values,
            'top_k_indices': top_k_indices
        }
        cls.results[original_query][query]['norm_q_embed'] = embed_query
        ranked_doc = {key: Operations.custom_score(rank) for rank, key in enumerate(map_ind_sim, 1)} if cls.score != 'emb' else map_ind_sim
        return CustomDictOperations(ranked_doc)


class VP_BM25:
    @classmethod
    def init(cls, model_name, docs, k, replace_find, score, threshold='None',
            oracle_examples_dict = None):
        cls.results = {}
        cls.k = k
        cls.replace_find = replace_find
        cls.docs = docs
        cls.score = score
        cls.threshold = threshold
        cls.oracle_examples_dict = oracle_examples_dict
        with open(model_name, 'rb') as f:
            cls.model = pickle.load(f)

    @classmethod
    def find_docs(cls, original_query, query, largest=True):
        top_k_indices, top_k_values = cls.model.get_docs_ids_and_scores(query, topk= cls.k)
        top_k_values = torch.tensor(top_k_values)
        #todo check that no problem is created from not converting to 1d lists here

        new_sim = min_max_normalization(top_k_values)
        top_k_values, top_k_indices = torch.topk(new_sim, cls.k, dim=-1, sorted=True, largest = largest)
        top_k_indices, top_k_values = apply_threshold(cls.threshold, top_k_indices, top_k_values)

        top_k_indices = convert_to_1dlist(top_k_indices)
        top_k_values = convert_to_1dlist(top_k_values)
        map_ind_sim = {ind:val for ind, val in zip(top_k_indices, top_k_values)}
        cls.results[original_query][query] = {
            'top_k_values': top_k_values,
            'top_k_indices': top_k_indices
        }
        ranked_doc = {key: Operations.custom_score(rank) for rank, key in enumerate(map_ind_sim, 1)} if cls.score != 'emb' else map_ind_sim
        return CustomDictOperations(ranked_doc)


def safe_execute(code_string: str, keys=None):
    '''
    If the keys parameter is a list of variable names, the function attempts to retrieve the values of those specific
    variables from the local variables obtained after executing the code using exec.
    If a variable is not found, its corresponding position in the list will contain None.
    If keys are provided, the function retrieves the values of these variables from the executed code.
    '''
    def execute(x):
        try:
            exec(x)
            locals_ = locals() # create copy of the current local variables
            return locals_

        except Exception as e:
            import traceback
            traceback.print_exc()
            return None
        
    # If the execution exceeds this time limit raise exception
    try: 
        ans = func_timeout.func_timeout(EXECUTION_TIMEOUT_TIME, execute, args=(code_string,))
    except func_timeout.FunctionTimedOut:
        ans = None
        print('execution timeout')
    return ans

def synthesize_program(result: str, prefix: str) -> str:
    program = prefix
    generated_txt = result.split('\n')
    subquestions_block = False
    subquestions = []
    programs = []
    inside_code_block = False
    for i, line in enumerate(generated_txt):
        if '```' in line:
            inside_code_block = not inside_code_block
        elif 'python' in line or line=='':
            continue
        elif inside_code_block:
            program += line + '\n'

    return program
