import func_timeout
# import faiss
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from bi_encoder import DenseBiEncoder
import torch
import pickle
import numpy as np
from copy import deepcopy
EXECUTION_TIMEOUT_TIME = 560
# score_calculation = {
#     'b25':

# }
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# docs = torch.load('scores_docs_check_4000.pt')
# checkpoints/google/t5-v1_1-base/model/checkpoint-4000
# 'google/t5-v1_1-small'
# model = DenseBiEncoder('checkpoints/4913e0dd-b8/checkpoint-13500/', False, False)
# model.to(device)
# with open('dum_bm25_obj.pickle', 'rb') as f:
#     retriever = pickle.load(f)
# model = SentenceTransformer('sentence-transformers/gtr-t5-large')

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
            # tmp_result = 0
            # counter = 0
            # if key in self.data:
            #     tmp_result += self.data[key]
            #     counter +=1
            # if key in other.data:
            #     tmp_result += other.data[key]
            #     counter +=1

            # result[key] = tmp_result / counter #min(self.data[key],other.data[key])
            #result[key] = (self.data[key] * other.data[key])
            result[key] = (self.data[key] + other.data[key]) / 2
            #result[key] = Operations.calculate_score(self.data[key],other.data[key],'and')
        return CustomDictOperations(result)
    # def __and__(self, *others):
    #     result = {}
    #     key_sets = [set(d.data.keys()) for d in [self] + list(others)]
    #     common_keys = set.intersection(*key_sets)

    #     for key in common_keys:
    #         values = [d.data[key] for d in [self] + list(others)]
    #         result[key] = sum(values) / len(values)

    #     return CustomDictOperations(result)


    def __or__(self, other): #__or__ union
        result = {}
        for key in set(self.data.keys()) | set(other.data.keys()):
            result[key] = max(self.data.get(key, float('-inf')), other.data.get(key, float('-inf')))
            #result[key] = Operations.calculate_score(self.data.get(key, float('-inf')), other.data.get(key, float('-inf')),'or')
        return CustomDictOperations(result)
    # def __or__(self, *others):
    #     result = {}
    #     key_sets = [set(d.data.keys()) for d in [self] + list(others)]
    #     all_keys = set.union(*key_sets)

    #     for key in all_keys:
    #         values = [d.data.get(key, float('-inf')) for d in [self] + list(others)]
    #         result[key] = max(values)

    #     return CustomDictOperations(result)

    def __sub__(self, other): #__sub__
        result = self.data.copy()
        for key in set(self.data.keys()) & set(other.data.keys()):
            result[key] -= other.data[key]
      
        return CustomDictOperations(result)

    def __repr__(self):
        return repr(self.data) # still return a dict

    def items(self):
        return self.data.items()

# dict1 = CustomDictOperations({'a': 1, 'b': 2, 'c': 3})
# dict2 = CustomDictOperations({'a': 2, 'b': 3, 'd': 4})
# dict3 = CustomDictOperations({'a': 3, 'b': 4, 'e': 5})

# result = dict1.union(dict2, dict3)

class DocumentFinder:
    # tokenizer = AutoTokenizer.from_pretrained('google/t5-v1_1-base')  # Your tokenizer initialization
    # docs = torch.load('checkpoints/4913e0dd-b8/scores_docs_check_13500.pt', map_location=device)  # Your docs initialization
    # # docs = torch.from_numpy(np.load('checkpoints/zero_shot_t5_large.npy'))
    # method = 't5-base'
    # results = {}

    @classmethod
    def init(cls, model_name, docs, tokenizer, k, method, replace_find, score, threshold='None',  use_sentence_transformer=False,normalize_embeddings=False, 
            return_dict=True, oracle_examples_dict = {}):
        cls.tokenizer = tokenizer 
        cls.results = {}
        cls.k = k
        cls.replace_find = replace_find
        cls.method = method
        cls.docs = docs.cuda() if method == 'bge-large' else docs
        cls.normalize_embeddings = normalize_embeddings
        cls.score = score
        cls.threshold = threshold
        cls.return_dict = return_dict
        cls.oracle_examples_dict = oracle_examples_dict
        if use_sentence_transformer:
            cls.model = SentenceTransformer(model_name)
        elif 'bm25' in model_name:
            with open(model_name, 'rb') as f:
                 cls.model = pickle.load(f)
        else:
            cls.model = DenseBiEncoder(model_name, False, False)

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
        # query = query.replace('what are some ','')
        # query = query.replace('what are ','')
        if cls.replace_find:
            query = query.replace('find ','')


        
        if DocumentFinder.method == 't5-base':
        # query = query.replace('find ','')

            tokenized_query = cls.tokenizer.encode_plus(query, return_tensors='pt')
            input_ids, attention_mask = tokenized_query['input_ids'], tokenized_query['attention_mask']
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)

            with torch.no_grad():
                embed_query = cls.model.encode_mean_polling(input_ids, attention_mask)

            similarities = torch.matmul(embed_query, cls.docs.t())


            max_sim = torch.max(similarities,dim=1).values
            min_sim = torch.min(similarities,dim=1).values
            
            new_sim = (similarities - min_sim) / (max_sim - min_sim) 

            top_k_values, top_k_indices = torch.topk(new_sim, cls.k, dim=1, sorted=True, largest = largest)

            # new_k_values, new_k_indices = torch.topk(new_sim, cls.k, dim=1, sorted=True, largest = largest)

            # top_k_values, top_k_indices = top_k_values.squeeze().tolist(), top_k_indices.squeeze().tolist()
            
        elif DocumentFinder.method == 'bm25':
            top_k_indices, top_k_values = cls.model.get_docs_ids_and_scores(query, topk= cls.k)
            top_k_values = torch.tensor(top_k_values) 
        elif DocumentFinder.method == 'gtr' or DocumentFinder.method == 'bge-large':
            if DocumentFinder.method == 'bge-large':
                Instruction = 'Represent this sentence for searching relevant passages: '#Represent this sentence for searching relevant passages: '
                query = Instruction+query
            embed_query = cls.model.encode(query,convert_to_tensor=True, normalize_embeddings = cls.normalize_embeddings)

            if cls.oracle_examples_dict is not None:
                docs = cls.replace_rows(original_query)
                similarities = torch.matmul(embed_query, docs.t())
            else:
                similarities = torch.matmul(embed_query, cls.docs.t())


            max_sim = torch.max(similarities,dim=-1).values
            min_sim = torch.min(similarities,dim=-1).values
            
            new_sim = (similarities - min_sim) / (max_sim - min_sim) 

            top_k_values, top_k_indices = torch.topk(new_sim, cls.k, dim=-1, sorted=True, largest = largest)

            # new_k_values, new_k_indices = torch.topk(new_sim, cls.k, dim=1, sorted=True, largest = largest)

        else:
            raise(f'{DocumentFinder.method} is not allowed. We only allow bm25, gtr, bge-large or t5-base')

        if original_query not in cls.results: # if first subquestion that we encounter
            cls.results[original_query] = {}

        if not isinstance(cls.threshold, str):
            top_k_indices = top_k_indices[top_k_values > cls.threshold]
            top_k_values = top_k_values[top_k_values > cls.threshold]

        if torch.is_tensor(top_k_values) or isinstance(top_k_values, np.ndarray):
            if len(top_k_values.shape) > 1:
                top_k_values = top_k_values.squeeze()

            top_k_values = top_k_values.tolist()

        if torch.is_tensor(top_k_indices) or isinstance(top_k_indices, np.ndarray):
            if len(top_k_indices.shape) > 1:
                top_k_indices = top_k_indices.squeeze()
            top_k_indices = top_k_indices.tolist()

        map_ind_values = {ind:val for ind, val in zip(top_k_indices, top_k_values)}

        cls.results[original_query][query] = {
            'top_k_values': top_k_values,
            'top_k_indices': top_k_indices
        }
        if cls.score == 'emb':
            ranked_doc = map_ind_values
        else:
            # ranked_doc = {key: 1/(60+rank) for rank, key in enumerate(map_ind_values, 1)}
            ranked_doc = {key: Operations.custom_score(rank) for rank, key in enumerate(map_ind_values, 1)}
        # answers = set(top_k_indices)
        if cls.return_dict:
            custom_dict = CustomDictOperations(ranked_doc)
            return custom_dict
        else:
            return ranked_doc


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
            # if keys is None:
            #     return locals_.get('answer', None)
            # else:
            #     return [locals_.get(k, None) for k in keys]
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
        # elif line == '# Define the subquestions':
        #     subquestions_block = True
        # elif line == '# Combine using the correct logical operator if needed':
        #     subquestions_block = False
        # elif subquestions_block:
        #     dict = safe_execute(line)
        #     subquestions.extend(list(dict.values()))


        # program += line + '\n'
            
    return program

# for i in range(1):
#     safe_execute("DocumentFinder.find_docs('lala','subquery1')")
#     safe_execute("DocumentFinder.find_docs('lala','subquery2')")
#     safe_execute("DocumentFinder.find_docs('qe','saba')")

# print(DocumentFinder.results)