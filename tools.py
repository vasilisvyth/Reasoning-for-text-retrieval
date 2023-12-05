import func_timeout
# import faiss
from transformers import AutoTokenizer
from bi_encoder import DenseBiEncoder
import torch
import pickle
EXECUTION_TIMEOUT_TIME = 260
# score_calculation = {
#     'b25':

# }
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# index = faiss.read_index(f'zero_shot_large.bin')
# docs = torch.load('scores_docs_check_4000.pt')
# checkpoints/google/t5-v1_1-base/model/checkpoint-4000
# 'google/t5-v1_1-small'
model = DenseBiEncoder('checkpoints/89a4eb2b-b3/checkpoint-6000/', False, False)
model.to(device)
with open('dum_bm25_obj.pickle', 'rb') as f:
    retriever = pickle.load(f)

class CustomDictOperations:
    def __init__(self, data):
        self.data = data

    def __and__(self, other):
        result = {}
        for key in set(self.data.keys()) & set(other.data.keys()): # intersected keys
            result[key] = self.data[key] + other.data[key]
        return CustomDictOperations(result)

    def __or__(self, other):
        result = {}
        for key in set(self.data.keys()) | set(other.data.keys()):
            result[key] = max(self.data.get(key, float('-inf')), other.data.get(key, float('-inf')))
        return CustomDictOperations(result)

    def __sub__(self, other):
        result = self.data.copy()
        for key in set(self.data.keys()) & set(other.data.keys()):
            result[key] -= other.data[key]
        return CustomDictOperations(result)

    def __repr__(self):
        return repr(self.data) # still return a dict

    def items(self):
        return self.data.items()

class DocumentFinder:
    # model = DenseBiEncoder('google/t5-v1_1-small', False, False).to(device)  # Your model initialization
    tokenizer = AutoTokenizer.from_pretrained('google/t5-v1_1-base')  # Your tokenizer initialization
    docs = torch.load('checkpoints/89a4eb2b-b3/checkpoint-6000/scores_docs_check_6000.pt', map_location=device)  # Your docs initialization
    method = 't5-base'
    k = 2000#1000
    results = {}

    @classmethod
    def find_docs(cls, original_query, query, largest=True):
        if DocumentFinder.method == 't5-base':
        # query = query.replace('find ','')

            tokenized_query = cls.tokenizer.encode_plus(query, return_tensors='pt')
            input_ids, attention_mask = tokenized_query['input_ids'], tokenized_query['attention_mask']
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)

            with torch.no_grad():
                embed_query = model.encode(input_ids, attention_mask)

            similarities = torch.matmul(embed_query, cls.docs.t())


            max_sim = torch.max(similarities,dim=1).values
            min_sim = torch.min(similarities,dim=1).values
            
            new_sim = (similarities - min_sim) / (max_sim - min_sim) 

            top_k_values, top_k_indices = torch.topk(new_sim, cls.k, dim=1, sorted=True, largest = largest)

            # new_k_values, new_k_indices = torch.topk(new_sim, cls.k, dim=1, sorted=True, largest = largest)

            top_k_values, top_k_indices = top_k_values.squeeze().tolist(), top_k_indices.squeeze().tolist()
            
        elif DocumentFinder.method == 'bm25':
            top_k_indices, top_k_values = retriever.get_docs_ids_and_scores(query, topk= cls.k)
   
        else:
            raise(f'{DocumentFinder.method} is not allowed. We only allow bm25 or t5-base')

        if original_query not in cls.results:
            cls.results[original_query] = {}

        map_ind_values = {ind:val for ind, val in zip(top_k_indices, top_k_values)}

        cls.results[original_query][query] = {
            'top_k_values': top_k_values,
            'top_k_indices': top_k_indices
        }

        # answers = set(top_k_indices)
        custom_dict = CustomDictOperations(map_ind_values)


        return custom_dict
# def find_docs(query):
#     return set({1})


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
            print(f'Exception: {e}')
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
        if 'python' in line or line=='':
            continue

        if line.strip() == '```':
            inside_code_block = not inside_code_block
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