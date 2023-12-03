import func_timeout
# import faiss
from transformers import AutoTokenizer
from bi_encoder import DenseBiEncoder
import torch
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

class DocumentFinder:
    # model = DenseBiEncoder('google/t5-v1_1-small', False, False).to(device)  # Your model initialization
    tokenizer = AutoTokenizer.from_pretrained('google/t5-v1_1-base')  # Your tokenizer initialization
    docs = torch.load('checkpoints/89a4eb2b-b3/checkpoint-6000/scores_docs_check_6000.pt', map_location=device)  # Your docs initialization
    k = 2000#1000
    results = {}

    @classmethod
    def find_docs(cls, original_query, query, largest=True):
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
        if original_query not in cls.results:
            cls.results[original_query] = {}

        cls.results[original_query][query] = {
            'top_k_values': top_k_values,
            'top_k_indices': top_k_indices
        }

        answers = set(top_k_indices)
        return answers
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