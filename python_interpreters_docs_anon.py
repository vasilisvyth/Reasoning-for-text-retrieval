import json
import argparse
import re
import numpy as np
from templates.templates import template2logic
import os
from data.prepare_dataset import read_docs, read_queries
from run_eval import calc_f1_pr_rec
from quest.common import example_utils
from quest.common.example_utils import Example, ExampleMetadata
import pickle
import numpy as np
import torch
from tools import Operations, safe_execute, synthesize_program, VP_BM25, VP_HuggingFace, VP_SentenceTransformer
from analyze_retriever import calc_mrec_rec
from quest.common import document_utils
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
from bi_encoder import DenseBiEncoder
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from utils import load_pickle
from seeds import set_seed
from utils import create_pickle, update_results
import torch.nn.functional as F
from example_with_subquestions import ExampleWithSubQuestions

def find_all_positions(text, substring):
    positions = []
    start_pos = text.find(substring)

    while start_pos != -1:
        positions.append(start_pos)
        start_pos = text.find(substring, start_pos + 1)

    return positions


def find_logical_operators(program):

    # looking for one or more whitespace characters, followed by either "or" or "and" or 'not' (case-sensitive), followed by one or more whitespace characters
    # I also want to look for 'not '  as well
    logical_operators = [' or ',' and not ', ' and ']
    ops_in_program = []
    pos_ops = []
    for op in logical_operators:
        # find all positions of current logical operator in the string
        positions = find_all_positions(program, op)
        if ' and ' ==op:
            for pos in positions:
                a=1

        pos_ops.extend(positions)
        ops_in_program.extend([op]*len(positions))

    indices = np.argsort(pos_ops)
    sorted_ops_in_program = [ops_in_program[i] for i in indices]
    return sorted_ops_in_program

def find_sets_logical_operators(program):

    # looking for one or more whitespace characters, followed by either "or" or "and" or 'not' (case-sensitive), followed by one or more whitespace characters
    # I also want to look for 'not '  as well
    logical_operators = [' & ',' | ', ' difference( ']
    ops_in_program = []
    pos_ops = []
    for op in logical_operators:
        # find all positions of current logical operator in the string
        positions = find_all_positions(program, op)
   
        pos_ops.extend(positions)
        ops_in_program.extend([op]*len(positions))

    indices = np.argsort(pos_ops)
    sorted_ops_in_program = [ops_in_program[i] for i in indices]
    return sorted_ops_in_program



def compare_logical_operators(generated_logical_ops, true_template):
    '''
    generated_logical_ops: list of strings any from 
    '''
    true_template_as_list = template2logic[true_template]
    label = true_template_as_list == generated_logical_ops
    return label

def find_all_positions(text, substring):
    positions = []
    start_pos = text.find(substring)

    while start_pos != -1:
        positions.append(start_pos)
        start_pos = text.find(substring, start_pos + 1)

    return positions

# I have strings like the following and I want to be able to put parentheses around the difference symbol. Complete the code

def set_conversion(str):
    str = str.replace('and not','-')
    str = str.replace(' and ',' & ')
    str = str.replace(' or ',' | ')

    return str

# Define a function that takes a string and returns the desired function call
def convert2function(text):
    return f"find_docs('{text}')"

def print_args(args):
    # Print the values using a for loop
    print("Argument values:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

def find_parentheses_positions(expression):
    stack = []
    positions = []

    for i, char in enumerate(expression):
        if char == '(':
            stack.append(i)
        elif char == ')':
            if stack:
                start = stack.pop()
                end = i
                positions.append((start, end))
            else:
                # Handle the case where there is a closing parenthesis without a matching opening parenthesis
                print(f"Error: Unmatched closing parenthesis at position {i}")

    if stack:
        # Handle the case where there are unmatched opening parentheses
        for start in stack:
            print(f"Error: Unmatched opening parenthesis at position {start}")

    return positions

def calculate_score(list_answer, var_dict, not_subqs, query, ind_scores):
    not_docs = [list_answer[ii+1] for ii, el in enumerate(list_answer) if el=='not']
                # ind_scores = {}
    for not_doc in not_docs: # as many times we observe not in the output
        # var_dict[not_doc]
        try:
            num = int(not_doc.split('_')[1]) #doc_num
        except IndexError:
            print("Index out of range!")
            continue

        subq = var_dict[f'question_{num}'] # find question_i for this doc_i variable
        not_subqs.append(subq)

        # subq = subq.replace('find ','')
        dict_subq = DocumentFinder.results[query][subq]
        for ind, score in zip(dict_subq['top_k_indices'],dict_subq['top_k_values']):
            max_score = dict_subq['top_k_values'][0]
            if ind not in ind_scores:
                ind_scores[ind] = - score
            else:
                ind_scores[ind] -= - score
    
    
    for subq in DocumentFinder.results[query]:
        # subq = subq.replace('find ','')
        if subq in not_subqs: #this subquestion regards not
            continue
        dict_subq = DocumentFinder.results[query][subq]
        for ind, score in zip(dict_subq['top_k_indices'],dict_subq['top_k_values']):
            if ind not in ind_scores:
                ind_scores[ind] = score
            else:
                ind_scores[ind] += score
    
    sorted_keys = sorted(ind_scores, key=lambda x: ind_scores[x], reverse=True)[:1000]
    if '(' in answer_code:
        answer_code = answer_code.replace('(','( ')
        count_parenthesis+=1
        print(answer_code_with_sets)

    unsorted_doc_ids = var_dict['ans'] #! previously I just used this
    unsorted_pred_doc_titles = [doc_title_map[id] for id in unsorted_doc_ids]
    unsorted_pred_doc_titles = [documents[idx].title for idx in unsorted_doc_ids]
    results_len.append(len(unsorted_pred_doc_titles))


def current_program(program):
    '''
    the last 3 arguments are added for debugging purpose
    '''
  
    answer_index = program.index('ans = ')
    answer_code = program[answer_index:]

    # if results_json[query]['template'] != '_ that are also _':
    #     continue
    answer_code_with_sets = set_conversion(answer_code)
    final_program = program[:answer_index] + answer_code_with_sets+"\n"

    var_dict = safe_execute(final_program)

    return var_dict

def preprocess_find_docs_call(query, result, class_name):
    replacement_query = query.replace("'", "\\'")  # Escape single quotes
    new_result = result.replace("find_docs(",f"{class_name}.find_docs('{replacement_query}',")
    return new_result

def create_results(new_gold_examples, pred_examples, count_bug, INFO):
    avg_prec, avg_rec, avg_f1 = calc_f1_pr_rec(new_gold_examples, pred_examples)

    avg_scores = {'avg_prec':avg_prec['all'], 'avg_rec':avg_rec['all'], 'avg_f1':avg_f1['all']}
    avg_recall_vals, avg_mrecall_vals, all_rec_per_template = calc_mrec_rec(new_gold_examples, pred_examples)

    avg_recall_vals = {f'avg_R@{key}':value for key, value in avg_recall_vals.items()}
    avg_mrecall_vals = {f'avg_MR@{key}':value for key, value in avg_mrecall_vals.items()}

    print(f'count bugs {count_bug}')
    path_results_csv = os.path.join('checkpoints','results.csv')
    update_results(path_results_csv, avg_recall_vals, avg_mrecall_vals, all_rec_per_template, avg_scores, INFO)

    create_pickle(avg_recall_vals,'baseline_avg_rec.pickle')
    create_pickle(avg_mrecall_vals,'baseline_avg_mrec.pickle')

def plot_lens(lens_per_template):
    means = {key: np.mean(values) for key, values in lens_per_template.items()}
    std_devs = {key: np.std(values) for key, values in lens_per_template.items()}

    # Create a bar plot
    fig, ax = plt.subplots()

    # Plotting the means
    ax.bar(means.keys(), means.values(), yerr=std_devs.values(), capsize=5, color='skyblue', label='Mean')

    # Adding labels and title
    ax.set_ylabel('Retrieved docs per template')
    ax.set_xlabel('Templates')
    ax.set_title('Mean and Standard Deviation for Each Key')
    ax.legend()

    # Show the plot
    plt.show()

def append_subq_examples(var_dict, docs_var, documents, example, question_id, new_gold_examples, new_pred_examples):
    subq_dids = list(var_dict[docs_var].data.keys())[:1000]
    subq_docs = [documents[idx].title for idx in subq_dids]
    # put as gold question the generated subquestion to avoid duplicates and as docs the ones of the original question
    gold_subq_example = Example(query=var_dict[f'question_{question_id}'], docs=example.docs, metadata=ExampleMetadata(template=example.metadata.template))
    new_gold_examples.append(gold_subq_example)

    subq_example = Example(query=var_dict[f'question_{question_id}'], docs=subq_docs, metadata=ExampleMetadata(template=example.metadata.template))
    new_pred_examples.append(subq_example)

def process_template_example(example, var_dict, documents, new_gold_examples, new_pred_examples):
    if (example.metadata.template=='_ that are also _' or example.metadata.template=='_ that are also both _ and _'):
        for key in var_dict:
            if key.startswith('docs_'):
                question_id = int(key[key.index('_')+1:])
                append_subq_examples(var_dict, key, documents, example, question_id, new_gold_examples, new_pred_examples)
                

    if (example.metadata.template == '_ that are not _'):

        append_subq_examples(var_dict, 'docs_0', documents, example, 0, new_gold_examples, new_pred_examples)

    if (example.metadata.template == '_ that are also _ but not _'):
        append_subq_examples(var_dict, 'docs_0', documents, example, 0, new_gold_examples, new_pred_examples)
        append_subq_examples(var_dict, 'docs_1', documents, example, 1, new_gold_examples, new_pred_examples)

def extract_subquestions(var_dict):
    questions = []
    for key in var_dict:
        if key.startswith('docs_'):
            question_id = int(key[key.index('_')+1:])
            question = var_dict[f'question_{question_id}']
            questions.append(question)
    
    return questions

def main(args):
    set_seed(0)
    print_args(args)

    path_doc_text_list = os.path.join(args.data_dir,'doc_text_list.pickle')
    path_doc_title_map = os.path.join(args.data_dir,'doc_title_map.tsv')
    doc_text_map, doc_title_map = read_docs(path_doc_text_list, path_doc_title_map)

    path_doc = os.path.join('quest_data','documents.jsonl')
    documents = document_utils.read_documents(path_doc)
    title_doc_map = {value: key for key, value in doc_title_map.items()} #! duplicates doc titles so smaller length

    path_test = os.path.join(args.data_dir,args.gold_examples_dir)
    gold_examples = example_utils.read_examples(path_test)
    args.oracle_docs = False#True # when we replace 
    if args.oracle_docs:
        domains = set() # 
        counter = 0
        count_new_docs = 0
        oracle_examples_dict = {}
        new_gold_examples = []
        for example in gold_examples:
            attributions_dict = example.metadata.attributions
            if attributions_dict is None:
                counter +=1
                domains.add(example.metadata.domain)
                continue
            assert(len(example.docs) == len(attributions_dict))

            new_gold_examples.append(example)
            oracle_examples_dict[example.query] = {}
            a=1
            for doc_title in attributions_dict: #dict
                doc_id = title_doc_map[doc_title]
                original_doc_text = doc_text_map[doc_id] 

                query_doc_list = [el for el in attributions_dict[doc_title] if el != None]
                new_doc_text = '' # sometimes duplicate doc substring  
                for i in range(len(query_doc_list)):
                    for subquery in query_doc_list[i]: #dict
                        doc_substring = query_doc_list[i][subquery]
                        new_doc_text += doc_substring

                if new_doc_text != '':
                    count_new_docs +=1
                    doc_id = title_doc_map[doc_title]

                    oracle_examples_dict[example.query][doc_id]  = new_doc_text
        gold_examples = new_gold_examples # we only care about examples with attributions
    else:
        oracle_examples_dict = None

    
    RANK_CONSTANT = 60
    replace_find = True
    SCORE = f'1/({RANK_CONSTANT}+rank' # 'emb' #
    THRESHOLD = 0.64#args.threshold#0.64 #'None'

    METHOD = 'bge-large'
    
    normalize_embeddings=False
    doc_embs = None
    tokenizer = None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu') # use this for mistral to 
    K = 30000
    if METHOD =='bge-large':
        doc_embs = np.load(os.path.join('checkpoints','bge-zero','zero_shot_t5_bge_large.npy'))
        doc_embs = torch.from_numpy(doc_embs)
        CHECKPOINT = 'BAAI/bge-large-en-v1.5'
        normalize_embeddings=True
        use_cash = False
        results = os.path.join(args.files_dir,'vp_{METHOD}_results.pickle') if use_cash else {}
        VP_SentenceTransformer.init(device, CHECKPOINT, doc_embs, K, METHOD, replace_find, SCORE, threshold=THRESHOLD,  normalize_embeddings=False, 
            oracle_examples_dict = oracle_examples_dict, use_cache=use_cash)
        class_name = 'VP_SentenceTransformer'
        assert(len(doc_embs) == len(documents)== len(doc_title_map) == 325505)
    elif METHOD == 't5-base':
        doc_embs = torch.load('checkpoints/4913e0dd-b8/scores_docs_check_13500.pt')
        CHECKPOINT = 'checkpoints/4913e0dd-b8/checkpoint-13500/'
        tokenizer = AutoTokenizer.from_pretrained('google/t5-v1_1-base')

        VP_HuggingFace.init(device, CHECKPOINT, doc_embs, tokenizer, K, METHOD, replace_find, SCORE, threshold='none', normalize_embeddings=normalize_embeddings, 
            oracle_examples_dict=oracle_examples_dict)
        class_name = 'VP_HuggingFace'
    elif METHOD == 'mistral':
        use_cash = True # use existing already calculated embeddings
        if use_cash: device = torch.device('cpu')
        
        doc_embs = np.load(os.path.join('checkpoints','mistral_instruct','rand_zero_shot_mistral.npy'))
        doc_embs = torch.from_numpy(doc_embs)
        doc_embs = F.normalize(doc_embs, p=2, dim=1) 
        doc_embs = doc_embs.to(device)

        # just avoid mistral locally 
        

        CHECKPOINT = 't5-small' if use_cash else 'intfloat/e5-mistral-7b-instruct'
        tokenizer = 'intfloat/e5-mistral-7b-instruct'
        METHOD = 'mistral'
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        class_name = 'VP_HuggingFace'
        K = len(doc_embs)
        
        
        results = load_pickle('vp_mistral_rand_results.pickle') if use_cash else {}
        
        VP_HuggingFace.init(device, CHECKPOINT, doc_embs, tokenizer, K, METHOD, replace_find, SCORE, threshold=THRESHOLD, normalize_embeddings=normalize_embeddings, 
            oracle_examples_dict=oracle_examples_dict, results=results, use_cache=use_cash)

        # the following code is only useful when we work with rand qids and rand dids
        rand_dids_path = os.path.join('files','with_true_bge_large__0_rand_ids_top_25.pickle')
        dids = load_pickle(rand_dids_path)
       

        test_dict_query_ids_queries, test_query_ids_doc_ids = read_queries(os.path.join(args.data_dir,'test_query_ids_queries.tsv'), 
                                                 os.path.join(args.data_dir,'test_query_ids_doc_ids.tsv'))
        rand_qids_path = os.path.join('files','rand_qids.pickle')
        rand_qids = load_pickle(rand_qids_path)

        new_gold_examples = []
        for qid in rand_qids:
            query = test_dict_query_ids_queries[qid]
            for ex in gold_examples:
                if ex.query == query:
                    new_gold_examples.append(ex)
                    break

        gold_examples = new_gold_examples

    elif METHOD=='bm25':
        CHECKPOINT = 'dum_bm25_obj.pickle'
        VP_BM25.init(CHECKPOINT, doc_embs, K, replace_find, SCORE, threshold='None',
            oracle_examples_dict=oracle_examples_dict)
        class_name = 'VP_BM25'
    else:
        raise(f'{METHOD} is not supported')
        


    args.gpt_results_dir = 'docs_anon.json'

    data = {'and':'avg','or':'maximum','diff':'subtract','score':SCORE}
    INFO = { 'info':'','model':METHOD,'checkpoint':CHECKPOINT,
             'K':K,'threshold':THRESHOLD,'replace_find':replace_find,'template':args.gpt_results_dir
    }
    INFO.update(data)

    Operations.init(data, RANK_CONSTANT)
    
    file=open(args.gpt_results_dir, "r")
    results_json = json.load(file)
    

    count_bug = 0
    count_forgot = 0
    pred_examples = []

    new_pred_examples = [ ]
    new_gold_examples = []
    lens_per_template = defaultdict(list)
    examples_with_subquestions = []
    i = -1
    args.seperate_subquestions = False
    print(f'Calculate seperate subquestions metrics? {args.seperate_subquestions}')
    for j, ex in enumerate(tqdm(gold_examples)):
        query = ex.query
        if not 'pred' in results_json[query]:
            # print('forgot sth')
            count_forgot +=1
            continue
        
        template_used = results_json[query]['template']

        result = results_json[query]['pred']

        new_result = preprocess_find_docs_call(query, result, class_name)
        # new_result = new_result.replace('intersection(docs_2)','intersection(docs_2,docs_2)')
        program = synthesize_program(result = new_result, prefix = '')
        # if not ' and ' in program:
        #     continue

        if 'ans = ' not in program:
            # count_not_ans +=1
            sorted_pred_doc_titles = []
        else:
            
            var_dict = current_program(program)

            if var_dict is not None:
                try:
                    i+=1
                    ans = var_dict['ans']
                    sorted_items = sorted(ans.items(), key=lambda x: x[1], reverse=True)

                    # Take the top k items
                    initial_top_k_keys = [key for key, _ in sorted_items]
                    top_k_keys = initial_top_k_keys[:1000]
              
                    if METHOD == 'mistral':
                        # create_pickle(top_k_keys,'debug_top_keys.pickle')
                        pred_docs_ids = [dids[index] for index in top_k_keys]
                        sorted_pred_doc_titles = [doc_title_map[pr_id] for pr_id in pred_docs_ids]
                    else:
                        sorted_pred_doc_titles = [documents[idx].title for idx in top_k_keys]

                    if args.seperate_subquestions:
                        process_template_example(ex, var_dict, documents, new_gold_examples, new_pred_examples)

                    subquestions = extract_subquestions(var_dict)
                    current_example = ExampleWithSubQuestions(example =ex, subquestions = subquestions)
                    examples_with_subquestions.append(current_example)
                    # gold_ids = [title_doc_map[tmp_doc] for tmp_doc in new_gold_examples[i].docs]

                    # gold_texts = [doc_text_map[id] for id in gold_ids]
                    # predicted_texts = [doc_text_map[id] for id in top_k_keys]
                    lens_per_template[template_used].append(len(top_k_keys))

                except Exception as e:
                    # Handle the case where ans is not a dictionary
                    import traceback
                    traceback.print_exc()
                    # print(final_program)
                    sorted_pred_doc_titles = []

            else:
                # count_bug +=1
                sorted_pred_doc_titles = []

        if sorted_pred_doc_titles == []:
            count_bug +=1
        tmp_pred_example = Example(query=query, docs=sorted_pred_doc_titles)
        pred_examples.append(tmp_pred_example)

    print(f'num of examples {len(pred_examples)}')
    print(f'count bug {count_bug}')

    create_pickle(examples_with_subquestions,os.path.join(args.files_dir,f'ex_with_subq_{METHOD}_rand_{args.gpt_results_dir}.pickle'))

    if METHOD == 'mistral': 
        create_pickle(VP_HuggingFace.results,os.path.join(args.files_dir,f'vp_{METHOD}_rand_{args.gpt_results_dir}.pickle'))
    elif METHOD == 'bge-large':
        create_pickle(VP_SentenceTransformer.results,os.path.join(args.files_dir,f'vp_{METHOD}_{args.gpt_results_dir}.pickle'))
    if args.seperate_subquestions:
        print(f'len {len(new_gold_examples)}')
        create_results(new_gold_examples, new_pred_examples, count_bug, INFO)
    else:
        create_results(gold_examples, pred_examples, count_bug, INFO)
    plot_lens(lens_per_template)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='GPT API')
    parser.add_argument('--gpt_results_dir', type=str, default="docs_anon.json")
    parser.add_argument('--data_dir', type=str, default="quest_data")
    parser.add_argument('--result_dir', type=str, default="res_docs_anon_1000.json")
    parser.add_argument('--gold_examples_dir', type=str, default='test.jsonl')
    parser.add_argument('--oracle_docs',action='store_true') # default false now!
    parser.add_argument('--seperate_subquestions',action='store_true')
    parser.add_argument('--files_dir',type=str,default='files')
    # parser.add_argument('--gold_examples_dir', type=str, default='test.jsonl')
    parser.add_argument('--k', type=int, default=1000)
    parser.add_argument('--threshold', type=float, default=0.64)
    args = parser.parse_args()
    main(args)