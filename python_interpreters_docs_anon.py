from tools import safe_execute, synthesize_program, DocumentFinder
import json
import argparse
import re
import numpy as np
from templates import template2logic
import os
from prepare_dataset import read_docs
from run_eval import calc_f1_pr_rec
from quest.common import example_utils
from quest.common.example_utils import Example
import pickle
import numpy as np
import torch
from tools import DocumentFinder, Operations
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
    # if 'not' in str:
    #     continue

    # list_str = str.split()
    # program = ''
    # inside_diff= False
    # for i, substr in enumerate(list_str):
    #     if substr=='difference':
    #         program+= '.difference('
    #         inside_diff = True

    #     elif substr=='intersect':
    #         program += ' & '
    #     elif substr=='union':
    #         program += ' | '
    #     else:
    #         if inside_diff:
    #             program += f'{substr})'
    #             inside_diff = False
    #         else:
    #             program += substr

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

def preprocess_find_docs_call(query, result):
    replacement_query = query.replace("'", "\\'")  # Escape single quotes
    new_result = result.replace("find_docs(",f"DocumentFinder.find_docs('{replacement_query}',")
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



def main(args):
    set_seed(0)
    print_args(args)

    # with open('checkpoints/4913e0dd-b8/all_dids_check_13500.pkl','rb') as f:
    #     loaded_object = pickle.load(f)
    #     assert(loaded_object == [i for i in range(len(loaded_object))])
    tokenizer = AutoTokenizer.from_pretrained('google/t5-v1_1-base')
    # load docs
    path_doc_text_list = os.path.join(args.data_dir,'doc_text_list.pickle')
    path_doc_title_map = os.path.join(args.data_dir,'doc_title_map.tsv')
    doc_text_map, doc_title_map = read_docs(path_doc_text_list, path_doc_title_map)

    path_doc = os.path.join('quest_data','documents.jsonl')
    documents = document_utils.read_documents(path_doc)
    title_doc_map = {value: key for key, value in doc_title_map.items()}

    path_test = os.path.join(args.data_dir,args.gold_examples_dir)
    gold_examples = example_utils.read_examples(path_test)
    args.oracle_docs = True#True
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
        gold_examples = new_gold_examples
    else:
        oracle_examples_dict = None
    a=1         

    # model = DenseBiEncoder('checkpoints/da0656a7-f3/checkpoint-13500', False, False)
    # docs = torch.load('checkpoints/da0656a7-f3/scores_docs_check_0.pt')
    # tokenized_txt = tokenizer.encode_plus(doc_text_map[0], return_tensors='pt')
    # input_ids, attention_mask = tokenized_txt['input_ids'], tokenized_txt['attention_mask']
    # with torch.no_grad():
    #     embed_doc = model.encode(input_ids, attention_mask)
    # all_dids = load_pickle('checkpoints/da0656a7-f3/all_dids_check_0.pkl')
    # pretrained = DenseBiEncoder('google/t5-v1_1-base', False, False)
    # pretrained_embed_doc = pretrained.encode(input_ids, attention_mask)
    METHOD = 'bge-large'
    use_sentence_transformer=False
    normalize_embeddings=False
    doc_embs = None
    if METHOD =='bge-large':
        doc_embs = np.load(os.path.join('checkpoints','bge-zero','zero_shot_t5_bge_large.npy'))
        doc_embs = torch.from_numpy(doc_embs)
        CHECKPOINT = 'BAAI/bge-large-en-v1.5'
        use_sentence_transformer=True
        normalize_embeddings=True
        #DocumentFinder.init('BAAI/bge-large-en-v1.5', doc_embs, tokenizer, k=30000, method='bge-large', use_sentence_transformer=True, normalize_embeddings=True)
    elif METHOD == 't5-base':
        doc_embs = torch.load('checkpoints/4913e0dd-b8/scores_docs_check_13500.pt')
        CHECKPOINT = 'checkpoints/4913e0dd-b8/checkpoint-13500/'
    else:
        CHECKPOINT = 'dum_bm25_obj.pickle'

    args.gpt_results_dir = 'docs_anon.json'

    RANK_CONSTANT = 60
    K = 30000
    replace_find = True
    SCORE = f'1/({RANK_CONSTANT}+rank' # 'emb' #
    THRESHOLD = 0.64 #'None'
    DocumentFinder.init(CHECKPOINT, doc_embs, tokenizer, k=K, method=METHOD, replace_find = replace_find, use_sentence_transformer=use_sentence_transformer,
                         normalize_embeddings=normalize_embeddings, score=SCORE, threshold=THRESHOLD,oracle_examples_dict=oracle_examples_dict)

    data = {'and':'avg','or':'maximum','diff':'subtract','score':SCORE}
    INFO = { 'model':METHOD,'checkpoint':CHECKPOINT,
             'K':K,'threshold':THRESHOLD,'replace_find':replace_find,'template':args.gpt_results_dir
    }
    INFO.update(data)

    Operations.init(data, RANK_CONSTANT)
    
    file=open(args.gpt_results_dir, "r")
    results_json = json.load(file)
    
    true_ops = 0
    count_ops = 0
    all_subquestions = [] # debug_purpose
    count_bug = 0
    count_not_ans = 0
    count_find_bugs = 0
    count_forgot = 0
    pred_examples = []
    results_len = []
    new_gold_examples = []
    count_parenthesis = 0
    lens_per_template = defaultdict(list)
    i = -1
    # for qid, query in enumerate(tqdm(results_json)):

    for j, ex in enumerate(tqdm(gold_examples)):
        query = ex.query
        if not 'pred' in results_json[query]:
            # print('forgot sth')
            count_forgot +=1
            continue
        
        template_used = results_json[query]['template']

        result = results_json[query]['pred']

        new_result = preprocess_find_docs_call(query, result)
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
                    sorted_pred_doc_titles = [documents[idx].title for idx in top_k_keys]
                    # dict_ranks = {}
                    # for key in var_dict:
                    #     if key.startswith('docs_'):
                    #         dict_ranks[key] = list(var_dict[key].data.keys())
                    #         new_gold_examples.append(curr_ex)

                    # gold_ids = [title_doc_map[tmp_doc] for tmp_doc in new_gold_examples[i].docs]

                    # gold_texts = [doc_text_map[id] for id in gold_ids]
                    # predicted_texts = [doc_text_map[id] for id in top_k_keys]
                    lens_per_template[template_used].append(len(top_k_keys))

                except AttributeError:
                    # Handle the case where ans is not a dictionary
                    print(f'original query {query}')
                    print(f"Error: {ans} is not a dictionary")
                    # print(final_program)
                    sorted_pred_doc_titles = []

            else:
                # count_bug +=1
                sorted_pred_doc_titles = []

        if sorted_pred_doc_titles == []:
            count_bug +=1
        tmp_pred_example = Example(query=query, docs=sorted_pred_doc_titles)
        pred_examples.append(tmp_pred_example)

    
    
    
    create_results(gold_examples, pred_examples, count_bug, INFO)
    plot_lens(lens_per_template)


    # with open('res_docs_anon_1000.pkl', 'wb') as file:
    #     pickle.dump(DocumentFinder.results, file)

        # results_json[query]['sub_questions'] = questions
        # all_subquestions.extend(questions)
    # print(f'count bug {count_bug}')
    # print(f'count_forgot {count_forgot}')
    # print(f'Generated answer set length: mean {np.mean(results_len)} std {np.std(results_len)}')
    a=1
        # safe_execute(program)
    
    # Serializing json
    # json_object = json.dumps(results_json, indent=4)
    
    # # Writing to sample.json
    # with open(args.result_dir, "w") as outfile:
    #     outfile.write(json_object)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='GPT API')
    parser.add_argument('--gpt_results_dir', type=str, default="docs_anon.json")
    parser.add_argument('--data_dir', type=str, default="quest_data")
    parser.add_argument('--result_dir', type=str, default="res_docs_anon_1000.json")
    parser.add_argument('--gold_examples_dir', type=str, default='test.jsonl')
    parser.add_argument('--oracle_docs',action='store_true') # default false now!
    # parser.add_argument('--gold_examples_dir', type=str, default='test.jsonl')
    parser.add_argument('--k', type=int, default=1000)
    args = parser.parse_args()
    main(args)