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
from tools import DocumentFinder
from analyze_retriever import calc_mrec_rec
from quest.common import document_utils
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt

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

def main(args):
    print_args(args)

    # load docs
    path_doc_text_list = os.path.join(args.data_dir,'doc_text_list.pickle')
    path_doc_title_map = os.path.join(args.data_dir,'doc_title_map.tsv')
    doc_text_map, doc_title_map = read_docs(path_doc_text_list, path_doc_title_map)
    documents = document_utils.read_documents("quest_data\\documents.jsonl")

    # tensors = {}
    # from safetensors import safe_open
    # with safe_open('checkpoints/89a4eb2b-b3/checkpoint-6000/model.safetensors', framework="pt", device="cpu") as f:
    #     for key in f.keys():
    #         tensors[key] = f.get_tensor(key)


    path_test = os.path.join(args.data_dir,args.gold_examples_dir)
    gold_examples = example_utils.read_examples(path_test)

    DocumentFinder.k= 3000#args.k
    DocumentFinder.method = 'bm25'
    # pred_examples = []
    # for exam in gold_examples:
    #     current_docs_ids = DocumentFinder.find_docs(exam.query, exam.query)
    #     unsorted_pred_doc_titles = [doc_title_map[id] for id in current_docs_ids]

    #     tmp_pred_example = Example(query=exam.query, docs=unsorted_pred_doc_titles)
    #     pred_examples.append(tmp_pred_example)

    # print('ORIGINAL QUERIES RESULTS')
    # calc_mrec_rec(gold_examples, pred_examples)
    # calc_f1_pr_rec(gold_examples, pred_examples)
    # print("Visual programming")
    # DocumentFinder.results = {}

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
    count_parenthesis = 0
    lens_per_template = defaultdict(list)
    
    for query in tqdm(results_json):
        if not 'pred' in results_json[query]:
            print('forgot sth')
            count_forgot +=1

        template_used = results_json[query]['template']
        result = results_json[query]['pred']
        replacement_query = query.replace("'", "\\'")  # Escape single quotes

        new_result = result.replace("find_docs(",f"DocumentFinder.find_docs('{replacement_query}',")
        program = synthesize_program(result = new_result, prefix = '')
        if 'ans = ' not in program:
            count_not_ans +=1
            sorted_pred_doc_titles = []
        else:
            answer_index = program.index('ans = ')
            answer_code = program[answer_index:]


            # len_par = find_all_positions(answer_code, '(')
            # if len(len_par) > 1:
            #     # print('count open par more than 1')
            #     print(answer_code)

            answer_code_with_sets = set_conversion(answer_code)
            final_program = program[:answer_index] + answer_code_with_sets+"\n"
            find_sets_logical_operators(answer_code)

            var_dict = safe_execute(final_program)

            if var_dict is not None:
                try:
                    ans = var_dict['ans']
                    sorted_items = sorted(ans.items(), key=lambda x: x[1], reverse=True)
                     # Take the top k items
                    top_k_keys = [key for key, _ in sorted_items[:1000]]
                    sorted_pred_doc_titles = [documents[idx].title for idx in top_k_keys]

                    lens_per_template[template_used].append(len(top_k_keys))
                except AttributeError:
                    # Handle the case where ans is not a dictionary
                    print(f'original query {query}')
                    print(f"Error: {ans} is not a dictionary")
                    print(final_program)
                    sorted_pred_doc_titles = []

            else:
                questions = [query]
                count_bug +=1
                sorted_pred_doc_titles = []
            
        tmp_pred_example = Example(query=query, docs=sorted_pred_doc_titles)
        pred_examples.append(tmp_pred_example)

    calc_f1_pr_rec(gold_examples, pred_examples)
    calc_mrec_rec(gold_examples, pred_examples)
    print(f'count bugs {count_bug}')

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
    parser.add_argument('--k', type=int, default=1000)
    args = parser.parse_args()
    main(args)