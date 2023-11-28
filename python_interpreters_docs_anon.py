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
    str = str.replace('and not',' difference ')
    str = str.replace(' and ',' intersect ')
    str = str.replace(' or ',' union ')
    # if 'not' in str:
    #     continue

    list_str = str.split()
    program = ''
    inside_diff= False
    for i, substr in enumerate(list_str):
        if substr=='difference':
            program+= '.difference('
            inside_diff = True

        elif substr=='intersect':
            program += ' & '
        elif substr=='union':
            program += ' | '
        else:
            if inside_diff:
                program += f'{substr})'
                inside_diff = False
            else:
                program += substr

    return program

# Define a function that takes a string and returns the desired function call
def convert2function(text):
    return f"find_docs('{text}')"

def main(args):

    # load docs
    path_doc_text_list = os.path.join(args.data_dir,'doc_text_list.pickle')
    path_doc_title_map = os.path.join(args.data_dir,'doc_title_map.tsv')
    doc_text_map, doc_title_map = read_docs(path_doc_text_list, path_doc_title_map)


    file=open(args.result_dir, "r")
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
    # lens_per_template = {}
    for query in results_json:
        if not 'pred' in results_json[query]:
            print('forgot sth')
            count_forgot +=1
        
        result = results_json[query]['pred']
        replacement_query = query.replace("'", "\\'")  # Escape single quotes

        new_result = result.replace("find_docs(",f"DocumentFinder.find_docs('{replacement_query}',")
        program = synthesize_program(result = new_result, prefix = '')
        if 'ans = ' not in program:
            count_not_ans +=1
            unsorted_pred_doc_titles = []
        else:
            answer_index = program.index('ans = ')
            answer_code = program[answer_index:]

            
            answer_code_with_sets = set_conversion(answer_code)
            final_program = program[:answer_index] + answer_code_with_sets+"\n"
            # list_ops = find_logical_operators(answer_code)
            # label = compare_logical_operators(list_ops, results_json[query]['template'])
            # true_ops += label
            # count_ops +=1

            var_dict = safe_execute(final_program)
            if var_dict is not None:
                # questions = [var_dict[var] for var in var_dict if 'answer' not in var and 'question' not in var and 'instruction' not in var and var != 'x' and isinstance(var_dict[var], str)]
                unsorted_doc_ids = var_dict['ans']
                unsorted_pred_doc_titles = [doc_title_map[id] for id in unsorted_doc_ids]
                results_len.append(len(unsorted_pred_doc_titles))
            else:
                questions = [query]
                count_bug +=1
                unsorted_pred_doc_titles = []
            
        tmp_pred_example = Example(query=query, docs=unsorted_pred_doc_titles)
        pred_examples.append(tmp_pred_example)

    path_test = os.path.join(args.data_dir,'test.jsonl')
    gold_examples = example_utils.read_examples(path_test)

    calc_f1_pr_rec(gold_examples, pred_examples)

    with open('res_docs_anon_1000.pkl', 'wb') as file:
        pickle.dump(DocumentFinder.results, file)

        # results_json[query]['sub_questions'] = questions
        # all_subquestions.extend(questions)
    print(f'count bug {count_bug}')
    print(f'count_forgot {count_forgot}')
    print(f'Generated answer set length: mean {np.mean(results_len)} std {np.std(results_len)}')
    a=1
        # safe_execute(program)
    
    # Serializing json
    # json_object = json.dumps(results_json, indent=4)
    
    # # Writing to sample.json
    # with open(args.result_dir, "w") as outfile:
    #     outfile.write(json_object)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='GPT API')
    parser.add_argument('--result_dir', type=str, default="docs_anon.json")
    parser.add_argument('--data_dir', type=str, default="quest_data")
    args = parser.parse_args()
    main(args)