from tools import safe_execute, synthesize_program
import json
import argparse
import re
import numpy as np
from templates import template2logic

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
    logical_operators = [' or ',' and ','not ']
    ops_in_program = []
    pos_ops = []
    for op in logical_operators:
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

def main(args):
    file=open(args.result_dir, "r")
    results_json = json.load(file)
    
    true_ops = 0
    count_ops = 0
    for query in results_json:
        if not 'pred' in results_json[query]:
            continue
        result = results_json[query]['pred']
    
        program = synthesize_program(result = result, prefix = '')
        answer_index = program.index('answer = ')
        answer_code = program[answer_index:]

        list_ops = find_logical_operators(answer_code)
        label = compare_logical_operators(list_ops, results_json[query]['template'])
        true_ops += label
        count_ops +=1

        var_dict = safe_execute(program)
        questions = [var_dict[var] for var in var_dict if 'answer' not in var and 'question' not in var and 'instruction' not in var and var != 'x']
        results_json[query]['sub_questions'] = questions
        # safe_execute(program)
    a=1

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='GPT API')
    parser.add_argument('--result_dir', type=str, default="how_about_python.json")
    args = parser.parse_args()
    main(args)