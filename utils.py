from template_construction import create_rand_demonstrations, concat_demonstations, concat_test2prompt
import random
from collections import Counter
from templates_wizard_lm import DEMONSTRATIONS_DOCS_ANON, TEST_TEMPLATE_DOCS_ANON
from templates_code_llama import DEMONSTRATIONS_USER_ASSISTANT, TEST_TEMPLATE_USER_ASSISTANT, INSTRUCTION_USER_ASSISTANT
import pickle
import numpy as np
import json
import pandas as pd
from pathlib import Path

def tokenize_demonstrations(tokenizer, demonstrations, seed):
    seed_demonstrations = demonstrations[seed]
    dict_tokenized_demonstations = {}

    for id in range(len(seed_demonstrations)):
        ex_key = 'ex'+str(id)
        tmp_demonstration_txt = seed_demonstrations[ex_key]
        dict_tokenized_demonstations[ex_key] = tokenizer(tmp_demonstration_txt)
        # dict_tokenized_demonstations[ex_key]['input_ids'].append(tokenizer.eos_token_id)
        # dict_tokenized_demonstations[ex_key]['attention_mask'].append(1)
        # a=1

    return dict_tokenized_demonstations




def update_results(results_path, avg_recall_vals, avg_mrecall_vals, all_rec_per_template, avg_scores, info):
    # Open store/results.csv or create a new DataFrame if it doesn't exist
    if Path(results_path).exists():
        results_df = pd.read_csv(results_path, index_col=0)
    else:
        results_df = pd.DataFrame()

    info.update(avg_recall_vals)
    info.update(all_rec_per_template)
    info.update(avg_mrecall_vals)
    info.update(avg_scores)
    new_data = pd.DataFrame([info])  # Convert the dictionary to a DataFrame

    # Iterate over columns and round float values to four decimal places
    for col in new_data.columns:
        if pd.api.types.is_numeric_dtype(new_data[col]) and col != 'replace_find':
            new_data[col] = new_data[col].round(4)

    # Update DataFrame with new results
    results_df = pd.concat([results_df, new_data])
    # Save DataFrame to store/results.csv
    results_df.to_csv(results_path)

def concat_tokenized_demonstrations(dict_tokenized_demonstations, selected_demonstrations_ids):
    input_ids = []
    attention_mask = []
    for dem_id in selected_demonstrations_ids:
        ex_key = 'ex'+str(dem_id)
        dict_tmp_demonstration = dict_tokenized_demonstations[ex_key]
        tmp_input_ids = dict_tmp_demonstration['input_ids']
        tmp_attention_mask = dict_tmp_demonstration['attention_mask']
        input_ids.extend(tmp_input_ids)
        attention_mask.extend(tmp_attention_mask)

    return input_ids, attention_mask

def print_args(args):
    # Print the values using a for loop
    print("Argument values:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

def tokenize_data(test_dict_query_ids_queries, tokenizer, dict_tokenized_demonstations, args, instruction):
    all_input_ids =  []
    all_attention_mask = []
    all_qids = []
    for query_id in test_dict_query_ids_queries:
        
        query = test_dict_query_ids_queries[query_id]
        if args.dem_method=='rand':
            selected_demonstrations_ids = create_rand_demonstrations(args.seed, args.num_demonstrations, DEMONSTRATIONS_DOCS_ANON)
        else:
            #  = [ DEMONSTRATIONS_BETTER_DEM['ex2'], DEMONSTRATIONS_BETTER_DEM['ex4'], DEMONSTRATIONS_BETTER_DEM['ex6'], DEMONSTRATIONS_BETTER_DEM['ex3'] ]
            selected_demonstrations_ids = [2,4,6,3]
            random.shuffle(selected_demonstrations_ids)

        #tokenized_instuction = tokenizer('Below is an instruction that describes a task. Write python code that appropriately completes the request. Think step by step')
        tokenized_instuction = tokenizer(instruction)
        input_ids = tokenized_instuction['input_ids']
        attention_mask = tokenized_instuction['attention_mask']

        dem_input_ids, dem_attention_mask = concat_tokenized_demonstrations(dict_tokenized_demonstations, selected_demonstrations_ids)
        input_ids = input_ids+ dem_input_ids 
        attention_mask = attention_mask + dem_attention_mask

        test_example = TEST_TEMPLATE_DOCS_ANON.format(question=query)

        tokenized_test_example = tokenizer(test_example)

        input_ids = input_ids + tokenized_test_example['input_ids']
        attention_mask = attention_mask + tokenized_test_example['attention_mask']

        all_input_ids.append(input_ids)
        all_attention_mask.append(attention_mask)
        all_qids.append(query_id)

    length_counts = Counter(len(inner_list) for inner_list in all_input_ids)

    return all_input_ids, all_attention_mask, all_qids

def selected_demonstrations(selected_demonstrations_ids, demonstrations):
    all_demonstrations = []
    for id in selected_demonstrations_ids:
        ex_key = 'ex'+str(id)
        tmp_demonstration = demonstrations[ex_key]
        for turn in tmp_demonstration:
            if turn['role']=='assistant':
                turn['content'] = turn['content'].strip()
                turn['content'] = '\n' + turn['content']

        all_demonstrations.extend(tmp_demonstration)
    return all_demonstrations

def tokenize_user_assistant(instruction, demonstrations, test_template,test_dict_query_ids_queries, args, tokenizer):
    all_input_ids =  []
    all_attention_mask = []
    all_qids = []
    all_lens = [] # debug purpose
    for query_id in test_dict_query_ids_queries:
        
        query = test_dict_query_ids_queries[query_id]
        if args.dem_method=='rand':
            selected_demonstrations_ids = create_rand_demonstrations(args.seed, args.num_demonstrations, DEMONSTRATIONS_DOCS_ANON)
        else:
            #  = [ DEMONSTRATIONS_BETTER_DEM['ex2'], DEMONSTRATIONS_BETTER_DEM['ex4'], DEMONSTRATIONS_BETTER_DEM['ex6'], DEMONSTRATIONS_BETTER_DEM['ex3'] ]
            selected_demonstrations_ids = [2,4,6,3]
            random.shuffle(selected_demonstrations_ids)

        chat = [instruction]
        formatted_demonstrations = selected_demonstrations(selected_demonstrations_ids, demonstrations[args.seed])
        chat.extend(formatted_demonstrations)
        test_example = TEST_TEMPLATE_USER_ASSISTANT.format(question=query)
        test_query = [{'role':'user','content':test_example}]
        chat.extend(test_query)
        input_ids = tokenizer.apply_chat_template(chat, tokenize=True)

        attention_mask = [1]*len(input_ids)
        all_attention_mask.append(attention_mask)
        
        all_input_ids.append(input_ids)
        all_qids.append(query_id)

        all_lens.append(len(input_ids))

    print(f'mean length {np.mean(all_lens)} max length {np.max(all_lens)}')
    return all_input_ids, all_attention_mask, all_qids

def bytes_to_giga_bytes(bytes):
  return bytes / 1024 / 1024 / 1024

def create_pickle(object, file_name):
    with open(file_name,'wb') as file:
        pickle.dump(object, file)

def load_pickle(file_name):
    with open(file_name,'rb') as file:
        object = pickle.load(file)
    return object

