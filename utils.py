from templates.template_construction import create_rand_demonstrations, concat_demonstations, concat_test2prompt
import random
from collections import Counter
from templates.templates_wizard_lm import DEMONSTRATIONS_DOCS_ANON, TEST_TEMPLATE_DOCS_ANON
from templates.templates_code_llama import DEMONSTRATIONS_USER_ASSISTANT, TEST_TEMPLATE_USER_ASSISTANT, INSTRUCTION_USER_ASSISTANT
import pickle
import numpy as np
import json
import pandas as pd
from pathlib import Path
import json

def tokenize_demonstrations(tokenizer, demonstrations, seed):
    seed_demonstrations = demonstrations[seed]
    dict_tokenized_demonstations = {}

    for id in range(len(seed_demonstrations)):
        ex_key = 'ex'+str(id)
        tmp_demonstration_txt = seed_demonstrations[ex_key]
        dict_tokenized_demonstations[ex_key] = tokenizer(tmp_demonstration_txt)
        # dict_tokenized_demonstations[ex_key]['input_ids'].append(tokenizer.eos_token_id)
        # dict_tokenized_demonstations[ex_key]['attention_mask'].append(1)
        

    return dict_tokenized_demonstations



def json_list_to_txt(json_list, instruction, txt_file_path):
    def write_chunk(chunk, file):
        for line in chunk.splitlines():
            file.write(line)
            file.write('\n')
    STEP = 19
    with open(txt_file_path, 'w') as txt_file:
        txt_file.write('Prompt: \n'+instruction+'\n')
        for json_obj in json_list:
            txt_file.write('-'*150+'\n')
            for key, value in json_obj[0].items():
                txt_file.write(f'"{key}": ')
                if isinstance(value, dict) and len(value['text']) > 100:  # Check if value is a huge string
 
                    chunks = value['text'].split(' ')  # Split the string into multiple lines
                    lines = [' '.join(chunks[i:i+STEP]) for i in range(0, len(chunks), STEP)]
                    for line in lines:
                        txt_file.write(line + '\n')
                else:
                    txt_file.write(json.dumps(value, indent=4))
                txt_file.write(',\n')  # Add comma and new line after each key-value pair
            txt_file.write('\n')  #


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

    # # Update DataFrame with new results
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
            selected_demonstrations_ids = [2,4,6,3]
            random.shuffle(selected_demonstrations_ids)

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


def set_conversion(str):
    str = str.replace('and not','-')
    str = str.replace(' and ',' & ')
    str = str.replace(' or ',' | ')

    return str


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

def build_oracle_docs():
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

    return gold_examples, oracle_examples_dict

def process_rand_ids_and_gold_examples():
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
    
    return new_gold_examples
