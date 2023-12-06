from quest.common import example_utils
import random
from template_construction import create_rand_demonstrations, concat_demonstations, concat_test2prompt
from templates_llama import INSTRUCTION_DOCS_ANON, DEMONSTRATIONS_DOCS_ANON, TEST_TEMPLATE_DOCS_ANON
import argparse
import torch
import os
from transformers import LlamaForCausalLM, CodeLlamaTokenizer, DataCollatorWithPadding, AutoModelForCausalLM, AutoTokenizer
from templates import demonstration_op_map
from code_llm_dataset import Code_llm_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import uuid
import pickle
import json
from seeds import set_seed
from prepare_dataset import read_queries

def llama_tokenize_demonstrations(tokenizer, demonstrations, seed):
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

def tokenize_data(test_dict_query_ids_queries, tokenizer, dict_tokenized_demonstations):
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

        input_ids, attention_mask = concat_tokenized_demonstrations(dict_tokenized_demonstations, selected_demonstrations_ids)

        test_example = TEST_TEMPLATE_DOCS_ANON.format(question=query)

        tokenized_test_example = tokenizer(test_example)

        input_ids.extend(tokenized_test_example['input_ids'])
        attention_mask.extend(tokenized_test_example['attention_mask'])

        all_input_ids.append(input_ids)
        all_attention_mask.append(attention_mask)
        all_qids.append(query_id)

    return all_input_ids, all_attention_mask, all_qids

def main(args):
    print_args(args)
    set_seed(args.seed)

    path = os.path.join(args.data_dir,args.test_file)

    test_dict_query_ids_queries, _ = read_queries(os.path.join(args.data_dir,'test_query_ids_queries.tsv'), 
                                                 os.path.join(args.data_dir,'test_query_ids_doc_ids.tsv'))

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    # tokenizer info
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    # tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.padding_side = "left"
    # print(f'tokenizer.padding_side {tokenizer.padding_side}')

    collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)

    dict_tokenized_demonstations = llama_tokenize_demonstrations(tokenizer, DEMONSTRATIONS_DOCS_ANON, args.seed)
    
    all_input_ids, all_attention_mask, all_qids = tokenize_data(test_dict_query_ids_queries, tokenizer, dict_tokenized_demonstations)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device {device}')

    code_dataset = Code_llm_dataset(all_input_ids, all_attention_mask, all_qids)
    batch_size = 6
    print(f'batch_size {batch_size}')
    dataloader = DataLoader(code_dataset,batch_size=batch_size, collate_fn=collator, shuffle=True)
    # if torch.cuda.is_bf16_supported():
    #     print('bf16 suported')

    # PROMPT = '[INST] Your task is to write 5 tests to check the correctness of a function that solves a programming problem. The tests must be between [TESTS] and [/TESTS] tags. You must write the comment "#Test case n:" on a separate line directly above each assert statement, where n represents the test case number, starting from 1 and increasing by one for each subsequent test case. Problem: Write a Python function to get the unique elements of a list. [/INST]'

    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.to(device)
    
    # model.resize_token_embeddings(model.config.vocab_size + 1) # because we added pad_token
    all_generated_ids = []
    all_qids = []
    i = 0
    print('before inference')
    for batch in tqdm(dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        qids = batch['qids']
        print('before generate')
        print(f"Model's device: {next(model.parameters()).device}")
        generated_ids = model.generate(input_ids, attention_mask= attention_mask)
        all_qids.extend(qids)
        all_generated_ids.extend(generated_ids)
        i+=1
        if i > 10:
            break
    
    dict_generated = {}
    for qid, generated_ids in zip(all_qids, all_generated_ids):
        qid= qid.item()
        query = test_dict_query_ids_queries[qid]
        dict_generated[query] = tokenizer.decode(generated_ids, skip_special_tokens=True)
   
    random_identifier = str(uuid.uuid4())[:5]
    if 'codellama' in args.model_name or 'WizardLM' in args.model_name:
        index_name =  args.model_name.index('/')
        file_name = random_identifier+'_'+args.model_name[index_name+1:]+'.json'
    
    with open(file_name, 'w') as f:
        json.dump(dict_generated, f, indent=2)
    '''
    dialogs: List[Dialog]
    Dialog = List[Message]
    Message(TypedDict):
        role: Role
        content: str
    '''

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Generate programs using open source programs')
    parser.add_argument('--test_file', type=str, default="test.jsonl")
    parser.add_argument('--data_dir', type=str, default="quest_data")
    parser.add_argument('--model_name', type=str, default="gpt2")
    parser.add_argument('--tokenizer', type=str, default="WizardLM/WizardCoder-Python-7B-V1.0")
    parser.add_argument('--dem_method', type=str, default="constant") # always the same four
    parser.add_argument('--seed',type=int, default=0)
    parser.add_argument('--num_demonstrations',type=int, default=4)
    args = parser.parse_args()
    main(args)