from quest.common import example_utils
import random
from template_construction import create_rand_demonstrations, concat_demonstations, concat_test2prompt
from templates_wizard_lm import INSTRUCTION_DOCS_ANON, DEMONSTRATIONS_DOCS_ANON, TEST_TEMPLATE_DOCS_ANON
from templates_code_llama import INSTRUCTION_USER_ASSISTANT, DEMONSTRATIONS_USER_ASSISTANT, TEST_TEMPLATE_USER_ASSISTANT
import argparse
import torch
import os
from transformers import LlamaForCausalLM, CodeLlamaTokenizer, DataCollatorWithPadding, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from templates import demonstration_op_map
from datasets.code_llm_dataset import Code_llm_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import uuid
import pickle
import json
from seeds import set_seed
from data.prepare_dataset import read_queries
from collections import Counter

from utils import print_args, tokenize_demonstrations,tokenize_data, bytes_to_giga_bytes, load_pickle, tokenize_user_assistant

def main(args):
    args.user_assistant = True
    print_args(args)
    set_seed(args.seed)

    path = os.path.join(args.data_dir,args.test_file)

    path_qid_queries = os.path.join(args.data_dir, args.test_query_ids_queries)
    path_qid_did = os.path.join(args.data_dir, args.test_query_ids_doc_ids)
    test_dict_query_ids_queries, _ = read_queries(path_qid_queries, path_qid_did)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    # tokenizer info
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    print(f'tokenizer.padding_side {tokenizer.padding_side} !!!!!')

    collator = DataCollatorWithPadding(tokenizer)#, pad_to_multiple_of=8)

    if args.user_assistant:
        all_input_ids, all_attention_mask, all_qids = tokenize_user_assistant(INSTRUCTION_USER_ASSISTANT, DEMONSTRATIONS_USER_ASSISTANT, TEST_TEMPLATE_USER_ASSISTANT, test_dict_query_ids_queries, args, tokenizer)
    else:
        dict_tokenized_demonstations = tokenize_demonstrations(tokenizer, DEMONSTRATIONS_DOCS_ANON, args.seed)
        all_input_ids, all_attention_mask, all_qids = tokenize_data(test_dict_query_ids_queries, tokenizer, dict_tokenized_demonstations, args, INSTRUCTION_DOCS_ANON)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device {device}')

    code_dataset = Code_llm_dataset(all_input_ids, all_attention_mask, all_qids)

    kwargs = {}
    if args.bit4:
        bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
        )
        kwargs['quantization_config'] = bnb_config
    model = AutoModelForCausalLM.from_pretrained(args.model_name,load_in_8bit=args.bit8,device_map="auto",**kwargs)
    
    if args.better_transformer:
        model = model.to_bettertransformer()

    if not (args.bit8 or args.bit4):
        model.to(device) # not needed for 4bit and 8bit

    dataloader = DataLoader(code_dataset,batch_size=args.batch_size, collate_fn=collator, shuffle=True)

    # model.resize_token_embeddings(model.config.vocab_size + 1) # because we added pad_token
    all_generated_ids = []
    all_qids = []

    max_new_tokens = args.max_gen_length
    print('before inference')
    # with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=False, enable_mem_efficient=False):
    for batch in tqdm(dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        qids = batch['qids']
        generated_ids = model.generate(input_ids, attention_mask = attention_mask,max_new_tokens=max_new_tokens)
        all_qids.extend(qids)
        all_generated_ids.extend(generated_ids[:,-max_new_tokens:])
       
    
        # print('GB ',bytes_to_giga_bytes(torch.cuda.max_memory_allocated()))

    dict_generated = {}
    for qid, generated_ids in zip(all_qids, all_generated_ids):
        qid= qid.item()
        query = test_dict_query_ids_queries[qid]
        dict_generated[query] = tokenizer.decode(generated_ids, skip_special_tokens=True)
   
    random_identifier = str(uuid.uuid4())[:5]
    print(f'random id {random_identifier}')
    if 'codellama' in args.model_name or 'WizardLM' in args.model_name:
        index_name =  args.model_name.index('/')
        file_name = random_identifier+'_'+args.model_name[index_name+1:]+'.json'
    
    with open(file_name, 'w') as f:
        json.dump(dict_generated, f, indent=2)
    '''test_query_ids_doc_ids
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
    parser.add_argument('--test_query_ids_queries', type=str, default="test_query_ids_queries.tsv")
    parser.add_argument('--test_query_ids_doc_ids', type=str, default="test_query_ids_doc_ids.tsv")
    parser.add_argument('--model_name', type=str, default="gpt2")
    parser.add_argument('--tokenizer', type=str, default="WizardLM/WizardCoder-Python-7B-V1.0")
    parser.add_argument('--dem_method', type=str, default="constant") # always the same four
    parser.add_argument('--seed',type=int, default=0)
    parser.add_argument('--batch_size',type=int, default=1)
    parser.add_argument('--num_demonstrations',type=int, default=4)
    parser.add_argument('--max_gen_length',type=int, default=5)
    parser.add_argument('--bit8',action='store_true') # default false now!
    parser.add_argument('--bit4',action='store_true') # default false now!
    parser.add_argument('--better_transformer',action='store_true') # default false now!
    parser.add_argument('--use_flash_attention_2',action='store_true') # default false now!
    parser.add_argument('--user_assistant',action='store_true') # default false now!
    args = parser.parse_args()
    main(args)