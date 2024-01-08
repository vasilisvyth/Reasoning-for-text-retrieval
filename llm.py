try:
    from vllm import LLM, SamplingParams
    # Rest of your code that uses the library
except ImportError:
    print("vllm library is not installed. You need a cuda")

import time
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
from code_llm_dataset import Code_llm_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import uuid
import pickle
import json
from seeds import set_seed
from prepare_dataset import read_queries
from collections import Counter
from generate_code import print_args
from collections import defaultdict
from utils import tokenize_user_assistant

from utils import print_args, tokenize_demonstrations,tokenize_data, bytes_to_giga_bytes, create_pickle, load_pickle

class Wizardlm():
    '''
    WizardLM/WizardCoder-Python-34B/13B/7B-V1.0. If you want to inference with WizardLM/WizardCoder-15B/3B/1B-V1.0, please change the stop_tokens = ['</s>']
    to stop_tokens = ['<|endoftext|>']
    '''
    @classmethod
    def init(cls, model_name="WizardLM/WizardCoder-Python-34B-V1.0", max_num_batched_tokens=16384):
        cls.llm = LLM(model=model_name, max_num_batched_tokens=max_num_batched_tokens, tensor_parallel_size=1)

    @classmethod
    def generate(cls, prompt=None, prompt_token_ids=None, stop_token=None, temperature=0, top_p=1, max_new_tokens=160):

        stop_tokens = ['</s>']
        if stop_token:
            stop_tokens.append(stop_token)
        sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_new_tokens, stop=stop_tokens)
        completions = cls.llm.generate(prompt, prompt_token_ids = prompt_token_ids, sampling_params = sampling_params)
        
        return completions
        # return completions[0].outputs[0].text


class Codellama():
    @classmethod
    def init(cls, model_name="codellama/CodeLlama-34b-Python-hf", max_num_batched_tokens=8192):
        cls.llm = LLM(
            model=model_name,
            dtype="float16",
            # trust_remote_code=True,
            # tokenizer=model_name,#"hf-internal-testing/llama-tokenizer",
            max_num_batched_tokens=max_num_batched_tokens, tensor_parallel_size=1)

    @classmethod
    def generate(cls, prompt=None, prompt_token_ids=None, stop_token=None, temperature=0, top_p=1, max_new_tokens=160):
        #temperature= 0 means greedy sampling.
        stop_tokens = ['</s>']
        if stop_token:
            stop_tokens.append(stop_token)
        sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_new_tokens, stop=stop_tokens)
        completions = cls.llm.generate(prompt, prompt_token_ids = prompt_token_ids, sampling_params = sampling_params)
        return completions

def init_gpt(model_name, max_input_tokens):
    if 'WizardLM' in model_name:
        Wizardlm.init(model_name, max_input_tokens)
    elif 'codellama' in model_name:
        Codellama.init(model_name, max_input_tokens)
    print('initialization done')

def main(args):
    # args.user_assistant = True
    print_args(args)
    set_seed(args.seed)

    # obj = load_pickle('8a349_WizardCoder-Python-7B-V1.0.pickle')

    path = os.path.join(args.data_dir,args.test_file)

    test_dict_query_ids_queries, _ = read_queries(os.path.join(args.data_dir,'test_query_ids_queries.tsv'), 
                                                 os.path.join(args.data_dir,'test_query_ids_doc_ids.tsv'))

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    # tokenizer info
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    # tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    print(f'tokenizer.padding_side {tokenizer.padding_side} !!!!!')

    collator = DataCollatorWithPadding(tokenizer)#, pad_to_multiple_of=8)

    if args.user_assistant:
        all_input_ids, all_attention_mask, all_qids = tokenize_user_assistant(INSTRUCTION_USER_ASSISTANT, DEMONSTRATIONS_USER_ASSISTANT, TEST_TEMPLATE_USER_ASSISTANT, test_dict_query_ids_queries, args, tokenizer)
    else:
        dict_tokenized_demonstations = tokenize_demonstrations(tokenizer, DEMONSTRATIONS_DOCS_ANON, args.seed)
        all_input_ids, all_attention_mask, all_qids = tokenize_data(test_dict_query_ids_queries, tokenizer, dict_tokenized_demonstations, args, INSTRUCTION_DOCS_ANON)


    init_gpt(args.model_name, args.max_num_batched_tokens)
    K = 30
    all_input_ids = all_input_ids[:K]

    print(f'take the first {K} examples ')
    if 'WizardLM' in args.model_name:
        completions = Wizardlm.generate(prompt_token_ids = all_input_ids, max_new_tokens = args.max_gen_length)
    elif 'codellama' in args.model_name:
        completions = Codellama.generate(prompt_token_ids = all_input_ids, max_new_tokens = args.max_gen_length)
    else:
        raise(f'WizardLM or codellama should be part of the modelname')
    

    dict_generated = {}
    for completion in completions:
        request_id = int(completion.request_id)
        query = test_dict_query_ids_queries[request_id]
        dict_generated[query] = []
        for output in completion.outputs:
            dict_generated[query].append(output.text)

    index_name =  args.model_name.index('/')
    random_identifier = str(uuid.uuid4())[:5]
    print(f'random id {random_identifier}')
    file_name = random_identifier+'_'+args.model_name[index_name+1:]+'.json'

    with open(file_name, 'w') as f:
        json.dump(dict_generated, f, indent=2)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(f'device {device}')

    # code_dataset = Code_llm_dataset(all_input_ids, all_attention_mask, all_qids)
    
    # # if torch.cuda.is_bf16_supported():
    # #     print('bf16 suported')

    # # PROMPT = '[INST] Your task is to write 5 tests to check the correctness of a function that solves a programming problem. The tests must be between [TESTS] and [/TESTS] tags. You must write the comment "#Test case n:" on a separate line directly above each assert statement, where n represents the test case number, starting from 1 and increasing by one for each subsequent test case. Problem: Write a Python function to get the unique elements of a list. [/INST]'
    # #! todo try bnf config
    # kwargs = {}
    # if args.bit4:
    #     bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.bfloat16
    #     )
    #     kwargs['quantization_config'] = bnb_config
    # model = AutoModelForCausalLM.from_pretrained(args.model_name,load_in_8bit=args.bit8,device_map="auto",**kwargs)
    
    # if args.better_transformer:
    #     model = model.to_bettertransformer()

    # if not (args.bit8 or args.bit4):
    #     model.to(device) # not needed for 4bit and 8bit

    # dataloader = DataLoader(code_dataset,batch_size=args.batch_size, collate_fn=collator, shuffle=True)

    # # model.resize_token_embeddings(model.config.vocab_size + 1) # because we added pad_token
    # all_generated_ids = []
    # all_qids = []
    # i = 0
    # max_new_tokens = args.max_gen_length
    # print('before inference')
    # # with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=False, enable_mem_efficient=False):
    # for batch in tqdm(dataloader):
    #     input_ids = batch['input_ids'].to(device)
    #     attention_mask = batch['attention_mask'].to(device)
    #     # input_ids, attention_mask = input_ids[:2,:32], attention_mask[:2,:32]
    #     # tokenizer.batch_decode(input_ids)[0]
    #     qids = batch['qids']
    #     generated_ids = model.generate(input_ids, attention_mask = attention_mask,max_new_tokens=max_new_tokens)
    #     all_qids.extend(qids)
    #     all_generated_ids.extend(generated_ids[:,-max_new_tokens:])
    #     i+=1
    #     if i > 3:
    #         break
    
    #     print('GB ',bytes_to_giga_bytes(torch.cuda.max_memory_allocated()))

    # dict_generated = {}
    # for qid, generated_ids in zip(all_qids, all_generated_ids):
    #     qid= qid.item()
    #     query = test_dict_query_ids_queries[qid]
    #     dict_generated[query] = tokenizer.decode(generated_ids, skip_special_tokens=True)
   
    # random_identifier = str(uuid.uuid4())[:5]
    # print(f'random id {random_identifier}')
    # if 'codellama' in args.model_name or 'WizardLM' in args.model_name:
    #     index_name =  args.model_name.index('/')
    #     file_name = random_identifier+'_'+args.model_name[index_name+1:]+'.json'
    
    # with open(file_name, 'w') as f:
    #     json.dump(dict_generated, f, indent=2)
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
    parser.add_argument('--max_num_batched_tokens',type=int, default=1)
    parser.add_argument('--num_demonstrations',type=int, default=4)
    parser.add_argument('--max_gen_length',type=int, default=160)
    parser.add_argument('--user_assistant',action='store_true') # default false now!
    args = parser.parse_args()
    main(args)