# baseline given a document our goal is to generate a query

import openai
import argparse
import os
from templates.data_augmentation.synthetic_data import instructions
from templates.data_augmentation.doc2query import doc2query
from chat_gpt_utils import calculate_cost, chat_completion, initialize_openai_client
import json
from utils import create_pickle, load_pickle, json_list_to_txt
from seeds import set_seed
from data.prepare_dataset import read_docs
import random

SYSTEM = 'Your mission is to write one text retrieval example in JSON format.'
def main(args):
    set_seed(args.seed)
    temperature = args.temperature
    N = args.N

    # load docs
    doc_text_path = os.path.join(args.data_dir,'doc_text_list.pickle')
    doc_title_path = os.path.join(args.data_dir,'doc_title_map.tsv')
    doc_text_map, doc_title_map = read_docs(doc_text_path, doc_title_path)


    tmp_instruction = doc2query[args.instruction_num]
    print(tmp_instruction)

    n =load_pickle('data_gpt-3.5-turbo-0125_instruction_-1_temp_1_n_2.pickle')

    filename = f'data_{args.model_name}_instruction_{args.instruction_num}_temp_{temperature}_n_{N}'
    # res = load_pickle('data_gpt-3.5-turbo-0125_instruction_0_temp_1_n_2.pickle')
    
    openai_key = input("What is your OpenAI key? ")

    client =  initialize_openai_client(openai_key)
    domains =['book']#,'plant','animal','book']
    completion_tokens = 0
    prompt_tokens = 0
    response_format = { "type": "json_object" }
    outputs = []
    for domain in domains:
        rand_doc = random.randint(0, len(doc_text_map)-1)
        instruct = tmp_instruction.format(domain=domain, document = doc_text_map[rand_doc])
 

        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": instruct} ]
        
        result = chat_completion(client, args.model_name, messages, args.seed, args.max_tokens, N, temperature, response_format)
        for n in range(N):
            try:
                prediction = json.loads(result.choices[n].message.content)
                token_log_pairs = [(t.token, t.logprob) for t in result.choices[n].logprobs.content]
                outputs.append((prediction, token_log_pairs))
            except Exception as e:
                print(e)
                continue

        prompt_tokens += result.usage.prompt_tokens
        completion_tokens += result.usage.completion_tokens
        cost = calculate_cost(completion_tokens, prompt_tokens, args.model_name)
        print(cost)
    create_pickle(outputs,f'{filename}.pickle')

    json_list_to_txt(outputs, instruct, f'book_change_q_{filename}.txt')

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Generate data')
    parser.add_argument('--instruction_num', default=0, type=int)
    parser.add_argument('--seed',type=int, default=0)
    parser.add_argument('--N',type=int, default=1)
    parser.add_argument('--temperature',type=int, default=None)
    parser.add_argument('--max_tokens',type=int, default=1200)
    parser.add_argument('--model_name', default='gpt-3.5-turbo-0125', type=str)
    parser.add_argument("--data_dir", type=str, default='quest_data', help="The data folder where you have the data")
    args = parser.parse_args()
    main(args)
    