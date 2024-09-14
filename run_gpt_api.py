import argparse
from tools import safe_execute
import re
from openai import OpenAI
from time import sleep
from tqdm import tqdm
from seeds import set_seed
import random
from quest.common import example_utils
from templates.templates import TEST_TEMPLATE, INSTRUCTION, DEMONSTRATIONS
from templates.templates_better_dem import INSTRUCTION_BETTER_DEM, DEMONSTRATIONS_BETTER_DEM, TEST_TEMPLATE_BETTER_DEM
from templates.templates_docs_anon_quest import INSTRUCTION_DOCS_ANON, DEMONSTRATIONS_DOCS_ANON, TEST_TEMPLATE_DOCS_ANON
from templates.templates_docs_new_dem import TEST_TEMPLATE_DOCS_NEW, INSTRUCTION_DOCS_NEW, DEMONSTRATIONS_DOCS_NEW
from templates.templates_ngram import INSTRUCTION_DOCS_N_GRAM, DEMONSTRATIONS_DOCS_N_GRAM, TEST_TEMPLATE_DOCS_N_GRAM
from templates.template_construction import create_rand_demonstrations, concat_demonstations, concat_test2prompt
from templates.templates_query2doc import INSTRUCTION_DOCS_Q2DOC
import json
import pprint
from chat_gpt_utils import calculate_cost, initialize_openai_client, chat_completion
from utils import create_pickle, load_pickle
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  

def gpt_api_call(test_dict, args, openai_key, file_path):
    completion_tokens = 0
    prompt_tokens = 0

    client =  initialize_openai_client(openai_key)

    for qid, query in enumerate(tqdm(test_dict)):

            full_prompt = test_dict[query]['prompt']
            instruction = test_dict[query]['instruction']

            if "pred" in test_dict[query]:
                continue
            print('exists')
            if args.greedy:
                # greedy decoding
                got_result = False
                while not got_result:
                    try:
                        if args.model_name == "gpt-3.5-turbo-1106":
                            messages=[
                                {"role": "system", "content": instruction},
                                {"role": "user", "content": full_prompt}]
                            
                            result = chat_completion(client, args.model_name, messages, args.seed, max_tokens= 178, n=1)
                            
                            got_result = True
                            prediction = result.choices[0].message.content
                            token_log_pairs = [(t.token, t.logprob) for t in result.choices[0].logprobs.content]
                        elif args.model_name == 'gpt-3.5-turbo-instruct':
                            result = client.completions.create(
                                model=args.model_name,
                                prompt = instruction + '\n' + full_prompt,
                                # messages=[
                                # {"role": "system", "content": instruction},
                                # {"role": "user", "content": full_prompt}],
                                max_tokens=165,
                                n=1,
                                seed = args.seed
                                #stop=['\n\n'], # default is none
                            )
                            got_result = True
                            prediction = result.choices[0].text
                        else:
                            raise(f'Not suitable name {args.model_name}')
                    except Exception as e:
                        print(e)
                        sleep(5)

            
            test_dict[query]['pred'] = prediction
            test_dict[query]['pairs'] = token_log_pairs

            completion_tokens += result.usage.completion_tokens
            prompt_tokens += result.usage.prompt_tokens

            cost_dict = calculate_cost(completion_tokens, prompt_tokens, args.model_name)
            print(cost_dict)

            # Convert dictionary to JSON string
            json_string = json.dumps(test_dict, indent=4)

            # Write the JSON string to the file
            with open(file_path, 'w') as json_file:
                json_file.write(json_string)

            # create_pickle(random_ids, 'last_gpt_random_ids.pickle')

def main(args):
    test_dir = args.test_dir
    openai_key = input("What is your OpenAI key? ")
    out_name = input('What is the output file name? Do not add .json at the end ')
    out_name +='.json'
    set_seed(args.seed)
    test_examples = example_utils.read_examples(test_dir)

    test_dict = {}
    for test_example in test_examples:
        query = test_example.query
        if args.dem_method=='rand':
            SELECTED_DEMONSTRATIONS_IDS = create_rand_demonstrations(args.seed, args.num_demonstrations, DEMONSTRATIONS_DOCS_N_GRAM)
        else:
            SELECTED_DEMONSTRATIONS_IDS = [2,4,6,3]

            random.shuffle(SELECTED_DEMONSTRATIONS_IDS)
        if not args.q2doc:
            demonstations_text, demonstrations_ops = concat_demonstations(args.seed, SELECTED_DEMONSTRATIONS_IDS, DEMONSTRATIONS_DOCS_N_GRAM)
            demonstations_text = concat_test2prompt(demonstations_text, query, TEST_TEMPLATE_DOCS_N_GRAM)
            demonstations_text = demonstations_text.lstrip()
        else:
            demonstations_text = f'question: {query}'
            demonstrations_ops = '0-shot'
        test_dict[query] = {}
        test_dict[query]['demonstrations_ops'] = demonstrations_ops
        test_dict[query]['prompt'] = demonstations_text
        test_dict[query]['instruction'] = INSTRUCTION_DOCS_N_GRAM
        test_dict[query]['model_name'] = args.model_name 
        test_dict[query]['template'] = test_example.metadata.template
        test_dict[query]['domain'] = test_example.metadata.domain

    file_path = out_name
    gpt_api_call(test_dict, args, openai_key, file_path)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='GPT API')
    parser.add_argument('--test_dir', type=str, default="quest_data\\test.jsonl")
    parser.add_argument('--model_name', type=str, default="gpt-3.5-turbo-1106")
    parser.add_argument('--dem_method', type=str, default="constant") # always the same four
    parser.add_argument('--greedy',action='store_false') # default true now!
    parser.add_argument('--q2doc',action='store_true') # default false now!
    parser.add_argument('--seed',type=int, default=0)
    parser.add_argument('--num_demonstrations',type=int, default=4)
    args = parser.parse_args()
    main(args)