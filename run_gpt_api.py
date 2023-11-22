import argparse
from tools import safe_execute
import re
from templates import template2logic
from openai import OpenAI
from time import sleep
from tqdm import tqdm
from seeds import set_seed
import random
from quest.common import example_utils
from templates import TEST_TEMPLATE, INSTRUCTION, DEMONSTRATIONS
from templates_better_dem import INSTRUCTION_BETTER_DEM, DEMONSTRATIONS_BETTER_DEM, TEST_TEMPLATE_BETTER_DEM
from template_construction import create_rand_demonstrations, concat_demonstations, concat_test2prompt
import json
import pprint
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

def calculate_cost(completion_tokens, prompt_tokens, model_name):
   # gpt-3.5-turbo-1106	input $0.0010 / 1K tokens completion $0.0020 / 1K tokens
   if model_name == "gpt-3.5-turbo-1106":
        cost = completion_tokens / 1000 * 0.002 + prompt_tokens / 1000 * 0.001
   elif model_name == 'gpt-3.5-turbo-instruct':
        cost = completion_tokens / 1000 * 0.002 + prompt_tokens / 1000 * 0.0015
   return {"completion_tokens": completion_tokens, "prompt_tokens": prompt_tokens, "cost": cost}


def gpt_api_call(test_dict, args, openai_key, file_path):
    completion_tokens = 0
    prompt_tokens = 0
    client = OpenAI(
    api_key=openai_key,  # this is also the default, it can be omitted
    )
 
    for query in tqdm(test_dict):
            # we randomly use a sentence with 0.23 probability
            # p = random.uniform(0, 1)
            # if p >= 0.2:
            #     continue
            # i += 1
            # if i == 43:
            #     break
            full_prompt = test_dict[query]['prompt']
            instruction = test_dict[query]['instruction']

            if "pred" in test_dict[query]:
                continue

            if args.greedy:
                # greedy decoding
                got_result = False
                while not got_result:
                    try:
                        if args.model_name == "gpt-3.5-turbo-1106":
                            result = client.chat.completions.create(
                                model=args.model_name,
                                messages=[
                                {"role": "system", "content": instruction},
                                {"role": "user", "content": full_prompt}],
                                max_tokens=200,
                                n=1,
                                seed = args.seed
                                #stop=['\n\n'], # default is none
                            )
                            got_result = True
                            prediction = result.choices[0].message.content
                        elif args.model_name == 'gpt-3.5-turbo-instruct':
                            result = client.completions.create(
                                model=args.model_name,
                                prompt = instruction + '\n' + full_prompt,
                                # messages=[
                                # {"role": "system", "content": instruction},
                                # {"role": "user", "content": full_prompt}],
                                max_tokens=200,
                                n=1,
                                seed = args.seed
                                #stop=['\n\n'], # default is none
                            )
                            got_result = True
                            prediction = result.choices[0].text
                        else:
                            raise(f'Not suitable name {args.model_name}')
                    except Exception:
                        sleep(5)

            
            test_dict[query]['pred'] = prediction

            completion_tokens += result.usage.completion_tokens
            prompt_tokens += result.usage.prompt_tokens

            cost_dict = calculate_cost(completion_tokens, prompt_tokens, args.model_name)
            print(cost_dict)

            # Convert dictionary to JSON string
            json_string = json.dumps(test_dict, indent=4)

            # Write the JSON string to the file
            with open(file_path, 'w') as json_file:
                json_file.write(json_string)

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
            selected_demonstrations_ids = create_rand_demonstrations(args.seed, args.num_demonstrations, DEMONSTRATIONS_BETTER_DEM)
        else:
            #  = [ DEMONSTRATIONS_BETTER_DEM['ex2'], DEMONSTRATIONS_BETTER_DEM['ex4'], DEMONSTRATIONS_BETTER_DEM['ex6'], DEMONSTRATIONS_BETTER_DEM['ex3'] ]
            selected_demonstrations_ids = [2,4,6,3]
            random.shuffle(selected_demonstrations_ids)

        demonstations_text, demonstrations_ops = concat_demonstations(args.seed, selected_demonstrations_ids, DEMONSTRATIONS_BETTER_DEM)
        demonstations_text = concat_test2prompt(demonstations_text, query, TEST_TEMPLATE_BETTER_DEM)
        demonstations_text = demonstations_text.lstrip()
        test_dict[query] = {}
        test_dict[query]['demonstrations_ops'] = demonstrations_ops
        test_dict[query]['prompt'] = demonstations_text
        test_dict[query]['instruction'] = INSTRUCTION_BETTER_DEM
        test_dict[query]['model_name'] = args.model_name 
        test_dict[query]['template'] = test_example.metadata.template
        test_dict[query]['domain'] = test_example.metadata.domain

    file_path = out_name
    gpt_api_call(test_dict, args, openai_key, file_path)

    # ans = safe_execute(program)
    # print(ans)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='GPT API')
    parser.add_argument('--test_dir', type=str, default="quest_data\\test.jsonl")
    parser.add_argument('--model_name', type=str, default="gpt-3.5-turbo-instruct")
    parser.add_argument('--dem_method', type=str, default="constant") # always the same four
    parser.add_argument('--greedy',action='store_false') # default true now!
    parser.add_argument('--seed',type=int, default=0)
    parser.add_argument('--num_demonstrations',type=int, default=4)
    args = parser.parse_args()
    main(args)