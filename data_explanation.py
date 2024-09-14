import openai
import argparse
import os
from templates.template_explanation import instructions
from chat_gpt_utils import calculate_cost, chat_completion, initialize_openai_client
import json
from utils import json_list_to_txt
from data.prepare_dataset import read_docs
import wandb
from wandb.integration.openai import autolog
from quest.common import example_utils
from tqdm import tqdm
import pandas as pd
from time import sleep
import random

def main(args):
    examples_test = example_utils.read_examples(os.path.join(args.data_dir,'test.jsonl'))

    examples_train = example_utils.read_examples(os.path.join(args.data_dir,'train_aug.jsonl'))
    random.shuffle(examples_test)
    tmp_instruction = instructions[args.instruction_num]
    print(tmp_instruction)

    filename = f'vp_{args.model_name}_instruction_{args.instruction_num}'

    with open(f'{filename}.json', 'r') as j:
        test_dict= json.loads(j.read())

  
    openai_key = input("What is your OpenAI key? ")

    client =  initialize_openai_client(openai_key)

    completion_tokens = 0
    prompt_tokens = 0
    outputs = []

    system_instruction = ''#'You are a prompt engineer expert. Your goal is to create a text instruction that will replace a query.'#'Your mission is to write one text retrieval example in JSON format.'
    wanb_logs = []
    wandb.init(project = 'Reasoning-for-text-retrieval')
    for i, example in tqdm(enumerate(examples_test)):
        if example.metadata.template != "_ that are not _": continue
        if example.query in test_dict:
            continue
        instruct = instructions[args.instruction_num].format(query=example.query)

        messages = []
        if system_instruction:
             messages.append({"role": "system", "content": system_instruction})
        
        messages.append({"role": "user", "content": instruct})
        
        got_result = False
        while not got_result:
            try:
                result = chat_completion(client, args.model_name, messages, args.seed, args.max_tokens, n=1)
                got_result = True
            except Exception as e:
                print(e)
                sleep(3)

        prediction = result.choices[0].message.content
        token_log_pairs = [(t.token, t.logprob) for t in result.choices[0].logprobs.content]

        wanb_logs.append({'system_instruction':system_instruction, 'instruction':instruct, 'prediction':prediction, 'template':example.metadata.template,'model':args.model_name})
        test_dict[example.query] = {}
        test_dict[example.query]['pred'] = prediction
        test_dict[example.query]['token_log_pairs'] = token_log_pairs  
        test_dict[example.query]['system_instruction'] = system_instruction
        test_dict[example.query]['user_instruction'] = instruct
        
        prompt_tokens += result.usage.prompt_tokens
        completion_tokens += result.usage.completion_tokens
        cost = calculate_cost(completion_tokens, prompt_tokens, args.model_name)
        print(cost)

        # Convert dictionary to JSON string
        json_string = json.dumps(test_dict, indent=4)

        # Write the JSON string to the file
        with open(f'{filename}.json', 'w') as json_file:
            json_file.write(json_string)

        csv_filename = f'vp_{args.model_name}_instruction_0.csv'
        if os.path.exists(csv_filename):
            # Load existing data
            existing_df  = pd.read_csv(csv_filename)
        else:
            # Create an empty DataFrame if the file doesn't exist
            existing_df  = pd.DataFrame()

        # let's convert parsed_generations to a pandas dataframe and save it locally
        df = pd.DataFrame(wanb_logs)
        df.to_csv(csv_filename, index=False)

        existing_df = pd.concat([existing_df,df], ignore_index=True)

        # log df as a table to W&B for interactive exploration
        wandb.log({"generated_examples": wandb.Table(dataframe=df)})

    # log csv file as an artifact to W&B for later use
    artifact = wandb.Artifact("generated_examples", type="dataset")
    artifact.add_file(f"{filename}.csv")
    wandb.log_artifact(artifact)
    wandb.finish()

    json_list_to_txt(outputs, tmp_instruction, f'{filename}.txt')

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Generate data')
    parser.add_argument('--instruction_num', default=0, type=int)
    parser.add_argument('--seed',type=int, default=0)
    parser.add_argument('--max_tokens',type=int, default=1200)
    parser.add_argument('--model_name', default='gpt-4-0125-preview', type=str)
    parser.add_argument("--data_dir", type=str, default='quest_data', help="The data folder where you have the data")
    args = parser.parse_args()
    main(args)