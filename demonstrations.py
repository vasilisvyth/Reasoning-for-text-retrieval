import argparse
from quest.common import example_utils
from collections import defaultdict
from seeds import set_seed
import random
from templates import template2logic
import openai
from time import sleep

def choose_demonstrations(examples_per_template):

    demonstrations = []
    for template in examples_per_template:
        cur_template_examples = examples_per_template[template]
        pos = random.randint(0, len(cur_template_examples))

        cur_template_demonstration = cur_template_examples[pos]
        demonstrations.append(cur_template_demonstration)

    return demonstrations

def queries_from_examples(examples):
    queries = []
    for example in examples:
        tmp_query = example.query
        queries.append(tmp_query)
    
    return queries


def main(args):
    train_dir = args.train_dir

    set_seed(args.seed)
    examples = example_utils.read_examples(train_dir)

    templates = []
    examples_per_template = defaultdict(list)
    for example in examples:
        tmp_template = example.metadata.template

        examples_per_template[tmp_template].append(example)

        templates.append(tmp_template)
    
    demonstrations = choose_demonstrations(examples_per_template)
    
    queries_demonstrations = queries_from_examples(demonstrations)

    print(queries_demonstrations)
    # if args.examples > 0:
    #     random.shuffle(examples)
    #     examples = examples[:FLAGS.sample]

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='List of Demonstrations')
    parser.add_argument('--train_dir', type=str, default="quest_data\\train_aug.jsonl")
    parser.add_argument('--num_per_template',type=int, default=1, help='pool size')
    parser.add_argument('--seed',type=int, default=0)
    args = parser.parse_args()
    main(args)