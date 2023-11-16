import argparse
from tools import safe_execute


def main(args):
    test_dir = args.test_dir
    program = '''
question = 'Birds found in the Venezuelan Andes but not in Colombia'

# Define the subquestions
venezuelan_andes_birds = 'Birds found in the Venezuelan Andes'
colombian_birds = 'Birds found in Colombia'

# Find the relevant docs for each subquestion
docs_venezuelan_andes  = find_docs(venezuelan_andes_birds) 
docs_colombian_birds = find_docs(colombian_birds)

# Combine using the correct logical operator if needed
answer = docs_venezuelan_andes and not docs_colombian_birds
    '''
    ans = safe_execute(program)
    print(ans)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='GPT API')
    parser.add_argument('--test_dir', type=str, default="quest_data\\val.jsonl")
    args = parser.parse_args()
    main(args)