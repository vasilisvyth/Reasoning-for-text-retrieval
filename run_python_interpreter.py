'''
Given the gpt predictions, we execute the python programs and save the predictions
'''

import json
import argparse
import re
import numpy as np
from templates.templates import template2logic
import os
from data.prepare_dataset import read_docs, read_queries
from run_eval import calc_f1_pr_rec
from quest.common import example_utils
from quest.common.example_utils import Example, ExampleMetadata
import pickle
import numpy as np
import torch
from tools import Operations, safe_execute, synthesize_program, VP_BM25, VP_HuggingFace, VP_SentenceTransformer
from analyze_retriever import calc_mrec_rec
from quest.common import document_utils
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
from bi_encoder import DenseBiEncoder
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from utils import load_pickle
from seeds import set_seed
from utils import create_pickle, update_results, print_args
import torch.nn.functional as F
from example_with_subquestions import ExampleWithSubQuestions

from utils import set_conversion, current_program, preprocess_find_docs_call, create_results, plot_lens, append_subq_examples, process_template_example, extract_subquestions, build_oracle_docs, process_rand_ids_and_gold_examples

def main(args):
    set_seed(0)
    print_args(args)

    path_doc_text_list = os.path.join(args.data_dir,'doc_text_list.pickle')
    path_doc_title_map = os.path.join(args.data_dir,'doc_title_map.tsv')
    doc_text_map, doc_title_map = read_docs(path_doc_text_list, path_doc_title_map)

    path_doc = os.path.join('quest_data','documents.jsonl')
    documents = document_utils.read_documents(path_doc)
    title_doc_map = {value: key for key, value in doc_title_map.items()} #! duplicates doc titles so smaller length

    path_test = os.path.join(args.data_dir,args.gold_examples_dir)
    gold_examples = example_utils.read_examples(path_test)
     
    if args.oracle_docs:
        gold_examples, oracle_examples_dict = build_oracle_docs()
    else:
        oracle_examples_dict = None

    
    RANK_CONSTANT = args.rank_constant
    replace_find = args.replace_find
    SCORE = f'1/({RANK_CONSTANT}+rank' # 'emb' #
    THRESHOLD = args.threshold#0.64 #'None'

    METHOD = args.method
    
    normalize_embeddings=False
    doc_embs = None
    tokenizer = None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    K = 30000
    if METHOD =='bge-large':
        doc_embs = np.load(os.path.join('checkpoints','bge-zero','zero_shot_t5_bge_large.npy'))
        doc_embs = torch.from_numpy(doc_embs)
        CHECKPOINT = 'BAAI/bge-large-en-v1.5'
        normalize_embeddings=True
        use_cash = False
        results = os.path.join(args.files_dir,'vp_{METHOD}_results.pickle') if use_cash else {}
        VP_SentenceTransformer.init(device, CHECKPOINT, doc_embs, K, METHOD, replace_find, SCORE, threshold=THRESHOLD,  normalize_embeddings=False, 
            oracle_examples_dict = oracle_examples_dict, use_cache=use_cash)
        class_name = 'VP_SentenceTransformer'
        assert(len(doc_embs) == len(documents)== len(doc_title_map) == 325505)
    elif METHOD == 't5-base':
        doc_embs = torch.load('checkpoints/4913e0dd-b8/scores_docs_check_13500.pt')
        CHECKPOINT = 'checkpoints/4913e0dd-b8/checkpoint-13500/'
        tokenizer = AutoTokenizer.from_pretrained('google/t5-v1_1-base')

        VP_HuggingFace.init(device, CHECKPOINT, doc_embs, tokenizer, K, METHOD, replace_find, SCORE, threshold='none', normalize_embeddings=normalize_embeddings, 
            oracle_examples_dict=oracle_examples_dict)
        class_name = 'VP_HuggingFace'
    elif METHOD == 'mistral':
        use_cash = True # use existing already calculated embeddings
        if use_cash: device = torch.device('cpu')
        
        doc_embs = np.load(os.path.join('checkpoints','mistral_instruct','rand_zero_shot_mistral.npy'))
        doc_embs = torch.from_numpy(doc_embs)
        doc_embs = F.normalize(doc_embs, p=2, dim=1) 
        doc_embs = doc_embs.to(device)

        # just avoid mistral locally 
        CHECKPOINT = 't5-small' if use_cash else 'intfloat/e5-mistral-7b-instruct'
        tokenizer = 'intfloat/e5-mistral-7b-instruct'

        tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        class_name = 'VP_HuggingFace'
        K = len(doc_embs)
        
        
        results = load_pickle('vp_mistral_rand_results.pickle') if use_cash else {}
        
        VP_HuggingFace.init(device, CHECKPOINT, doc_embs, tokenizer, K, METHOD, replace_find, SCORE, threshold=THRESHOLD, normalize_embeddings=normalize_embeddings, 
            oracle_examples_dict=oracle_examples_dict, results=results, use_cache=use_cash)

        # the following code is only useful when we work with rand qids and rand dids
        gold_examples = process_rand_ids_and_gold_examples()

    elif METHOD=='bm25':
        CHECKPOINT = 'dum_bm25_obj.pickle'
        VP_BM25.init(CHECKPOINT, doc_embs, K, replace_find, SCORE, threshold='None',
            oracle_examples_dict=oracle_examples_dict)
        class_name = 'VP_BM25'
    else:
        raise(f'{METHOD} is not supported')
        

    data = {'and':'avg','or':'maximum','diff':'subtract','score':SCORE}
    INFO = { 'info':'','model':METHOD,'checkpoint':CHECKPOINT,
             'K':K,'threshold':THRESHOLD,'replace_find':replace_find,'template':args.gpt_results_dir
    }
    INFO.update(data)

    Operations.init(data, RANK_CONSTANT)
    
    file=open(args.gpt_results_dir, "r")
    results_json = json.load(file)
    

    count_bug = 0
    count_forgot = 0
    pred_examples = []

    new_pred_examples = [ ]
    new_gold_examples = []
    lens_per_template = defaultdict(list)
    examples_with_subquestions = []
    i = -1
    args.seperate_subquestions = False
    print(f'Calculate seperate subquestions metrics? {args.seperate_subquestions}')
    for j, ex in enumerate(tqdm(gold_examples)):
        query = ex.query
        if not 'pred' in results_json[query]:
            # print('forgot sth')
            count_forgot +=1
            continue
        
        template_used = results_json[query]['template']

        result = results_json[query]['pred']

        new_result = preprocess_find_docs_call(query, result, class_name)
        program = synthesize_program(result = new_result, prefix = '')
 
        if 'ans = ' not in program:
            # count_not_ans +=1
            sorted_pred_doc_titles = []
        else:
            
            var_dict = current_program(program)

            if var_dict is not None:
                try:
                    i+=1
                    ans = var_dict['ans']
                    sorted_items = sorted(ans.items(), key=lambda x: x[1], reverse=True)

                    # Take the top k items
                    initial_top_k_keys = [key for key, _ in sorted_items]
                    top_k_keys = initial_top_k_keys[:1000]
              
                    if METHOD == 'mistral':
                        # create_pickle(top_k_keys,'debug_top_keys.pickle')
                        pred_docs_ids = [dids[index] for index in top_k_keys]
                        sorted_pred_doc_titles = [doc_title_map[pr_id] for pr_id in pred_docs_ids]
                    else:
                        sorted_pred_doc_titles = [documents[idx].title for idx in top_k_keys]

                    if args.seperate_subquestions:
                        process_template_example(ex, var_dict, documents, new_gold_examples, new_pred_examples)

                    subquestions = extract_subquestions(var_dict)
                    current_example = ExampleWithSubQuestions(example =ex, subquestions = subquestions)
                    examples_with_subquestions.append(current_example)
  
                    lens_per_template[template_used].append(len(top_k_keys))

                except Exception as e:
    
                    import traceback
                    traceback.print_exc()
                    # print(final_program)
                    sorted_pred_doc_titles = []

            else:
                # count_bug +=1
                sorted_pred_doc_titles = []

        if sorted_pred_doc_titles == []:
            count_bug +=1
        tmp_pred_example = Example(query=query, docs=sorted_pred_doc_titles)
        pred_examples.append(tmp_pred_example)

    print(f'num of examples {len(pred_examples)}')
    print(f'count bug {count_bug}')

    create_pickle(examples_with_subquestions,os.path.join(args.files_dir,f'ex_with_subq_{METHOD}_rand_{args.gpt_results_dir}.pickle'))

    if METHOD == 'mistral': 
        create_pickle(VP_HuggingFace.results,os.path.join(args.files_dir,f'vp_{METHOD}_rand_{args.gpt_results_dir}.pickle'))
    elif METHOD == 'bge-large':
        create_pickle(VP_SentenceTransformer.results,os.path.join(args.files_dir,f'vp_{METHOD}_{args.gpt_results_dir}.pickle'))
    if args.seperate_subquestions:
        print(f'len {len(new_gold_examples)}')
        create_results(new_gold_examples, new_pred_examples, count_bug, INFO)
    else:
        create_results(gold_examples, pred_examples, count_bug, INFO)
    plot_lens(lens_per_template)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='GPT API')
    parser.add_argument('--gpt_results_dir', type=str, default="predictions.json")
    parser.add_argument('--method', type=str, default='bge-large')
    parser.add_argument('--data_dir', type=str, default="quest_data")
    parser.add_argument('--result_dir', type=str, default="res_predictions_1000.json")
    parser.add_argument('--gold_examples_dir', type=str, default='test.jsonl')
    parser.add_argument('--oracle_docs',action='store_true') # default false now!
    parser.add_argument('--replace_find',action='store_false') # default true now!
    parser.add_argument('--seperate_subquestions',action='store_true')
    parser.add_argument('--files_dir',type=str,default='files')
    # parser.add_argument('--gold_examples_dir', type=str, default='test.jsonl')
    parser.add_argument('--k', type=int, default=1000)
    # rank_constant
    parser.add_argument('--rank_constant', type=int, default=60)
    parser.add_argument('--threshold', type=float, default=0.64)
    args = parser.parse_args()
    main(args)
