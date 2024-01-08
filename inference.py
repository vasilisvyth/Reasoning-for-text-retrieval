from tools import safe_execute, synthesize_program, DocumentFinder
import json
import argparse
import re
import numpy as np
from templates import template2logic
import os
from prepare_dataset import read_docs
from run_eval import calc_f1_pr_rec
from quest.common import example_utils
from quest.common.example_utils import Example
import pickle
import numpy as np
import torch
from tools import DocumentFinder, Operations
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
from utils import create_pickle, update_results
from python_interpreters_docs_anon import create_results

# run inference using DocumentFinder
def main(args):
    set_seed(0)

    tokenizer = AutoTokenizer.from_pretrained('google/t5-v1_1-base')
    # load docs
    path_doc_text_list = os.path.join(args.data_dir,'doc_text_list.pickle')
    path_doc_title_map = os.path.join(args.data_dir,'doc_title_map.tsv')
    doc_text_map, doc_title_map = read_docs(path_doc_text_list, path_doc_title_map)
    documents = document_utils.read_documents("quest_data\\documents.jsonl")
    title_doc_map = {value: key for key, value in doc_title_map.items()}

    path_test = os.path.join(args.data_dir,args.gold_examples_dir)
    gold_examples = example_utils.read_examples(path_test)
    

    METHOD = 'bge-large'
    if METHOD =='bge-large':
        doc_embs = np.load(os.path.join('checkpoints','bge-zero','zero_shot_t5_bge_large.npy'))
        doc_embs = torch.from_numpy(doc_embs)
        CHECKPOINT = 'BAAI/bge-large-en-v1.5'
        use_sentence_transformer=True
        normalize_embeddings=True
        #DocumentFinder.init('BAAI/bge-large-en-v1.5', doc_embs, tokenizer, k=30000, method='bge-large', use_sentence_transformer=True, normalize_embeddings=True)
    elif METHOD == 't5-base':
        doc_embs = torch.load('checkpoints/4913e0dd-b8/scores_docs_check_13500.pt')
        CHECKPOINT = 'checkpoints/4913e0dd-b8/checkpoint-13500/'
        use_sentence_transformer=False
        normalize_embeddings=False

    args.gpt_results_dir = 'q2doc.json'

    RANK_CONSTANT = 60
    K = 30000
    replace_find = True
    SCORE = f'1/({RANK_CONSTANT}+rank' #'emb'
    THRESHOLD = 0 #'None'
    DocumentFinder.init(CHECKPOINT, doc_embs, tokenizer, k=K, method=METHOD, replace_find = replace_find, use_sentence_transformer=use_sentence_transformer,
                         normalize_embeddings=normalize_embeddings, score=SCORE, threshold=THRESHOLD,return_dict=False)

    data = {'and':'avg','or':'maximum','diff':'subtract','score':SCORE}
    INFO = { 'model':METHOD,'checkpoint':CHECKPOINT,
             'K':K,'threshold':THRESHOLD,'replace_find':replace_find,'template':args.gpt_results_dir
    }
    INFO.update(data)

    Operations.init(data, RANK_CONSTANT)
    
    file=open(args.gpt_results_dir, "r")
    results_json = json.load(file)
    
    true_ops = 0
    count_ops = 0
    all_subquestions = [] # debug_purpose
    count_bug = 0
    count_not_ans = 0
    count_find_bugs = 0
    count_forgot = 0
    pred_examples = []
    results_len = []
    new_gold_examples = []
    count_parenthesis = 0
    lens_per_template = defaultdict(list)
    i = -1
    for qid, query in enumerate(tqdm(results_json)):
        if not 'pred' in results_json[query]:
            # print('forgot sth')
            count_forgot +=1
            continue

        
        for ex in gold_examples:
            if ex.query==query:
                curr_ex = ex
                break
        result = results_json[query]['pred']
        ans = DocumentFinder.find_docs(query,query)

    
        sorted_items = sorted(ans.items(), key=lambda x: x[1], reverse=True)
            # Take the top k items
        initial_top_k_keys = [key for key, _ in sorted_items]
        top_k_keys = initial_top_k_keys[:1000]
        sorted_pred_doc_titles = [documents[idx].title for idx in top_k_keys]
        new_gold_examples.append(curr_ex)

        tmp_pred_example = Example(query=query, docs=sorted_pred_doc_titles)
        pred_examples.append(tmp_pred_example)

    
    
    
    create_results(new_gold_examples, pred_examples, count_bug)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='GPT API')
    parser.add_argument('--gpt_results_dir', type=str, default="docs_anon.json")
    parser.add_argument('--data_dir', type=str, default="quest_data")
    parser.add_argument('--result_dir', type=str, default="res_docs_anon_1000.json")
    parser.add_argument('--gold_examples_dir', type=str, default='test.jsonl')
    # parser.add_argument('--gold_examples_dir', type=str, default='test.jsonl')
    parser.add_argument('--k', type=int, default=1000)
    args = parser.parse_args()
    main(args)