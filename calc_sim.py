import numpy as np
import torch
import torch.nn.functional as F
from utils import load_pickle
from quest.common.example_utils import Example
from data.prepare_dataset import read_queries, read_docs
import os
from quest.common import example_utils
from python_interpreters_docs_anon import create_results
import argparse

def main(args):
    data_dir = 'quest_data'
    rand_qids = load_pickle('files/rand_qids.pickle')
    rand_qids = load_pickle('files/rand_qids_without_bugs.pickle')

    dids = load_pickle('files/rand_doc_ids.pickle')
    eval_k = 1000
    gold_examples_dir = 'test.jsonl'
    path_test = os.path.join(data_dir,gold_examples_dir)
    gold_examples = example_utils.read_examples(path_test)
    dict_query_ids_queries, test_query_ids_doc_ids = read_queries(os.path.join(data_dir, 'test_query_ids_queries.tsv'), 
                                                                            os.path.join(data_dir,'test_query_ids_doc_ids.tsv'))

    # load docs
    path_doc_text_list = os.path.join(data_dir,'doc_text_list.pickle')
    path_doc_title_map = os.path.join(data_dir,'doc_title_map.tsv')
    doc_text_map, doc_title_map = read_docs(path_doc_text_list, path_doc_title_map)

    query_embed = np.load('checkpoints/mistral_instruct/query_decomp_and_initial_rand_zero_shot_mistral.npy')
    query_embed =torch.from_numpy(query_embed)
    query_embed = F.normalize(query_embed, p=2, dim=1)

    doc_embed = np.load('checkpoints/mistral_instruct/rand_zero_shot_mistral.npy')
    doc_embed = torch.from_numpy(doc_embed)
    doc_embed = F.normalize(doc_embed, p=2, dim=1) 
    sim = query_embed @ doc_embed.T
    top_k_values, top_k_indices = torch.topk(sim, eval_k, dim=1, sorted=True)

    all_pred_examples = []
    new_gold_examples = []
    # Creating examples and adding them to the list
    for i, qid in enumerate(rand_qids):
        query_text = dict_query_ids_queries[qid] # correct query
        try:
            pred_docs_ids = [dids[index] for index in top_k_indices[i].tolist()]
            doc_texts = [doc_title_map[pr_id] for pr_id in pred_docs_ids]
            scores = top_k_values[i].tolist()
        except Exception as e:
            print(e)
            doc_texts = []
            scores = None

        example = Example(query=query_text, docs=doc_texts, scores=scores)
        all_pred_examples.append(example)

        for ex in gold_examples:
            if ex.query==query_text:
                curr_ex = ex
                break

        new_gold_examples.append(curr_ex)

    INFO = { 'info':'decomposition 0 rand qids true and top25 dids from bge','model':'mistral-instruct','checkpoint':'0-shot'
        }
    create_results(new_gold_examples, all_pred_examples, 0, INFO)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Calculate similarities given docs and queries')
    parser.add_argument('--rand_dids',default='')
    parser.add_argument('--rand_qids',default='')
    args = parser.parse_args()
    main(args)