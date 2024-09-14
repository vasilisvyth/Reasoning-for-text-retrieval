import numpy as np
import torch
import torch.nn.functional as F
from utils import load_pickle
from quest.common.example_utils import Example
from data.prepare_dataset import read_queries, read_docs
import os
from quest.common import example_utils
from utils import create_results
import argparse
from tqdm import tqdm
# import faiss

def create_doc_index():
    doc_embed_1 = np.load(os.path.join(args.files_dir,'docs_0_70000_last_zero_shot_mistral.npy'))
    doc_embed_2 = np.load(os.path.join(args.files_dir,'docs_70000_150000_last_zero_shot_mistral.npy'))
    doc_embed_3 = np.load(os.path.join(args.files_dir,'docs_150000_220000_last_zero_shot_mistral.npy'))
    doc_embed_4 = np.load(os.path.join(args.files_dir,'docs_220000_300000_last_zero_shot_mistral.npy'))
    doc_embed_5 = np.load(os.path.join(args.files_dir,'docs_300000_325504_last_zero_shot_mistral.npy'))
    doc_embed_6 = np.load(os.path.join(args.files_dir,'docs_325504_325505_last_zero_shot_mistral.npy'))
    doc_embed = np.concatenate((doc_embed_1, doc_embed_2, doc_embed_3, doc_embed_4, doc_embed_5, doc_embed_6))

    return doc_embed

def test_correct_docs(doc_embed):
    rand_docs = np.load(os.path.join(args.files_dir,'docs_rand_last_zero_shot_mistral.npy'))
    rand_doc_ids = [0,1,70000-1,70000,70001, 150000-1,150000,150000+1, 220000-1,220000,220000+1,300000-1,300000,300000+1,325504-1,325504]
    for index,rand_id in enumerate(rand_doc_ids):
        if not np.allclose(rand_docs[index], doc_embed[rand_id]):
            print(np.abs(rand_docs[index]- doc_embed[rand_id]))


def main(args):

    # load docs
    path_doc_text_list = os.path.join(args.data_dir,'doc_text_list.pickle')
    path_doc_title_map = os.path.join(args.data_dir,'doc_title_map.tsv')
    doc_text_map, doc_title_map = read_docs(path_doc_text_list, path_doc_title_map)

    dict_query_ids_queries, test_query_ids_doc_ids = read_queries(os.path.join(args.data_dir, 'test_query_ids_queries.tsv'), 
                                                                            os.path.join(args.data_dir,'test_query_ids_doc_ids.tsv'))
    if args.rand_qids:
        qids = load_pickle(args.rand_qids) #'files/rand_qids.pickle', 'files/rand_qids_without_bugs.pickle'
        dict_query_ids_queries = {qid : dict_query_ids_queries[qid] for qid in qids}

    if args.rand_dids:
        dids = load_pickle(args.rand_dids)
    else:
        dids = list(range(len(doc_text_map)))

    eval_k = 1000
    path_test = os.path.join(args.data_dir,args.test_examples)
    gold_examples = example_utils.read_examples(path_test)

    query_embed = np.load(os.path.join(args.files_dir,'query_0__last_zero_shot_mistral.npy'))
    query_embed = torch.from_numpy(query_embed)
    query_embed = F.normalize(query_embed, p=2, dim=1)
 
    doc_embed = create_doc_index()

    test_correct_docs(doc_embed)
    exit()

    doc_embed = torch.from_numpy(doc_embed)
    doc_embed = F.normalize(doc_embed, p=2, dim=1)
 
    all_pred_examples = []
    new_gold_examples = []
    # Creating examples and adding them to the list
    for i, (qid, query_text) in tqdm(enumerate(dict_query_ids_queries.items())):
        sim = query_embed[i] @ doc_embed.T
        top_k_values, top_k_indices = torch.topk(sim, eval_k, dim=-1, sorted=True)
        print(f'ind {top_k_indices.shape} values {top_k_values.shape} sim {sim.shape}')
        
        pred_docs_ids = [dids[index] for index in top_k_indices.tolist()]
        doc_texts = [doc_title_map[pr_id] for pr_id in pred_docs_ids]
        scores = sim.tolist()
        

        example = Example(query=query_text, docs=doc_texts, scores=scores)
        all_pred_examples.append(example)

        for ex in gold_examples:
            if ex.query==query_text:
                curr_ex = ex
                break
        new_gold_examples.append(curr_ex)

    INFO = { 'info':f'decomposition 0 {args.rand_qids} and {args.rand_dids}','model':'mistral-instruct','checkpoint':'0-shot'
        }
    create_results(new_gold_examples, all_pred_examples, 0, INFO)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Calculate similarities given docs and queries')
    parser.add_argument('--rand_dids',type=str,default='') #'files/rand_doc_ids.pickle'
    parser.add_argument('--rand_qids',type=str,default='')
    parser.add_argument('--test_examples',type=str,default='test.jsonl')
    parser.add_argument('--data_dir',type=str,default='quest_data')
    parser.add_argument('--files_dir',type=str,default='files')
    args = parser.parse_args()
    main(args)