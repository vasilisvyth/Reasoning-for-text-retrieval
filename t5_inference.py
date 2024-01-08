from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from quest.common import example_utils
import argparse
# import faiss
from quest.common import tsv_utils
from collections import defaultdict
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
from quest.common.example_utils import Example
from analyze_retriever import calc_mrec_rec
from prepare_dataset import read_docs, read_queries
from utils import load_pickle, update_results
import os
from prepare_dataset import read_docs
from run_eval import calc_f1_pr_rec

def calculate_queries_scores(args, model, doc_embs, dict_query_ids_queries, doc_title_map, eval_k, all_dids, gold_examples):
    run = defaultdict(list)
    doc_embs = torch.from_numpy(doc_embs)
    Instruction = 'Represent this sentence for searching relevant passages: '
    all_pred_examples = []
    queries = list(dict_query_ids_queries.values())
    qids = list(dict_query_ids_queries.keys())
    for idx in tqdm(
        range(0, len(queries), args.batch_size),
        desc="Encoding queries and search",
        position=0,
        leave=True,
    ):
        batch = queries[idx : idx + args.batch_size]
        tmp_qids = qids[idx : idx + args.batch_size] 
        query_texts = [Instruction+e for e in batch]
        with torch.cuda.amp.autocast(), torch.no_grad():
            batch_query_embs = (
                model.encode(query_texts,normalize_embeddings = args.normalize_embeddings,convert_to_tensor=True)
            )

        similarities = torch.matmul(batch_query_embs, doc_embs.t())
        top_k_values, top_k_indices = torch.topk(similarities, eval_k, dim=1, sorted=True)

        # Creating examples and adding them to the list
        for i, qid in enumerate(tmp_qids):
            query_text = dict_query_ids_queries[qid] # correct query
            pred_docs_ids = [all_dids[index] for index in top_k_indices[i].tolist()]
            doc_texts = [doc_title_map[pr_id] for pr_id in pred_docs_ids]
            scores = top_k_values[i].tolist()

            example = Example(query=query_text, docs=doc_texts, scores=scores)
            all_pred_examples.append(example)
        

    avg_prec, avg_rec, avg_f1 = calc_f1_pr_rec(gold_examples, all_pred_examples)

    avg_scores = {'avg_prec':avg_prec['all'], 'avg_rec':avg_rec['all'], 'avg_f1':avg_f1['all']}
    avg_recall_vals, avg_mrecall_vals, all_rec_per_template = calc_mrec_rec(gold_examples, all_pred_examples)

    avg_recall_vals = {f'avg_R@{key}':value for key, value in avg_recall_vals.items()}
    avg_mrecall_vals = {f'avg_MR@{key}':value for key, value in avg_mrecall_vals.items()}
    INFO = { 'model':'bge-large','checkpoint':'BAAI/bge-large-en-v1.5',
             'K':-1,'threshold':'None'
    }

    path_results_csv = os.path.join('checkpoints','results.csv')
    update_results(path_results_csv, avg_recall_vals, avg_mrecall_vals, all_rec_per_template, avg_scores, INFO)
    # special care is needed for the NOT operator
    k = 10  # Number of results you want
    # scores, indices = index.search(query_vector, k)

    # # Get the k documents with the lowest scores
    # sorted_indices = np.argsort(scores.flatten())
    # top_k_indices = indices[0][sorted_indices]



def print_args(args):
    # Print the values using a for loop
    print("Argument values:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

def main(args):
    # args.split =True
    args.evaluate = True
    print_args(args)
    # args.doc_dir = 'test_query_ids_queries.tsv'
    import os
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device {device}')
    print(torch.__version__)
    print(torch.version.cuda)

    file_path = os.path.join('quest_data', args.doc_dir)

    docs = tsv_utils.read_tsv(file_path)
    
    # tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    model = SentenceTransformer(args.model_name)
    
    # model.to(device)
    if args.embed_docs: 
        docs_embs = []
        true_doc_ids = []
        for idx in tqdm(
            range(0, len(docs), args.batch_size), desc="Encoding documents", position=0, leave=True
        ):
            batch = docs[idx : idx + args.batch_size]
            docs_texts = [e[1] for e in batch] #! I include doc_title in the text
            # print(docs_texts[0])
            true_doc_ids.extend([int(e[0]) for e in batch])

            with torch.no_grad():
                batch_embs = model.encode(docs_texts, normalize_embeddings = args.normalize_embeddings)
                docs_embs.append(batch_embs)


        # Exact Search for Inner Product, full list of indices https://github.com/facebookresearch/faiss/wiki/Faiss-indexes
        # index = faiss.IndexFlatIP(docs_embs[0].shape[1])
        docs_embs = np.concatenate(docs_embs).astype("float32")
        # index.add(docs_embs)
        name=''
        if 'bge' in args.model_name:
            name = 'bge'
        if 'base' in args.model_name:
            name =+ '_base'
        elif 'large' in args.model_name:
            name += '_large'
        # faiss.write_index(index, f'zero_shot_{name}.bin')
        np.save(f'zero_shot_t5_{name}.npy', docs_embs)

        with open(f'doc_ids_zero_shot_{name}.pickle', 'wb') as f:
            pickle.dump(true_doc_ids, f) 

    if args.evaluate:
        print('evaluate')
        id_query = tsv_utils.read_tsv(args.id_query_dir)
        # calculate_queries_scores(id_query, args, model, index, true_doc_ids, device)
        all_dids = load_pickle(os.path.join('checkpoints','bge-zero','doc_ids_zero_shot_bge_large.pickle'))
        doc_embs = np.load(os.path.join('checkpoints','bge-zero','zero_shot_t5_bge_large.npy'))
        data_dir = 'quest_data'
        gold_examples = example_utils.read_examples(os.path.join(data_dir,'test.jsonl'))
        eval_k = 1000
        doc_text_map, doc_title_map = read_docs(os.path.join(data_dir,'doc_text_list.pickle'), os.path.join(data_dir,'doc_title_map.tsv'))
        dict_query_ids_queries, _ = read_queries(os.path.join(data_dir,'test_query_ids_queries.tsv'), 
                                                 os.path.join(data_dir,'test_query_ids_doc_ids.tsv'))
        calculate_queries_scores(args, model, doc_embs, dict_query_ids_queries, doc_title_map, eval_k, all_dids, gold_examples)
    



    else:
        print('not evaluate')

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Ranking with BiEncoder")
    parser.add_argument(
        "--doc_dir", type=str, default="doc_text_map.tsv", help="path to document collection \
                which consists of pairs (idx, document.title + " " + document.text) ")
    parser.add_argument(
        "--id_query_dir", type=str, default='quest_data\\test_query_ids_queries.tsv',help="path to query_id to query text \
                which consists of pairs (idx, query_text) ")
    parser.add_argument(
        "--queries", type=str, default="quest_data\\test_queries.tsv", help="path to queries"
    )
    parser.add_argument(
        "--model_name", type=str, default="BAAI/bge-large-en-v1.5", help="model name"
    )
    parser.add_argument('--evaluate',action='store_true') # default false now!

    parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    parser.add_argument("--normalize_embeddings", action='store_true') # default false now!
    parser.add_argument("--embed_docs", action='store_true') # default false now!
    parser.add_argument(
        "--checkpoint",
        default="output/dense/model",
        type=str,
        help="path to model checkpoint",
    )

    args = parser.parse_args()
    main(args)