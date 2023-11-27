from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import argparse
# import faiss
from quest.common import tsv_utils
from collections import defaultdict
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle

def calculate_queries_scores(queries, args, model, index, true_doc_ids, device):
    run = defaultdict(list)
    queries_embs = []
    for idx in tqdm(
        range(0, len(queries), args.batch_size),
        desc="Encoding queries and search",
        position=0,
        leave=True,
    ):
        batch = queries[idx : idx + args.batch_size]
        query_texts = [e[1] for e in batch]
        with torch.cuda.amp.autocast(), torch.no_grad():
            batch_query_embs = (
                model.encode(query_texts).astype("float32")
            )
        scores, docs_index_id = index.search(batch_query_embs, 1000)
        for idx in range(len(batch)):
            query_id = batch[idx][0]
            for i, score in zip(docs_index_id[idx], scores[idx]):
                if i < 0:
                    continue
                doc_id = true_doc_ids[i]
            run[query_id].append((doc_id, score))

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
    print_args(args)
    args.doc_dir = 'test_query_ids_queries.tsv'
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
            batch_embs = model.encode(docs_texts)
            docs_embs.append(batch_embs)


    # Exact Search for Inner Product, full list of indices https://github.com/facebookresearch/faiss/wiki/Faiss-indexes
    # index = faiss.IndexFlatIP(docs_embs[0].shape[1])
    docs_embs = np.concatenate(docs_embs).astype("float32")
    # index.add(docs_embs)
    name=''
    if 'base' in args.model_name:
        name = 'base'
    elif 'large' in args.model_name:
        name = 'large'
    # faiss.write_index(index, f'zero_shot_{name}.bin')
    np.save(f'zero_shot_t5_{name}.npy', docs_embs)

    with open(f'doc_ids_zero_shot_{name}.pickle', 'wb') as f:
      pickle.dump(true_doc_ids, f) 

    if args.evaluate:
        print('evaluate')
        id_query = tsv_utils.read_tsv(args.id_query_dir)
        # calculate_queries_scores(id_query, args, model, index, true_doc_ids, device)
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
        "--model_name", type=str, default="sentence-transformers/gtr-t5-base", help="model name"
    )
    parser.add_argument('--evaluate',action='store_true') # default false now!
    parser.add_argument(
        "--tokenizer_name", type=str, default="sentence-transformers/gtr-t5-base", help="tokenizer name"
    )
    parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    parser.add_argument(
        "--checkpoint",
        default="output/dense/model",
        type=str,
        help="path to model checkpoint",
    )
    parser.add_argument(
        "--o",
        type=str,
        default="output/dense/test_run.trec",
        help="path to output run file",
    )
    args = parser.parse_args()
    main(args)