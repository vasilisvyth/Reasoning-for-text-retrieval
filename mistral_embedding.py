import torch
import torch.nn.functional as F
import random
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, DataCollatorWithPadding
from prepare_dataset import read_docs, read_queries
import os
from datasets.evaluate_dataset import  EvaluateDocsDataset, tokenize_docs
import argparse
import numpy as np
from functools import partial
from datasets import Dataset #pip install datasets
from seeds import set_seed
from utils import print_args, bytes_to_giga_bytes,load_pickle, create_pickle
import gc

def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def get_detailed_query_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'

def build_docs_from_rand_qids(random_query_ids, test_dict_query_ids_queries, test_query_ids_doc_ids, doc_text_map):
    rand_doc_text_map = {}
    for qid, doc_id in test_query_ids_doc_ids:
        qid, doc_id = int(qid), int(doc_id)
        if qid not in random_query_ids: continue
        query = test_dict_query_ids_queries[qid]
        doc = doc_text_map[doc_id]

        rand_doc_text_map[doc_id] = doc
            
    return rand_doc_text_map

def main(args):
    # docs = np.load('rand_zero_shot_mistral.npy')
    # ids = load_pickle('rand_doc_ids.pickle')
    # args.model_name  = 'gpt2'
    print_args(args)
    set_seed(args.seed)
    data_dir = 'quest_data'
    doc_text_map, doc_title_map = read_docs(os.path.join(data_dir,'doc_text_list.pickle'), os.path.join(data_dir,'doc_title_map.tsv'))
    test_dict_query_ids_queries, test_query_ids_doc_ids = read_queries(os.path.join(args.data_dir,'test_query_ids_queries.tsv'), 
                                                 os.path.join(args.data_dir,'test_query_ids_doc_ids.tsv'))
    random_query_ids = [random.randint(0, len(test_dict_query_ids_queries)-1) for _ in range(220)]
    create_pickle(random_query_ids,'rand_qids.pickle')
    
    rand_doc_text_map = build_docs_from_rand_qids(random_query_ids, test_dict_query_ids_queries, test_query_ids_doc_ids, doc_text_map)
    doc_ids = list(rand_doc_text_map.keys())
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    docs_dataset = Dataset.from_dict({'input_texts': list(rand_doc_text_map.values())})
    docs_dataset.set_transform(partial(tokenize_docs, tokenizer))

    # docs_dataset = EvaluateDocsDataset(rand_doc_text_map, tokenizer)
    collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    

    # # Tokenize the input texts
    batch_dict = tokenizer(list(rand_doc_text_map.values())[:10], max_length=args.max_length - 1, return_attention_mask=False, padding=False, truncation=True)
    # append eos_token_id to every input_ids
    batch_dict['input_ids'] = [input_ids + [tokenizer.eos_token_id] for input_ids in batch_dict['input_ids']]
    # batch_dict = tokenizer.pad(batch_dict, padding=True, return_attention_mask=True, return_tensors='pt') # this is left padding automatically
  
    #exit()
    model = AutoModel.from_pretrained(args.model_name) #
    if args.better_transformer:
        model = model.to_bettertransformer()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device {device}')

    

    # model.resize_token_embeddings(model.config.vocab_size + 1) # because we added pad_token
    
   
    print('before inference')
    # with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=False, enable_mem_efficient=False):
    max_retries = 5  # Set the maximum number of retries
    with torch.no_grad():
        for _ in range(max_retries):
            dataloader = DataLoader(docs_dataset,batch_size=args.batch_size, shuffle=False,collate_fn=collator) # num_workers=2
            model.to(device) # not needed for 4bit and 8bit
            all_dids = []
            encoded_embeds = [] 
            try:
                for i, batch in enumerate(tqdm(dataloader)):
                    #assert(batch_dict['input_ids'][0] == batch['input_ids'][0][12:].tolist())
                    dids = doc_ids[i:i+args.batch_size]
                    # dids = batch.pop('ids')
                    batch = {key: batch[key].to(device) for key in batch}
                    # tokenizer.batch_decode(input_ids)[0]
                    
                    # generated_ids = model.generate(input_ids, attention_mask = attention_mask,max_new_tokens=max_new_tokens)
                    outputs = model(**batch)
                    embeddings = last_token_pool(outputs.last_hidden_state, batch['attention_mask'])
                    encoded_embeds.append(embeddings.cpu().numpy())

                    
                    all_dids.extend(dids)
                    if i > 4:
                        break

                    print('GB ',bytes_to_giga_bytes(torch.cuda.max_memory_allocated()))

                encoded_embeds = np.concatenate(encoded_embeds, axis=0)

                np.save(f'rand_zero_shot_mistral.npy', encoded_embeds)
                create_pickle(doc_ids,'rand_doc_ids.pickle')
                print('finished encoding...')
                print(f'BATCH SIZE {args.batch_size}')
                break
            except RuntimeError as e:
                print(e)
                    
                # Reduce batch size
                args.batch_size //= 2
                print(f"Reduced batch size to {args.batch_size}")

                # Clear CUDA memory
                torch.cuda.empty_cache()
                gc.collect()
    # # normalize embeddings
    # embeddings = F.normalize(embeddings, p=2, dim=1)
    # scores = (embeddings[:2] @ embeddings[2:].T) * 100
    # print(scores.tolist())

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Generate programs using open source programs')
    parser.add_argument('--test_file', type=str, default="test.jsonl")
    parser.add_argument('--data_dir', type=str, default="quest_data")
    parser.add_argument('--model_name', type=str, default="intfloat/e5-mistral-7b-instruct")
    parser.add_argument('--tokenizer', type=str, default="intfloat/e5-mistral-7b-instruct")
    parser.add_argument('--seed',type=int, default=0)
    parser.add_argument('--batch_size',type=int, default=8)
    parser.add_argument('--max_length',type=int, default=512)
    parser.add_argument('--bit8',action='store_true') # default false now!
    parser.add_argument('--bit4',action='store_true') # default false now!
    parser.add_argument('--better_transformer',action='store_true') # default false now!
    parser.add_argument('--use_flash_attention_2',action='store_true') # default false now!
    args = parser.parse_args()
    main(args)