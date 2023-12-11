from python_interpreter import find_logical_operators
from sentence_transformers import SentenceTransformer
import numpy as np
import os
from prepare_dataset import read_docs
import torch
from tools import CustomDictOperations
from transformers import AutoTokenizer
from bi_encoder import DenseBiEncoder
from seeds import set_seed
import transformers

def test_logical_operators():
    program = 'A and not B'
    find_logical_operators(program)

def zero_shot_gtr_docs():
    model = SentenceTransformer('sentence-transformers/gtr-t5-large')
    path_doc_text_list = os.path.join('quest_data','doc_text_list.pickle')
    path_doc_title_map = os.path.join('quest_data','doc_title_map.tsv')
    doc_text_map, doc_title_map = read_docs(path_doc_text_list, path_doc_title_map)
    docs = np.load('checkpoints/zero_shot_t5_large.npy')
    with torch.no_grad():
        embed_doc = model.encode(doc_text_map[0])
    assert(np.allclose(embed_doc,docs[0], atol=10**(-7)))

def test_and_operator():
    A_map_ind_values = {1:10,2:5,3:-1}#{ind:val for ind, val in zip(top_k_indices, top_k_values)}
    A_custom_dict = CustomDictOperations(A_map_ind_values)

    B_map_ind_values = {2:2,1:-1}#{ind:val for ind, val in zip(top_k_indices, top_k_values)}
    B_custom_dict = CustomDictOperations(B_map_ind_values)
    dict_result = A_custom_dict & B_custom_dict
    # currently use min for AND
    assert(dict_result.data['indices'] == [1,2])
    assert(dict_result.data['scores'] == [-1,2])

def finetuned_embed_docs():
    set_seed(0)
    tokenizer = AutoTokenizer.from_pretrained('google/t5-v1_1-base')
    model = DenseBiEncoder('checkpoints/da0656a7-f3/checkpoint-13500', False, False)
    docs = torch.load('checkpoints/da0656a7-f3/scores_docs_check_0.pt')
    # load docs
    path_doc_text_list = os.path.join('quest_data','doc_text_list.pickle')
    path_doc_title_map = os.path.join('quest_data','doc_title_map.tsv')
    doc_text_map, _ = read_docs(path_doc_text_list, path_doc_title_map)

    doc_id = 0

    tokenized_txt = tokenizer.encode_plus(doc_text_map[doc_id], return_tensors='pt',max_length=512,truncation=True)
    input_ids, attention_mask = tokenized_txt['input_ids'], tokenized_txt['attention_mask']
    with torch.no_grad():
        embed_doc = model.encode(input_ids, attention_mask)
    
    print(torch.max(torch.abs(docs[doc_id]-embed_doc[doc_id])))
    print(transformers.__version__)
    a=1

print('finetuned embed docs')
finetuned_embed_docs()
test_and_operator()
zero_shot_gtr_docs()
test_logical_operators()