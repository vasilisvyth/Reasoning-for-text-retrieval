
from sentence_transformers import SentenceTransformer
import numpy as np
import os
from data.prepare_dataset import read_docs
import torch
from tools import CustomDictOperations
from transformers import AutoTokenizer
from bi_encoder import DenseBiEncoder
from seeds import set_seed
import transformers
import json
from tqdm import tqdm
from utils import current_program, preprocess_find_docs_call,synthesize_program
from quest.common import document_utils
from tools import DocumentFinder


def test_zero_shot_gtr_docs():
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
        embed_doc = model.encode_mean_polling(input_ids, attention_mask)
    
    print(torch.max(torch.abs(docs[doc_id]-embed_doc[doc_id])))
    print(transformers.__version__)
 

def test_program_check():
    file = open(os.path.join('data','predictions.json'), "r")
    documents = document_utils.read_documents("quest_data\\documents.jsonl")
    results_json = json.load(file)
    for query in tqdm(results_json):
        result = results_json[query]['pred']
        new_result = preprocess_find_docs_call(query, result)
        program = synthesize_program(result = new_result, prefix = '')
        
        template_used = results_json[query]['template']
        sorted_pred_doc_titles, var_dict = current_program(program, documents, query, template_used)

        custom_dict_0 = DocumentFinder.find_docs(query,var_dict['question_0'])
        custom_dict_1 = DocumentFinder.find_docs(query,var_dict['question_1'])
        
        assert(custom_dict_0.data ==var_dict['docs_0'].data)
        assert(custom_dict_1.data ==var_dict['docs_1'].data)
        break
   

def test_replace_rows():
    import torch


    original_tensor = torch.randn(5, 2)

    replacement_tensor = torch.randn(3, 2)

    # List of indices representing rows to be changed
    indices_to_change = [2, 4, 0]

    # Convert the list of indices to a tensor
    indices_tensor = torch.tensor(indices_to_change)

    # Replace the corresponding rows in the original tensor
    original_tensor[indices_to_change] = replacement_tensor

    assert(torch.all(replacement_tensor[0] ==original_tensor[indices_tensor[0]]))
    assert(torch.all(replacement_tensor[1] ==original_tensor[indices_tensor[1]]))
    assert(torch.all(replacement_tensor[2] ==original_tensor[indices_tensor[2]]))
