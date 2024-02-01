import torch
import torch.nn.functional as F
import random
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, DataCollatorWithPadding
from data.prepare_dataset import read_docs, read_queries
import os
from data.evaluate_dataset import  EvaluateDocsDataset, tokenize
import argparse
import numpy as np
from quest.common import example_utils
from functools import partial
from datasets import Dataset #pip install datasets
from seeds import set_seed
from utils import print_args, bytes_to_giga_bytes,load_pickle, create_pickle
import gc
from templates.mistral_prompt import Instruction, templates_dict
import cProfile

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

def build_docs(doc_text_map, rand_dids):

    rand_docs = []
    for id in rand_dids:
        doc = doc_text_map[id]
        rand_docs.append(doc)

    return rand_docs

def create_embeddings(dataloader, rand_dids, device, model):
    all_dids = []
    last_layer_embeds = [] 
    all_layers_embed = []
    model.to(device) # not needed for 4bit and 8bit
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            #assert(batch_dict['input_ids'][0] == batch['input_ids'][0][12:].tolist())
            dids = rand_dids[i:i+args.batch_size]
            # dids = batch.pop('ids')
            batch = {key: batch[key].to(device) for key in batch}
            # tokenizer.batch_decode(input_ids)[0]
            # generated_ids = model.generate(input_ids, attention_mask = attention_mask,max_new_tokens=max_new_tokens)
            outputs = model(**batch)
            embeddings = last_token_pool(outputs.last_hidden_state, batch['attention_mask'])
            last_layer_embeds.append(embeddings.cpu().numpy())

            cur_all_embed = [hidden_state[:,-1].cpu() for hidden_state in outputs.hidden_states] # len(layers) each torch size[batch_size, hidden_dim]
            cur_all_embed = torch.stack(cur_all_embed)
            all_layers_embed.append(cur_all_embed)
            
            all_dids.extend(dids)
            if i > 6: break
            

        last_layer_embeds = np.concatenate(last_layer_embeds, axis=0)
        
        # # Step 1: Stack each element in the inner lists along the second dimension
        # stacked_batches = [torch.stack(batch, dim=1) for batch in all_layers_embed]

        # # Step 2: Concatenate the batches along the first dimension
        # final_tensor = torch.cat(stacked_batches, dim=0)


    return last_layer_embeds, all_layers_embed

def build_queries(test_dict_query_ids_queries, random_query_ids, instruction):
    rand_queries = []
    for qid in random_query_ids:
        q = test_dict_query_ids_queries[qid]
        q = get_detailed_query_instruct(instruction,q)
        rand_queries.append(q)
    return rand_queries

def create_dataset_docs(doc_text_map, tokenizer, rand_ids):
    
    rand_doc_text = build_docs(doc_text_map, rand_ids)
    dataset = Dataset.from_dict({'input_texts': rand_doc_text})
    dataset.set_transform(partial(tokenize, tokenizer, 512))
    input_type = 'docs'
    return dataset, input_type, rand_ids

def create_dataset_queries(tokenizer, instruct):
    instruction = Instruction[instruct]
    test_dict_query_ids_queries, test_query_ids_doc_ids = read_queries(os.path.join(args.data_dir,'test_query_ids_queries.tsv'), 
                                                 os.path.join(args.data_dir,'test_query_ids_doc_ids.tsv'))
    rand_qids_path = os.path.join('files','rand_qids.pickle')
    rand_ids = load_pickle(rand_qids_path)
    rand_queries = build_queries(test_dict_query_ids_queries, rand_ids, instruction)
    dataset = Dataset.from_dict({'input_texts': rand_queries})
    dataset.set_transform(partial(tokenize, tokenizer, 64))
    input_type = f'query_{instruct}_'
    return dataset, input_type, rand_ids

def create_dataset_subqueries(tokenizer, templates_index):
    examples_with_subquestions = load_pickle(os.path.join('files','ex_with_subq_mistral_rand_docs_anon.json.pickle'))
    
    path_test = os.path.join(args.data_dir,'test.jsonl')
    gold_examples = example_utils.read_examples(path_test)

    all_inputs = []
    all_ex_ids = []
    rand_qids_path = os.path.join('files','rand_qids.pickle')
    rand_ids = load_pickle(rand_qids_path) # includes 2 questions which have bugs so they do not have any subquestions
    txt_text = ''
    for ex in examples_with_subquestions:
        template = ex.example.metadata.template
        subquestions = [f'\nQuery {i+1}: '+subq.replace('find ','') for i, subq in enumerate(ex.subquestions)]
        subquestions = ''.join(subquestions)
        instruction_id = templates_index[template]
        if template == '_ that are also _':
            txt_text += f'\n {ex.example.query}'
            txt_text += subquestions
            txt_text += '\n'

        instruction = f'Instruct: {templates_dict[template][instruction_id]}'
        original_question = f' The initial question is: {ex.example.query}'
        mistral_input = instruction + subquestions + original_question
        all_inputs.append(mistral_input)
        
        for ex_id in rand_ids:
            if gold_examples[ex_id].query == ex.example.query:
                all_ex_ids.append(ex_id)
                break
    
    with open('rand_and_with_instruction.txt', 'w') as file:
        file.write(txt_text)

    create_pickle(all_ex_ids, os.path.join('files','rand_qids_without_bugs.pickle'))
    dataset = Dataset.from_dict({'input_texts': all_inputs})
    dataset.set_transform(partial(tokenize, tokenizer, 256))
    input_type = f'query_decomp_and_initial'
    return dataset, input_type, all_ex_ids

def main(args):
    # a = np.load('dok.npz')
    # docs = np.load('rand_zero_shot_mistral.npy')
    # ids = load_pickle('rand_doc_ids.pickle')
    args.output_hidden_states = True
    args.encode_all_docs = True
    # args.model_name= 'gpt2'
    print_args(args)
    set_seed(args.seed)
    data_dir = 'quest_data'
    
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device {device}')

    if args.encode_rand_docs:
        print('Encode rand docs...')
        doc_text_map, doc_title_map = read_docs(os.path.join(data_dir,'doc_text_list.pickle'), os.path.join(data_dir,'doc_title_map.tsv'))
        rand_dids_path = os.path.join('files','with_true_bge_large__0_rand_ids_top_25.pickle')
        ids = load_pickle(rand_dids_path)
        dataset, input_type, rand_ids = create_dataset_docs(doc_text_map, tokenizer, ids)

    if args.encode_all_docs:
        doc_text_map, doc_title_map = read_docs(os.path.join(data_dir,'doc_text_list.pickle'), os.path.join(data_dir,'doc_title_map.tsv'))

        args.doc_end = len(doc_text_map) if args.doc_end == 0 else args.doc_end
        print(f'encode docs from {args.doc_start} to {args.doc_end}')
        ids = [i for i in range(args.doc_start, args.doc_end,1)]
        dataset, input_type, rand_ids = create_dataset_docs(doc_text_map, tokenizer, ids)
        input_type = input_type +f'_{args.doc_start}_{args.doc_end}'

    if args.encode_queries:
        print('Encode queries...')
        dataset, input_type, rand_ids = create_dataset_queries(tokenizer, args.instruct)
    if args.encode_decomposition:
        print('Question decomposition')
        dataset, input_type, rand_ids = create_dataset_subqueries(tokenizer, templates_index = {
            '_ that are also _ but not _':0,
            '_':0,
            '_ or _ or _':0,
            '_ that are not _':0,
            '_ that are also _':0,
            '_ that are also both _ and _':0,
            '_ or _':0
        })

    # docs_dataset = EvaluateDocsDataset(rand_doc_text_map, tokenizer)
    
    

    # # Tokenize the input texts
    # batch_dict = tokenizer(rand_doc_text[:10], max_length=args.max_length - 1, return_attention_mask=False, padding=False, truncation=True)
    # # append eos_token_id to every input_ids
    # batch_dict['input_ids'] = [input_ids + [tokenizer.eos_token_id] for input_ids in batch_dict['input_ids']]
    # batch_dict = tokenizer.pad(batch_dict, padding=True, return_attention_mask=True, return_tensors='pt') # this is left padding automatically
  
    #exit()
   

    # model.resize_token_embeddings(model.config.vocab_size + 1) # because we added pad_token    
    model = AutoModel.from_pretrained(args.model_name, output_hidden_states=args.output_hidden_states) #
    if args.better_transformer:
        model = model.to_bettertransformer()
    print('before inference')
    #! no shuffle
    dataloader = DataLoader(dataset,batch_size=args.batch_size, shuffle=False,collate_fn=collator)#, num_workers=2, pin_memory=True)
    
   
    encoded_embeds, all_layers_embed = create_embeddings(dataloader, rand_ids, device, model)
    
    all_layers_embed = np.savez(f'{input_type}_all_zero_shot_mistral',torch.stack(all_layers_embed).numpy())
    print('finished saving all layers')

    embed_path = os.path.join('files',f'{input_type}_last_zero_shot_mistral.npy')
    np.save(embed_path, encoded_embeds)
    print('finished encoding...')

    # create_pickle(all_layers_embed, os.path.join('files',f'{input_type}_zero_shot_mistral.pickle'))

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Generate programs using open source programs')
    parser.add_argument('--test_file', type=str, default="test.jsonl")
    parser.add_argument('--data_dir', type=str, default="quest_data")
    parser.add_argument('--model_name', type=str, default="intfloat/e5-mistral-7b-instruct")
    parser.add_argument('--tokenizer', type=str, default="intfloat/e5-mistral-7b-instruct")
    parser.add_argument('--instruct', type=str, default="instr_2")
    parser.add_argument('--seed',type=int, default=0)
    parser.add_argument('--batch_size',type=int, default=8)
    parser.add_argument('--doc_start',type=int, default=0)
    parser.add_argument('--doc_end',type=int, default=0)
    parser.add_argument('--max_length',type=int, default=512) 
    parser.add_argument('--encode_rand_docs',action='store_true') # default false now!
    parser.add_argument('--output_hidden_states', action='store_true') # default false now!
    parser.add_argument('--encode_all_docs',action='store_true') # default false now!
    parser.add_argument('--encode_queries',action='store_true') # default false now!
    parser.add_argument('--encode_decomposition',action='store_true') # default false now!
    parser.add_argument('--bit8',action='store_true') # default false now!
    parser.add_argument('--bit4',action='store_true') # default false now!
    parser.add_argument('--better_transformer',action='store_true') # default false now!
    parser.add_argument('--use_flash_attention_2',action='store_true') # default false now!
    args = parser.parse_args()
    main(args)