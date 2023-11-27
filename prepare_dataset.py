from quest.common import tsv_utils
from quest.common.example_utils import Example, ExampleMetadata
import pickle
from sklearn.model_selection import train_test_split
import random
from copy import deepcopy
EXTRA_SIMPLE_VAL_EX = 250

def remove_complex_queries(examples, dict_query_ids_queries):
    copied_examples = deepcopy(examples)
    for example_instance in copied_examples:
        if example_instance.metadata.template != '_':
            for query_id, query_text in dict_query_ids_queries.items():
                if query_text == example_instance.query:
                    del dict_query_ids_queries[query_id]
                    examples.remove(example_instance)
                    break 


def read_queries(query_ids_queries_dir, query_ids_doc_ids_dir):
    '''
    dict_query_ids_queries: dict of the form query_id:query_text
    query_ids_doc_ids: list of list of [query_id, doc_id] 
    '''
    
    query_ids_queries = tsv_utils.read_tsv(query_ids_queries_dir)
    dummy_id = []
    dummy_q = []
    c=0
    for qid, query in query_ids_queries:
        if int(qid) in dummy_id:
            c+=1
        dummy_id.append(int(qid))
        dummy_q.append(query)

    # dict_query_ids_queries = dict(query_ids_queries)
    # convert to int keys query_ids
    dict_query_ids_queries = {int(qid): query for qid, query in query_ids_queries}
    assert(len(query_ids_queries) == len(dict_query_ids_queries))
    #  query_ids_doc_ids they are strings
    query_ids_doc_ids = tsv_utils.read_tsv(query_ids_doc_ids_dir) # list of lists of the form [query_id, doc_id]
    
    return dict_query_ids_queries, query_ids_doc_ids

def filter_data(sub_qids,initial_dict_query_ids_queries, initial_train_query_ids_doc_ids, split_str, examples):
    shifted_qid = 1000 if split_str == 'val' else 0 # avoid overlap with qid from validation as it may create problem in the dictionary
    # Filter train_dict_query_ids_queries dict random order now
    train_dict_query_ids_queries_filtered = {qid+shifted_qid: initial_dict_query_ids_queries[qid] for qid in sub_qids}

    # Filter train_query_ids_doc_ids
    train_query_ids_doc_ids_filtered = [[int(pair[0])+shifted_qid, pair[1]] for pair in initial_train_query_ids_doc_ids if int(pair[0]) in sub_qids]
    
    train_queries_filtered = list(train_dict_query_ids_queries_filtered.values())

    filtered_examples = [example_instance for example_instance in examples if example_instance.query in train_queries_filtered]
    
    return train_dict_query_ids_queries_filtered, train_query_ids_doc_ids_filtered, filtered_examples

def split(dict_query_ids_queries, query_ids_doc_ids, examples_train):
    '''
    I want to keep from train_dict_query_ids_queries and train_query_ids_doc_ids the relevant train_qids
    
    dict_query_ids_queries: dict of the form query_id:query_text
    query_ids_doc_ids: list of list of [str(query_id), str(doc_id)] 
    '''
    qids = list(dict_query_ids_queries.keys())
    random.seed(0)
    random.shuffle(qids)
    train_qids = qids[:-EXTRA_SIMPLE_VAL_EX]
    val_qids = qids[-EXTRA_SIMPLE_VAL_EX:]

    # train_dict_query_ids_queries_filtered, train_query_ids_doc_ids_filtered, filtered_train_examples = filter_data(train_qids, dict_query_ids_queries, query_ids_doc_ids,'train', examples_train)
    # val_dict_query_ids_queries_filtered, val_query_ids_doc_ids_filtered, filtered_val_examples = filter_data(val_qids, dict_query_ids_queries, query_ids_doc_ids,'val', examples_train)

    train_data  = filter_data(train_qids, dict_query_ids_queries, query_ids_doc_ids,'train', examples_train)
    val_data  = filter_data(val_qids, dict_query_ids_queries, query_ids_doc_ids,'val', examples_train)

    return (*train_data, *val_data)
    # return train_dict_query_ids_queries_filtered, train_query_ids_doc_ids_filtered, val_dict_query_ids_queries_filtered, val_query_ids_doc_ids_filtered, filtered_train_examples, filtered_val_examples

def read_docs(doc_text_list_dir, doc_title_map_dir):
    #  doc_ids_documents
    with open(doc_text_list_dir, 'rb') as f:
        doc_text_list = pickle.load(f)
    # the keys already are ints
    doc_text_map = dict(doc_text_list) # int doc_id: string representing doc_title + doc_text
    assert(len(doc_text_list) == len(doc_text_map))
    doc_title_map = tsv_utils.read_tsv(doc_title_map_dir)
    doc_title_map = dict(doc_title_map)
    int_doc_title_map = {int(key):value for key, value in doc_title_map.items()}
    assert(len(int_doc_title_map)) == len(int_doc_title_map)
    return doc_text_map, int_doc_title_map

def build_positive_pairs(dict_query_ids_queries, query_ids_doc_ids, doc_text_map):
    '''
    Args:
        dict_query_ids_queries: dict of the form query_id: query
        query_ids_doc_ids: list of list of the form [str(query_id), str(doc_id)]
        doc_text_map: dict of the form doc_id : doc
        use_complex_queries: bool
        examples: list of Examples
    '''
    
    positive_query_ids = []
    positive_doc_ids = []
    positive_queries = []
    positive_docs = []
    for query_id, doc_id in query_ids_doc_ids: # for every query_id, relevant_doc_id
        query_id, doc_id = int(query_id), int(doc_id)
        if query_id in dict_query_ids_queries: # if it not removed
            query = dict_query_ids_queries[query_id]
            doc = doc_text_map[doc_id]

            positive_query_ids.append(query_id)
            positive_doc_ids.append(doc_id)
            positive_queries.append(query)
            positive_docs.append(doc)
    
    return positive_query_ids, positive_doc_ids, positive_queries, positive_docs

def test_positive_pairs():
    dict_query_ids_queries = {0:'0 q',12:'12 q'}
    query_ids_doc_ids = [['0','10'],['0','12'],['12','10']]
    doc_text_map = {12:'12 doc',10:'10 doc', 99:'99 doc'}
    use_complex_queries = True
    examples = []
    
    positive_query_ids, positive_doc_ids, positive_queries, positive_docs = \
            build_positive_pairs(dict_query_ids_queries, query_ids_doc_ids, doc_text_map)
    assert(positive_query_ids==[0,0,12])
    assert(positive_doc_ids==[10,12,10])
    assert(positive_queries==['0 q','0 q','12 q'])
    assert(positive_docs == ['10 doc','12 doc','10 doc'])
    
    examples = [
                Example(query='0 q',docs=[11],metadata=ExampleMetadata(template='_')),
                Example(query='12 q',docs=[11],metadata=ExampleMetadata(template='or'))]
    

    train_dict_query_ids_queries_filtered, train_query_ids_doc_ids_filtered,\
    val_dict_query_ids_queries_filtered, val_query_ids_doc_ids_filtered, \
    filtered_train_examples, filtered_val_examples = split(dict_query_ids_queries, query_ids_doc_ids, examples)
    qids= [0,12]
    assert(val_dict_query_ids_queries_filtered == {1000+0:'0 q',1000+12:'12 q'})
    assert(val_query_ids_doc_ids_filtered == [[1000,'10'],[1000,'12'],[1012,'10']])
    assert(filtered_val_examples == examples)
    a=1

# test_positive_pairs()