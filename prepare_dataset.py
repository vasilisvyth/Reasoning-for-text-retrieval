from quest.common import tsv_utils
import pickle

def remove_complex_queries(examples, dict_query_ids_queries):
    for example_instance in examples:
        if example_instance.metadata.template != '_':
            for query_id, query_text in dict_query_ids_queries.items():
                if query_text == example_instance.query:
                    del dict_query_ids_queries[query_id]
                    break 

def read(query_ids_queries_dir, query_ids_doc_ids_dir, doc_text_list_dir):
    query_ids_queries = tsv_utils.read_tsv(query_ids_queries_dir)
    dict_query_ids_queries = dict(query_ids_queries)
    # convert to int keys query_ids
    dict_query_ids_queries = {int(key): value for key, value in dict_query_ids_queries.items()}

    #  query_ids_doc_ids they are strings
    query_ids_doc_ids = tsv_utils.read_tsv(query_ids_doc_ids_dir) # list of lists of the form [query_id, doc_id]

    #  doc_ids_documents
    with open(doc_text_list_dir, 'rb') as f:
        doc_text_list = pickle.load(f)
    # the keys already are ints
    doc_text_map = dict(doc_text_list) # int doc_id: string representing doc_title + doc_text
    return dict_query_ids_queries, query_ids_doc_ids, doc_text_map

def build_positive_pairs(query_ids_queries_dir, query_ids_doc_ids_dir, doc_text_list_dir, use_complex_queries, examples):

    
    # Remove queries with metadata!= '_' from the dictionary
    if not use_complex_queries:
        print(f'Total queries before removing complex queries: {len(dict_query_ids_queries)}')
        remove_complex_queries(examples, dict_query_ids_queries)
        print(f'Total queries after removing complex queries: {len(dict_query_ids_queries)}')


    query_ids_with_duplicates = []
    doc_ids_with_duplicates = []
    queries_with_duplicates = []
    docs_with_duplicates = []
    for query_id, doc_id in query_ids_doc_ids: # for every query_id, relevant_doc_id
        query_id, doc_id = int(query_id), int(doc_id)
        if query_id in dict_query_ids_queries:
            query = dict_query_ids_queries[query_id]
            doc = doc_text_map[doc_id]

            query_ids_with_duplicates.append(query_id)
            doc_ids_with_duplicates.append(doc_id)
            queries_with_duplicates.append(query)
            docs_with_duplicates.append(doc)
    
    return query_ids_with_duplicates, doc_ids_with_duplicates, queries_with_duplicates, docs_with_duplicates

def build_all_possible_pairs(query_ids_queries_dir, doc_text_list_dir, use_complex_queries, examples):
    
    query_ids_queries = tsv_utils.read_tsv(query_ids_queries_dir)
    dict_query_ids_queries = dict(query_ids_queries)
    # convert to int keys query_ids
    dict_query_ids_queries = {int(key): value for key, value in dict_query_ids_queries.items()}

    #  doc_ids_documents
    with open(doc_text_list_dir, 'rb') as f:
        doc_text_list = pickle.load(f)

    query_ids = []
    doc_ids = []
    queries = []
    docs = []
    for query_id, query in query_ids_queries:
        query_id = int(query_id)

        for doc_id, doc in doc_text_list:
            # doc_id = int(doc_id)

            # doc = doc_text_map[doc_id]

            query_ids.append(query_id)
            doc_ids.append(doc_id)
            queries.append(query)
            docs.append(doc)

    return query_ids, doc_ids, queries, docs