from torch.utils.data import Dataset
from quest.common import tsv_utils
import pickle

class PairDataset(Dataset):

    def __init__(self, query_ids_queries_dir, query_ids_doc_ids_dir, doc_text_list_dir) -> None:
        super().__init__()
        # query_ids_queries,
        query_ids_queries = tsv_utils.read_tsv(query_ids_queries_dir)
        dict_query_ids_queries = dict(query_ids_queries)
        # convert to int keys query_ids
        self.dict_query_ids_queries = {int(key): value for key, value in dict_query_ids_queries.items()}

        #  query_ids_doc_ids they are strings
        query_ids_doc_ids = tsv_utils.read_tsv(query_ids_doc_ids_dir) # list of lists of the form [query_id, doc_id]

        #  doc_ids_documents
        with open(doc_text_list_dir, 'rb') as f:
            doc_text_list = pickle.load(f)
        # the keys already are ints
        self.doc_text_map = dict(doc_text_list) # int doc_id: string representing doc_title + doc_text
        
        self.query_ids_with_duplicates = []
        self.doc_ids_with_duplicates = []
        self.queries_with_duplicates = []
        self.docs_with_duplicates = []
        for query_id, doc_id in query_ids_doc_ids:
            query_id, doc_id = int(query_id), int(doc_id)
            query = self.dict_query_ids_queries[query_id]
            doc = self.doc_text_map[doc_id]

            self.query_ids_with_duplicates.append(query_id)
            self.doc_ids_with_duplicates.append(doc_id)
            self.queries_with_duplicates.append(query)
            self.docs_with_duplicates.append(doc)
        
        a=1
        # #! they are shuffled
        # query_docs = tsv_utils.read_tsv(queries_rel_docs_path) # (query, relevant_doc_text)
        # self.collection = dict(self.collection) # key is string

        # self.queries = read_pairs(queries_path)
        # self.queries = dict(self.queries) # key is string

    def __len__(self):
        """
        Return the number of pairs to re-rank
        """
        return len(self.query_ids_with_duplicates)

    def __getitem__(self, idx):
        """
        Return the idx-th pair of the dataset in the format of (query_id, doc_id, query_text, doc_text)
        """
        query_id = self.query_ids_with_duplicates[idx]
        doc_id = self.doc_ids_with_duplicates[idx]
        query_text = self.queries_with_duplicates[idx]
        doc_text = self.docs_with_duplicates[idx]
        return query_id, doc_id, query_text, doc_text