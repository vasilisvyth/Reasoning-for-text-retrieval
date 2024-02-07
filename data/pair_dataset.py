from torch.utils.data import Dataset
from quest.common import tsv_utils
import pickle
from quest.common import example_utils
class PairDataset(Dataset):

    def __init__(self, query_ids, doc_ids, queries, docs, gold_examples) -> None:
        super().__init__()
        self.query_ids, self.doc_ids, self.queries, self.docs = query_ids, doc_ids, queries, docs
        self.gold_examples = gold_examples
        # #! they are shuffled
        # query_docs = tsv_utils.read_tsv(queries_rel_docs_path) # (query, relevant_doc_text)
        # self.collection = dict(self.collection) # key is string

        # self.queries = read_pairs(queries_path)
        # self.queries = dict(self.queries) # key is string

    def __len__(self):
        """
        Return the number of pairs to re-rank
        """
        return len(self.query_ids)

    def __getitem__(self, idx):
        """
        Return the idx-th pair of the dataset in the format of (query_id, doc_id, query_text, doc_text)
        """
        query_id = self.query_ids[idx]
        doc_id = self.doc_ids[idx]
        query_text = self.queries[idx]
        doc_text = self.docs[idx]
        return query_id, doc_id, query_text, doc_text