from torch.utils.data import Dataset
from quest.common import tsv_utils
import pickle
from quest.common import example_utils
class EvaluateQueryDataset(Dataset):


    def __init__(self, gold_examples, dict_query_ids_queries) -> None:
        super().__init__()
        self.gold_examples = gold_examples

        self.ids, self.input = list(dict_query_ids_queries.keys()), list(dict_query_ids_queries.values())
        self.dict_query_ids_queries = dict_query_ids_queries


    def __len__(self):
        """
        Return the number of inputs i.e. len(queries)
        """
        return len(self.ids)

    def __getitem__(self, idx):

        id = self.ids[idx]
        input = self.input[idx]
        return id, input
    
class EvaluateDocsDataset(Dataset):


    def __init__(self, doc_text_map) -> None:
        super().__init__()
        
        self.ids, self.input =  list(doc_text_map.keys()), list(doc_text_map.values())

    def __len__(self):
        """
        Return the number of inputs i.e. len(docs)
        """
        return len(self.ids)

    def __getitem__(self, idx):

        id = self.ids[idx]
        input = self.input[idx]
        return id, input
