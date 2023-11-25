from torch.utils.data import Dataset
from quest.common import tsv_utils
import pickle
from quest.common import example_utils
class EvaluateDataset(Dataset):


    def __init__(self, query_ids, doc_ids, queries, docs, examples_file) -> None:
        super().__init__()
        self.gold_examples = example_utils.read_examples(examples_file)

        all_inputs = [] # either query or doc
        all_ids = [] # either query_id or doc_id
        is_doc = [] # either (1) for doc or (0) for query

        for doc_id, doc in zip(doc_ids, docs):
            all_inputs.append(doc)
            is_doc.append(1)
            all_ids.append(doc_id)

        for query_id, query in zip(query_ids, queries):
            all_inputs.append(query)
            is_doc.append(0)
            all_ids.append(query_id)

        self.all_inputs = all_inputs
        self.all_ids = all_ids
        self.is_doc = is_doc

    def __len__(self):
        """
        Return the number of inputs i.e. len(queries) + len(docs)
        """
        return len(self.all_inputs)

    def __getitem__(self, idx):
        """
        Return the idx-th item of the dataset in the format of (input, is_doc, input_id)
        either query or doc not both
        """
        input = self.all_inputs[idx]
        is_doc = self.is_doc[idx]
        input_id = self.all_ids[idx]
        return input, is_doc, input_id
