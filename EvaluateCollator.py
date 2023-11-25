class EvaluateCollator:
    def __init__(self, tokenizer, query_max_length, doc_max_length):
        self.tokenizer = tokenizer
        self.query_max_length = query_max_length
        self.doc_max_length = doc_max_length

    def __call__(self, batch):
        all_inputs = [] # either query or doc
        all_ids = [] # either query_id or doc_id
        all_is_doc = []
        for input, is_doc, id in batch:
            all_inputs.append(input) # either query or doc
            all_ids.append(id)# either query_id or doc_id
            all_is_doc.append(is_doc)

        tokenized_inputs = queries = self.tokenizer(
            all_inputs,
            padding=True,
            truncation=True,
            max_length=self.query_max_length,
            return_tensors="pt",
        )
        return {
            "ids": all_ids,
            "inputs": tokenized_inputs,
            "is_doc": all_is_doc,
        }
        # queries = []
        # docs = []
        # query_ids = []
        # doc_ids = []
        # for qid, did, query, doc in batch:
        #     query_ids.append(qid)
        #     doc_ids.append(did)
        #     queries.append(query)
        #     docs.append(doc)
        # queries = self.tokenizer(
        #     queries,
        #     padding=True,
        #     truncation=True,
        #     max_length=self.query_max_length,
        #     return_tensors="pt",
        # )
        # docs = self.tokenizer(
        #     docs,
        #     padding=True,
        #     truncation=True,
        #     max_length=self.doc_max_length,
        #     return_tensors="pt",
        # )
        # return {
        #     "query_ids": query_ids,
        #     "doc_ids": doc_ids,
        #     "queries": queries,
        #     "docs": docs,
        # }