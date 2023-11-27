class BiEncoderPairCollator:
    def __init__(self, tokenizer, query_max_length, doc_max_length):
        self.tokenizer = tokenizer
        self.query_max_length = query_max_length
        self.doc_max_length = doc_max_length

    def __call__(self, batch):
        queries = []
        docs = []
        query_ids = []
        doc_ids = []
        for qid, did, query, doc in batch:
            query_ids.append(qid)
            doc_ids.append(did)
            queries.append(query)
            docs.append(doc)
        queries = self.tokenizer(
            queries,
            padding=True,
            truncation=True,
            max_length=self.query_max_length,
            return_tensors="pt",
            pad_to_multiple_of=8
        )
        docs = self.tokenizer(
            docs,
            padding=True,
            truncation=True,
            max_length=self.doc_max_length,
            return_tensors="pt",
            pad_to_multiple_of=8
        )
        return {
            "query_ids": query_ids,
            "doc_ids": doc_ids,
            "queries": queries,
            "docs": docs,
        }