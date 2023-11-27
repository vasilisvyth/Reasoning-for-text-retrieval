class EvaluateCollator:
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        # self.doc_max_length = doc_max_length

    def __call__(self, batch):
        all_inputs = [] # either query or doc
        all_ids = [] # either query_id or doc_id
    
        for id, input in batch:
            all_inputs.append(input) # either query or doc
            all_ids.append(id)# either query_id or doc_id
    

        tokenized_inputs = self.tokenizer(
            all_inputs,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            pad_to_multiple_of=8
        )
        return {
            "ids": all_ids,
            "inputs": tokenized_inputs,
        }
