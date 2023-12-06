from torch.utils.data import Dataset

class Code_llm_dataset(Dataset):
    def __init__(self,input_ids, attention_mask, all_qids):
        super().__init__()
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.all_qids = all_qids
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, index):
        return {"input_ids": self.input_ids[index], "attention_mask": self.attention_mask[index],"qids":self.all_qids[index]}


