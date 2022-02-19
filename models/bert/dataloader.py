import torch as th
from transformers import BertTokenizer

class Dataset(th.utils.data.Dataset):
    def __init__(self, mode):
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        sents = []
        self.labels = []
        with open("../../data/"+mode+".txt") as f:
            for line in f:
                sents.append(line[:-3])
                self.labels.append(int(line[-2]))

        self.encodings = tokenizer(sents, padding=True, truncation=True, max_length=512)

    def __getitem__(self, idx):
        item = {key: th.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = th.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])