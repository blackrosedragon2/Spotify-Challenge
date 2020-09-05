import sys

sys.path.append("./")

import os
import random

import numpy as np
import torch
import pickle
from torch.utils.data import Dataset, DataLoader
import json
from tqdm.auto import tqdm


RANDOM_SEED = 12345
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


class JsonTextDataset(Dataset):
    """
        Args:
            data_path: (list) Path of the pickled list
            Format of the list: [{'queryid': query, 'docid': doc,'label': label}, ...]
            where query, doc are strings and label is an integer.
            batch_size: (int) 
            tokenizer: 
            split: randomly shuffle dataset if split='training'
            device: 'cpu' or 'cuda'
    """

    def __init__(
        self, data_path, tokenizer, block_size=1024, split="training",
    ):
        super(JsonTextDataset, self).__init__()
        print("=" * 40, "Loading the " + split + " dataset", "=" * 40)
        with open(data_path, "r") as f:
            self.examples = json.load(f)

        if split == "training":
            random.shuffle(self.examples)

        print("=" * 40, "Tokenizing the Text", "=" * 40)
        self.examples = tokenizer(
            self.examples,
            add_special_tokens=True,
            truncation=True,
            max_length=block_size,
        )["input_ids"]

        print("=" * 40, "Making the lengths even", "=" * 40)
        for i, ele in enumerate(self.examples):
            if len(ele) % 2 != 0:
                token_last = self.examples[i][-1]
                self.examples[i][-1] = tokenizer.pad_token_id
                self.examples[i].append(token_last)
            # print(len(self.examples[i]))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i], dtype=torch.long)
