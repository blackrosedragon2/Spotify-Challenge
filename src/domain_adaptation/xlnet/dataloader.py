import sys

sys.path.append("./")

import os
import random

import numpy as np
import torch
import pickle


RANDOM_SEED = 12345
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


class JsonTextDataset(object):
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
        self,
        data_paths,
        batch_size,
        tokenizer,
        split="training",
        device=torch.device("cuda"),
    ):
        super(DataGenerator, self).__init__()
        self.data = []
        for paths in data_paths:
            with open(path, "r") as f:
                self.data += json.load(f)

        if split != "test":
            np.random.shuffle(self.data)

        self.data_i = 0
        self.data_size = len(self.data)
        self.batch_size = batch_size
        self.device = device
        self.tokenizer = tokenizer
        self.start = True

    def get_instance(self):
        """Returns one data-point, i.e., one dictionary {'queryid': query, 'docid': doc,'label': label} from the input list"""
        ret = self.data[self.data_i % self.data_size]
        self.data_i += 1
        return ret

    def __len__(self):
        return self.data_size

    def epoch_end(self):
        """Returns true when the end of the epoch is reached, otherwise false"""
        return self.data_i % self.data_size == 0

    def load_batch(self):
        """Takes the required number of data-points (batch_size), computes all the masks and returns the appended inputs+masks"""
        (docid_batch, untokenized_doc,) = (
            [],
            [],
        )
        while True:
            if not self.start and self.epoch_end():
                self.start = True
                break
            self.start = False
            instance = self.get_instance()

            a = str(instance["id"])
            b = instance["content"]

            docid_batch.append(a)
            untokenized_doc.append(b)

            if len(docid_batch) >= self.batch_size or self.epoch_end():
                # Convert inputs to PyTorch tensors
                inputs = self.tokenizer(
                    text=untokenized_doc,
                    max_length=1024,
                    truncation=True,
                    padding="longest",
                )

                for key in inputs:
                    inputs[key] = torch.tensor(inputs[key], device=self.device)

                # qid_tensor = torch.tensor(qid_batch, device=self.device)
                # docid_tensor = torch.tensor(docid_batch, device=self.device)

                return inputs

        return None
