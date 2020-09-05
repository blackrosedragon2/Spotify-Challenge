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


class DataGenerator(object):
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
        data_path,
        batch_size,
        tokenizer,
        split="training",
        device=torch.device("cuda"),
    ):
        super(DataGenerator, self).__init__()
        self.data = pickle.load(open(data_path, "rb"))

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
        (
            untokenized_query,
            untokenized_desc,
            untokenized_doc,
            label_batch,
            qid_batch,
            docid_batch,
        ) = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        while True:
            if not self.start and self.epoch_end():
                self.start = True
                break
            self.start = False
            instance = self.get_instance()
            qid, docid, label = instance

            a = instance[qid]
            b = instance[docid]
            label = instance[label]

            query = a.split(".")[0]
            desc = ".".join(a.split(".")[1:])

            qid = int(qid)
            docid = docid
            # print("DETAILS A,QID",a,qid)
            # print("DETAILS B,DOCID,LABEL",b,docid,label)
            # break
            # qid_batch.append(qid)
            # docid_batch.append(docid)
            untokenized_query.append(query)
            untokenized_desc.append(desc)
            untokenized_doc.append(b)
            label_batch.append(label / 4)

            if len(untokenized_query) >= self.batch_size or self.epoch_end():
                # Convert inputs to PyTorch tensors
                initial = tokenizer(
                    text=untokenized_query,
                    text_pair=untokenized_desc,
                    max_length=512,
                    truncation=True,
                    padding="longest",
                )
                for key in initial:
                    initial[key] = [out[:-1] for out in initial[key]]

                second = tokenizer(
                    text=untokenized_doc,
                    max_length=1400,
                    truncation=True,
                    padding="longest",
                )
                inputs = {}
                for key in second:
                    inputs[key] = torch.tensor(
                        [
                            first_ele + second_ele
                            for first_ele, second_ele in zip(initial[key], second[key])
                        ],
                        device=self.device,
                    )

                label_tensor = torch.tensor(label_batch, device=self.device)
                # qid_tensor = torch.tensor(qid_batch, device=self.device)
                # docid_tensor = torch.tensor(docid_batch, device=self.device)

                return (
                    inputs,
                    label_tensor,
                    # qid_tensor,
                    # docid_tensor,
                )

        return None
