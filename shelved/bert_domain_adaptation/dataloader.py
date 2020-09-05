import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange


class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path, block_size, logger):
        assert os.path.isfile(file_path)
        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            directory,
            args.model_name_or_path + "_cached_lm_" + str(block_size) + "_" + filename,
        )

        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            # print("Loading features from cached file " + str(cached_features_file))
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
        else:
            logger.info("Creating features from dataset file at %s", directory)
            # print("Creating features from dataset file at " + str(directory))

            self.examples = []
            with open(file_path, encoding="utf-8") as f:
                text = f.read()
                text = text.lower()
                # print(text)

            tokenized_text = tokenizer.convert_tokens_to_ids(
                tokenizer.tokenize(text=text)
            )

            for i in range(
                0, len(tokenized_text) - block_size + 1, block_size
            ):  # Truncate in block of block_size
                self.examples.append(
                    tokenizer.build_inputs_with_special_tokens(
                        tokenized_text[i : i + block_size]
                    )
                )
            # Note that we are loosing the last truncated example here for the sake of simplicity (no padding)
            # If your dataset is small, first you should look for a bigger one :-) and second you
            # can change this behavior by adding (model specific) padding.

            logger.info("Saving features into cached file %s", cached_features_file)
            # print("Saving features into cached file " + str(cached_features_file))
            with open(cached_features_file, "wb") as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item])


def mask_tokens(inputs, tokenizer, args):
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, args.mlm_probability)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
        for val in labels.tolist()
    ]

    probability_matrix.masked_fill_(
        torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0
    )
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = (
        torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    )
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = (
        torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
        & masked_indices
        & ~indices_replaced
    )
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels


def load_and_cache_examples(args, tokenizer, logger, evaluate=False):
    dataset = TextDataset(
        tokenizer,
        args,
        file_path=args.eval_data_file if evaluate else args.train_data_file,
        block_size=args.block_size,
        logger=logger,
    )
    return dataset
