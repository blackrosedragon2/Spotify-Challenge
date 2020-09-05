import itertools
import nltk
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem.porter import *
from nltk.stem.snowball import SnowballStemmer
import argparse
import json

import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns
import sklearn.metrics

from tqdm.auto import tqdm


def run(queryList, model):

    # stemmer = PorterStemmer()
    stemmer = SnowballStemmer("english")

    f = open("data/expanded.txt", "w+")
    for query in tqdm(queryList):
        querySplitted = query["query"]

        # tokenizing the query
        tokens = nltk.word_tokenize(querySplitted)

        # removing stop words in the query
        filtered_words = [
            word for word in tokens if word not in stopwords.words("english")
        ]

        # pos tagging of tokens
        pos = nltk.pos_tag(filtered_words)

        synonyms = []  # synonyms of all the tokens

        index = 0
        # iterating through the tokens
        for item in filtered_words:
            synsets = wordnet.synsets(item)

            if not synsets:
                # stemming the tokens in the query
                synsets = wordnet.synsets(stemmer.stem(item))

            # synonyms of the current token
            currentSynonyms = []
            currentPOS = get_wordnet_pos(pos[index])

            # iterating through the synsets
            for i in synsets:
                # first we check if token and synset have the same part of speech
                if str(i.pos()) == str(currentPOS):
                    for j in i.lemmas():
                        if j.name() not in currentSynonyms:  # if we have not
                            currentSynonyms.append(j.name().replace("_", " "))
                synonyms.append(currentSynonyms)
            index += 1

        f.write(querySplitted + "\n")

        # removing duplicate lists in the synonyms list
        tmp = []
        for elem in synonyms:
            if elem and elem not in tmp:
                tmp.append(elem)
        synonyms = tmp

        # now that we have all the synonyms
        for x in itertools.product(*synonyms):
            current = ""
            for item in x:
                current += item
                current += " "
            if (
                sklearn.metrics.pairwise.cosine_similarity(
                    X=model([querySplitted]), Y=model([current])
                )[0][0]
                > 0.7
            ):
                current += "<delim> "
                f.write(current)
        f.write("\n")
        f.write("Next Query\n")


def get_wordnet_pos(treebank_tag):

    if treebank_tag[1].startswith("J"):
        return wordnet.ADJ
    elif treebank_tag[1].startswith("V"):
        return wordnet.VERB
    elif treebank_tag[1].startswith("N"):
        return wordnet.NOUN
    elif treebank_tag[1].startswith("R"):
        return wordnet.ADV
    else:
        return ""


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--queries_list",
        type=str,
        default="data/testset.json",
        help="path of json file containing captions, image URLs and factual words",
    )
    args = parser.parse_args()

    queries_list = args.queries_list

    module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
    model = hub.load(module_url)
    print("module %s loaded" % module_url)

    nltk.download("averaged_perceptron_tagger")

    with open(queries_list, "rb") as f:
        queries = json.load(f)

    run(queries, model)


if __name__ == "__main__":
    main()
