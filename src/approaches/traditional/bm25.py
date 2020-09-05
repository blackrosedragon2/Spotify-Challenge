# Use Pyserini

import pyserini
from rank_bm25 import BM25Okapi


def compute_bm25_scores(corpus, query, top_k=3):

    """
		Computes the BM25 scores for each document in the corpus, with respect to the query.
		Args:
		- corpus: list of strings, where each string represents a document
		- query: a string

		Returns a tuple, first of which is a list, containing the scores, and the second is a list containing top_k docs

		Example:
		corpus = [
		     "Hello there good man!",
		     "It is quite windy in London",
		     "How is the weather today?"
		     ]
		query = "windy night"
		import traditional.bm25 as bm
		print(bm.compute_bm25_scores(corpus, query, 1))
		(array([0.        , 0.46864736, 0.        ]), ['It is quite windy in London'])
	
	"""

    tokenised_corpus = [doc.split(" ") for doc in corpus]
    tokenised_query = query.split(" ")

    bm25 = BM25Okapi(tokenised_corpus)

    return (
        bm25.get_scores(tokenised_query),
        bm25.get_top_n(tokenised_query, corpus, n=top_k),
    )


def search_bm25_rm3(index_path, query, top_k=3, raw_doc=True):

    """
		Uses the traditional BM25+RM3 method for ranking documents.
		Args:
		index_path: path of the index you create for your dataset
		query: (string) 
		top_k: (int, default: 3) number of hits you want to return (positive number)
		raw_doc: (boolean, default: True) returns the raw text of the document as well, along with the dictionary 

		Returns a dictionary of the doc_ids and the corresponding scores of the first top_k documents
		{'docid':[...], 'score':[...]}
		The format of the dictionary if raw_doc = True is:
		{'docid':[...], 'score':[...], 'text':...}

	"""

    searcher = pyserini.search.SimpleSearcher(index_path)
    hits = searcher.search(query, k=top_k)

    if raw_doc:
        hits_top_k = {"docid": [], "score": [], "text": []}
    else:
        hits_top_k = {"docid": [], "score": []}

    for i in range(top_k):
        hits_top_k["docid"].append(hits[i].docid)
        hits_top_k["score"].append(hits[i].score)

        if raw_doc:
            hits_top_k["text"].append(hits[i].raw)

    return hits_top_k
