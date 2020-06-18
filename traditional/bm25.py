# https://pypi.org/project/rank-bm25/
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
