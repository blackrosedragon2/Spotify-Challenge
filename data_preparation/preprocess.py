import spacy
from spacy.lang.en import English


def to_lower(doc):
    """
		Converts all letters of the document to lower case
		Args:
		doc: (string) 
	"""
    return doc.lower()


def lemmatization(doc):
    """
		Replaces every word in the document with the root word using spacy
		Args:
		doc: (string)
	"""
    nlp = spacy.load("en", disable=["parser", "ner"])
    doc = nlp(doc)

    return " ".join([token.lemma_ for token in doc])


def remove_stop_words(doc):
    """
		Removes stop words 
		Args:
		doc: (string)
	"""
    nlp = English()
    my_doc = nlp(doc)
    tokens = [token.text for token in my_doc]
    filtered = [word for word in tokens if not (nlp.vocab[word].is_stop)]

    return " ".join(filtered)


def preprocess(doc):
    """
		Performs to_lower and lemmatization on doc.
		Args:
		doc: (string)
	"""

    return remove_stop_words(lemmatization(to_lower(doc)))
