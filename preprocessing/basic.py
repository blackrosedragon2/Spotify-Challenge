import spacy


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


def preprocess(doc):
    """
		performs to_lower and lemmatization on doc.
		Args:
		doc: (string)
	"""

    return lemmatization(to_lower(doc))
