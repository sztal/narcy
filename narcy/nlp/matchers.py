"""NLP matchers."""
# pylint: disable=E0611
# from spacy.matcher import Matcher


# def load_compound_noun_matcher(nlp):
#     """Get matcher for compound nouns."""
#     matcher = Matcher(nlp.vocab)
#     patterns = [ [
#         {'DEP': 'compound', 'OP': '*'},
#         {'POS':  { 'IN': ['NOUN', 'PROPN', 'PRON', 'DET'] } },
#         {'DEP': 'compound', 'OP': '*'}
#     ] ]
#     matcher.add('COMPOUND_VERB', None, *patterns)
#     return matcher
