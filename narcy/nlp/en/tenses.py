"""ENGLISH: Tense detectors and related utilities."""
# pylint: disable=E0611
from ..tenses import PRESENT, PAST, FUTURE, MODAL, NORMAL


_TAGS_PAST = ('VBD', 'VBN')

_WORDS_HAVE = ('have', 'has', 'had')
_WORDS_PAST = ('did', 'was', 'were')
_WORDS_FUTURE = ('will', 'shall')
_WORDS_MODAL = (
    'should', 'would', 'may', 'might', 'can', 'could',
    'must', 'ought', 'need', 'needs', 'want', 'wants'
)

def detect_tense(verb):
    """Detect main tense of a compound verb.

    Parameters
    ----------
    verb : spacy.tokens.Span
        Compound verb.
    """
    tense = PRESENT
    mode = NORMAL
    if not verb:
        return tense, mode
    first = verb[0]
    try:
        second = first.nbor(1)
    except IndexError:
        second = None
    first_text = first.text.lower()
    is_second_to = second and second.text.lower() == 'to'
    # Check mode
    if first_text in _WORDS_MODAL:
        mode = MODAL
        if len(verb) > 1:
            first = verb[1]
    elif first_text in _WORDS_HAVE and is_second_to:
        mode = MODAL
        if len(verb) > 2:
            first = verb[2]
    # Check tense
    try:
        second = first.nbor(1)
    except IndexError:
        second = None
    first_text = first.text.lower()
    is_second_to = second and second.text.lower() == 'to'
    if first_text in _WORDS_PAST:
        tense = PAST
    elif first_text in _WORDS_HAVE and not is_second_to:
        tense = PAST
    elif first.tag_ in _TAGS_PAST:
        tense = PAST
    elif first_text in _WORDS_FUTURE:
        tense = FUTURE
    elif is_second_to:
        tense = FUTURE
    return tense, mode
