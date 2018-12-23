"""Natural language processing utilities."""
# pylint: disable=E0611
# pylint: disable=R0911,R0912,R0914
import re
from collections import namedtuple
import unicodedata
import hashlib
from .en.tenses import detect_tense as detect_tense_en
from .tenses import PRESENT, NORMAL


_ENT = ('B', 'I')

_IS_RX = re.compile(r"^\Ws$", re.IGNORECASE)
_WILL_RX = re.compile(r"^\Wll$", re.IGNORECASE)
_IS_NOT = re.compile(r"^n\Wt$", re.IGNORECASE)


Relation = namedtuple('Relation', [
    'tense', 'mode', 'rel', 'rtype', 'head', 'sub'
])


def get_relation(head, sub):
    """Get dependency relation from a head-child pair.

    Parameters
    ----------
    head : spacy.tokens.Span
        Head token in the compound form.
    sub : spacy.tokens.Span
        Subordinate token in the compound form.


    Relation types
    --------------

    verb-verb
      Relation between two verbs.
      Head is superordinate and sub is subordinate.

    subject-verb
      Subject and verb.
      This is to be interpreted in terms of an action performance.

    complement-verb
        Action connected to a complement.

    verb-objected
        Object of a performed action.
        The right side of the subject-object-triple.

    verb-complement
        Complement of a verb (action).

    left_adposition
        Adposition. It may be connected to any type of token.
        Adpositions introduce additional contextual information
        concerning things like time and/or space location of events.
        They also link related subsentences.

        Left adposition designates the subordinate of the head of the
        corresponding right adposition.

    right_adposition
        See ``left_adposition``.

        Right adposition designates the head of the corresponding
        left adposition.

    compound
        Two nouns constituting a compound noun.

    noun-noun
        Two nouns in a descriptive relation.
        For instance, "John Smith, school president".

    description
        Description relation.
        The head is described by the sub.

    misc
        Other types of relations.
        They can be safely ignored in most cases.
    """

    def _get_rtype(head, sub):
        ht = head._.drive
        st = sub._.drive
        if (st._.is_noun and st._.is_subj_dep) \
        or (ht._.is_verb and st._.is_subj_dep):
            head, sub = sub, head
            ht, st = st, ht
        if ht._.is_verb and st._.is_verb:
            rtype = 'verb-verb'
        elif (ht._.is_noun or ht._.is_subj_dep) and st._.is_verb:
            rtype = 'subject-verb'
        elif ht._.is_comp_dep and st._.is_verb:
            rtype = 'complement-verb'
        elif (ht._.is_verb or ht._.is_adj_verb) \
        and (st._.is_noun or st._.is_obj_dep):
            rtype = 'verb-object'
        elif ht._.is_verb and st._.is_comp_dep:
            rtype = 'verb-complement'
        elif ht._.is_adp:
            rtype = 'left_adposition'
        elif st._.is_adp:
            rtype = 'right_adposition'
        elif ht._.is_in_compound_noun and ht._.compound == st._.compound:
            rtype = 'compound'
        elif ht._.is_noun and st._.is_noun:
            rtype = 'noun-noun'
        elif st._.is_description:
            rtype = 'description'
        else:
            rtype = 'misc'
        return head, sub, rtype

    head, sub, rtype = _get_rtype(head, sub)
    hpos, hdep = head.root.pos_, head.root.dep_
    spos, sdep = sub.root.pos_, sub.root.dep_
    rel = f"{hpos}.{hdep}=>{spos}.{sdep}"
    if head.root._.is_verb or not sub.root._.is_verb:
        tense, mode = head._.tense
    else:
        tense, mode = sub._.tense
    return Relation(tense, mode, rel, rtype, head, sub)


def get_compound_verb(token):
    """Get compound verb from a verb token."""
    next_token = token
    while next_token._.is_verblike:
        try:
            next_token = next_token.nbor(1)
        except IndexError:
            break
    end = next_token._.si
    prev_token = token
    while prev_token._.is_verblike:
        try:
            prev_token = prev_token.nbor(-1)
        except IndexError:
            break
    while not prev_token._.is_verb:
        try:
            prev_token = prev_token.nbor(1)
        except IndexError:
            break
        if prev_token == token:
            break
    start = prev_token._.si
    while not token.sent[start]._.is_verb and start < end - 1:
        start += 1
    return token.sent[start:end]

def get_compound_noun(token):
    """Get compound noun from a noun token."""
    next_token = token
    while next_token._.is_in_compound_noun:
        try:
            next_token = next_token.nbor(1)
        except IndexError:
            break
    end = next_token._.si
    prev_token = token
    while prev_token._.is_in_compound_noun:
        try:
            prev_token = prev_token.nbor(-1)
        except IndexError:
            break
    start = prev_token._.si + 1
    while token.sent[start]._.is_det and start < end - 1:
        start += 1
    return token.sent[start:end]

def get_entity_from_span(span):
    """Get entity from a span by shrinking and/or expanding."""
    try:
        start_token = next(t for t in span if t._.is_ent)
    except StopIteration:
        return None
    start = start_token.i
    end = start_token.i
    token = start_token.nbor(1)
    while token.ent_iob_ == 'I':
        end = token.i
        token = token.nbor(1)
    if start_token.ent_iob_ != 'B':
        token = start_token
        while True:
            token = token.nbor(-1)
            if token.ent_iob_ == 'B':
                start = token.i
                break
    for ent in span.sent.ents:
        if ent.start == start and ent.end == end + 1:
            return ent

def detect_tense(verb):
    """Detect tense of a verb."""
    if verb.vocab.lang == 'en':
        return detect_tense_en(verb)
    return PRESENT, NORMAL

def make_hash(*args):
    md5 = hashlib.md5()
    string = '___'.join(map(str, args))
    md5.update(string.encode())
    return md5.hexdigest()

def make_doc(nlp, text, normalize_unicode=True):
    """Make document with normalized text.

    Parameters
    ----------
    nlp : spacy.lang
        Language object.
    text : str
        Document text.
    normalize_unicode : bool
        Should string be unicode-normalized.
    """
    def _text(x):
        if x.is_punct:
            text = x.lemma_
        elif x._.is_verb and _IS_RX.match(x.text):
            text = ' is'
        elif x._.is_verb and _WILL_RX.match(x.text):
            text = ' will'
        elif x._.is_neg_dep and _IS_NOT.match(x.text):
            text = ' not'
        else:
            text = x.text
        if normalize_unicode:
            text = unicodedata.normalize('NFC', text)
        return text
    doc = nlp(text)
    norm_text = ''
    for token in doc:
        text = _text(token)
        norm_text += text + token.whitespace_
    return nlp(norm_text)
