"""Getters for extension attributes defined on *Spacy* objects."""
# pylint: disable=E0611,C0321,W0212
from itertools import product
from spacy.symbols import NOUN, PROPN, PRON, DET
from spacy.symbols import VERB, PART
from spacy.symbols import ADV, ADJ, ADP
from spacy.symbols import SPACE
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from ..utils import get_compound_verb, get_compound_noun, get_entity_from_span
from ..utils import get_relation, detect_tense, make_hash
from ..tenses import PRESENT, NORMAL

vader = SentimentIntensityAnalyzer()

_NOT_SEMANTIC = (DET, PRON, PART)
_NOUN = (NOUN, PROPN)
_NOUNLIKE = _NOUN
_VERB = (VERB,)
_VERB_DESC = (ADV, ADJ)
_NONWORDS = (SPACE,)

_AUX = ('aux',)
_NSUBJ = ('nsubj', 'nsubjpass')
_CSUBJ = ('csubj', 'nsubj')
_SUBJ = _NSUBJ + _CSUBJ
_NONVERB_DEP = ('acl', 'acomp', 'amod', 'advmod')
_CLAUSE_VERB_DEP = ('advcl', 'ccomp')
_PREP = ('prep',)
_POSSESIVES = ('poss',)
_NEG = ('neg',)
_CONJ = ('conj',)
_ADJECTIVAL = ('acl', 'amod', 'amod')
_OBJ = ('obj', 'pobj', 'dobj')
_COMPOUND = ('compound',)
_COMPLEMENT = ('acomp',)
_ATTR = ('attr',)

_TAGS_PART = ('VBN', 'VBD', 'VBG')
_TAGS_INF = ('VB',)
_TAGS_POSS = ('POS',)
_TAGS_COMPOUND = ('HYPH',)

_ENT = ('B', 'I')

# Token extensions ------------------------------------------------------------

is_wordlike_t_g = lambda t: not t.is_punct and not t.like_num \
    and not t.tag_ in _TAGS_POSS and t.pos not in _NONWORDS
is_semantic_t_g = lambda t: t._.is_wordlike and t.pos not in _NOT_SEMANTIC \
    and t.dep_ not in _POSSESIVES
is_drive_t_g = lambda t: t._.compound._.drive == t
is_root_t_g = lambda t: t._.compound.root == t

is_noun_t_g = lambda t: t.pos in _NOUN
is_nounlike_t_g = lambda t: t._.is_noun
is_in_compound_noun_t_g = lambda t: (t._.is_compound_dep \
    or any(c._.is_compound_dep for c in t.children))
is_verb_t_g = lambda t: t.pos in _VERB and t.dep_ not in _NONVERB_DEP \
    and not t.dep_ in _SUBJ and not t._.is_adj_verb
is_verblike_t_g = lambda t: t._.is_verb  or t._.is_part or t._.is_prep_dep \
    or t._.is_neg_dep
is_adj_verb_t_g = lambda t: t.dep_ in _ADJECTIVAL and t.tag_ in _TAGS_PART
is_clause_verb_t_g = lambda t: t._.is_verb and t.dep_ in _CLAUSE_VERB_DEP
is_part_t_g = lambda t: t.pos == PART
is_det_t_g = lambda t: t.pos == DET
is_auxpart_t_g = lambda t: t._.is_part and t._.is_aux_dep
is_adp_t_g = lambda t: t.pos == ADP
is_adj_t_g = lambda t: t.pos == ADJ
is_adv_t_g = lambda t: t.pos == ADV

is_description_t_g = lambda t: t._.is_adj or t._.is_adv or t._.is_adj_verb
is_term_t_g = lambda t: t._.is_drive and t._.is_semantic \
    and (t._.is_description or t._.is_noun or t._.is_verb)
is_obj_t_g = lambda t: t._.is_drive \
    and (t._.is_obj_dep or t._.is_comp_dep or t._.is_attr_dep)

is_prep_dep_t_g = lambda t: t.dep_ in _PREP
is_aux_dep_t_g = lambda t: t.dep_ in _AUX
is_conj_dep_t_g = lambda t: t.dep_ in _CONJ
is_obj_dep_t_g = lambda t: t.dep_ in _OBJ
is_compound_dep_t_g = lambda t: t.dep_ in _COMPOUND
is_subj_dep_t_g = lambda t: t.dep_ in _SUBJ
is_comp_dep_t_g = lambda t: t.dep_ in _COMPLEMENT
is_attr_dep_t_g = lambda t: t.dep_ in _ATTR
is_neg_dep_t_g = lambda t: t.dep_ in _NEG
is_poss_dep_t_g = lambda t: t.dep_ in _POSSESIVES

is_compound_tag_t_g = lambda t: t.dep_ in _TAGS_COMPOUND

is_ent_t_g = lambda t: t.ent_iob_ in _ENT

def is_desc_verb_t_g(token):
    try:
        previous_token = token.nbor(-1)
    except IndexError:
        previous_token = None
    return token._.is_clause_verb and previous_token and previous_token._.is_auxpart

def compound_t_g(token):
    compound = token.sent[token._.si:token._.si+1]
    if compound._.is_ent:
        return get_entity_from_span(compound)
    if token._.is_verblike:
        compound = get_compound_verb(token)
        if compound:
            return compound
    if token._.is_nounlike:
        compound = get_compound_noun(token)
    if not compound:
        compound = token.sent[token._.si:token._.si+1]
    return compound

def conjuncts_t_g(token):
    for child in token.children:
        if child._.is_conj_dep:
            yield child
            yield from child._.conjuncts

def si_t_g(token):
    return token.i - token.sent.start

def relations_t_g(token):
    token_c = token._.compound
    if not token._.is_verblike and token._.is_drive and token_c._.is_compound:
        for t1, t2 in product(token_c, token_c):
            if t1 == t2 or not t1._.is_wordlike or not t2._.is_wordlike:
                continue
            sent = token.sent
            i = t1.i - sent.start
            j = t2.i - sent.start
            yield get_relation(sent[i:i+1], sent[j:j+1])
    for child in token.children:
        if not child._.is_wordlike:
            continue
        child_c = child._.compound
        if not child._.is_conj_dep:
            for conjunct in child._.conjuncts:
                yield get_relation(token_c, conjunct._.compound)
                for conj_child in conjunct.children:
                    if conj_child._.is_obj_dep:
                        yield get_relation(child_c, conj_child._.compound)
        if token_c != child_c and not child._.is_conj_dep:
            yield get_relation(token_c, child_c)
        yield from child._.relations

def subterms_t_g(token):
    for st in token.subtree:
        if st._.is_term and st not in token._.compound:
            yield st._.compound


# Span extensions -------------------------------------------------------------

def root_s_g(span):
    root = span.root
    if not root._.is_wordlike:
        try:
            root = next(x for x in root.children if x._.is_verb)
        except StopIteration:
            return None
    return root


def verbs_s_g(span):
    i = 0
    n = len(span)
    while i < n:
        token = span[i]
        if token._.is_verb:
            compound = token._.compound
            yield compound
            i = compound.end - span.start
        else:
            i += 1

def nouns_s_g(span):
    for token in span:
        if token._.is_noun and token._.is_drive:
            yield token._.compound

def tense_s_g(span):
    if span.root._.is_verb:
        if span.root._.is_desc_verb or span.root._.is_conj_dep:
            return span.root.head._.compound._.tense
        return detect_tense(span)
    vparent = span._.vparent
    if not vparent:
        return PRESENT, NORMAL
    return vparent._.tense

def vparent_s_g(span):
    try:
        return next(t._.compound for t in span.root.ancestors if t._.is_verb)
    except StopIteration:
        return None

def vobjects_s_g(span):
    for child in span._.drive.children:
        if child._.is_obj:
            yield child._.compound
            if child._.is_conj_dep:
                for conj in child._.conjuncts:
                    if conj._.is_drive:
                        yield conj._.compound
        elif child._.is_conj_dep and child._.is_verb:
            yield from child._.compound._.vobjects
        for conj in child._.conjuncts:
            if conj._.is_obj:
                yield conj._.compound

def is_compound_s_g(span):
    return len(span) > 1

def is_ent_s_g(span):
    return any(t.ent_iob_ in _ENT for t in span)

def is_neg_s_g(span):
    return any(t._.is_neg_dep for t in span)

def drive_s_g(span):
    if span.root._.is_verb:
        for token in reversed(span):
            if token._.is_verb:
                return token
    return span.root

def lead_s_g(span):
    drive = span._.drive
    if drive._.is_verb:
        try:
            neg = next(t for t in reversed(span) if t._.is_neg_dep)
        except StopIteration:
            neg = None
        start = min(drive._.si, neg._.si) if neg else drive._.si
        end = max(drive._.si + 1, neg._.si + 1) if neg else drive._.si + 1
        try:
            next_token = drive.nbor(1)
            if next_token._.is_part:
                end += 1
        except IndexError:
            pass
        return span.sent[start:end]
    return span

def lemma_s_g(span):
    drive = span._.drive
    if drive._.is_verb:
        neg = any(t._.is_neg_dep for t in reversed(span))
        lemma = drive.lemma_
        if neg:
            lemma = 'not ' + lemma
    else:
        lemma = span._.lead.lemma_
    return lemma

def relations_s_g(span):
    root = span._.root
    if not root:
        return
    yield from root._.relations

def id_s_g(span):
    return span.doc._.id+'__'+str(span.start)+'__'+str(span.end)

def lang_s_g(span):
    return span.doc.vocab.lang

def polarity_s_g(span):
    _polarity = span._._polarity
    if not _polarity:
        _polarity = vader.polarity_scores(span.text)
        span._.set('polarity', _polarity)
    return _polarity

def valence_s_g(span):
    scores = span._.polarity
    return (scores['pos']**.5 - scores['neg']**5) * (1 - scores['neu'])**.5

def sentiment_s_g(span):
    scores = span._.polarity
    return scores['compound']*(1 - scores['neu'])

def start_s_g(span):
    return span.start - span.sent.start

def end_s_g(span):
    return span.end - span.sent.start

def tokens_s_g(span):
    i = 0
    while i < len(span):
        token = span[i]._.compound
        if token._.drive._.is_wordlike:
            yield token
        i = token._.end


# Doc extensions --------------------------------------------------------------

def id_d_g(doc):
    _id = doc._._id
    if not _id:
        _id = make_hash(doc.text)
        doc._.set('id', _id)
    return _id

def relations_d_g(doc):
    for sent in doc.sents:
        yield from sent._.relations

def polarity_d_g(doc):
    _polarity = doc._._polarity
    if not _polarity:
        _polarity = vader.polarity_scores(doc.text)
        doc._.set('polarity', _polarity)
    return _polarity

def valence_d_g(doc):
    scores = doc._.polarity
    return (scores['pos']**.5 - scores['neg']**5) * (1 - scores['neu'])**.5

def sentiment_d_g(doc):
    scores = doc._.polarity
    return scores['compound']*(1 - scores['neu'])

def tokens_d_g(doc):
    for sent in doc.sents:
        yield from sent._.tokens
