"""Data processors."""
# pylint: disable=E0611,W0640
# pylint: disable=R0914
from collections import namedtuple
from itertools import takewhile
import pandas as pd
from .nlp.utils import get_relation


Record = namedtuple('Record', [
    'head_tense', 'head_mode', 'sub_tense', 'sub_mode', 'rtype',
    'head', 'sub', 'head_lead', 'sub_lead', 'head_lemma', 'sub_lemma',
    'head_neg', 'sub_neg', 'head_pos', 'head_dep', 'sub_pos', 'sub_dep',
    'head_ent', 'head_ent_label', 'sub_ent', 'sub_ent_label',
    'head_vector_norm', 'sub_vector_norm', 'head_vector', 'sub_vector',
    'head_start', 'head_end', 'sub_start', 'sub_end',
    'sentiment', 'sent_sentiment',
    'valence', 'sent_valence',
    'docid', 'sentid'
])

SVO = namedtuple('SVO', [
    'tense', 'mode', 'neg', 'rtype',
    'subj', 'subj_terms', 'verb', 'obj', 'obj_terms',
    'sentiment', 'sent_sentiment',
    'valence', 'sent_valence'
])

SVORecord = namedtuple('SVORecord', [
    'tense', 'mode', 'neg', 'rtype',
    'subj', 'verb', 'obj',
    'subj_lead', 'verb_lead', 'obj_lead',
    'subj_lemma', 'verb_lemma', 'obj_lemma',
    'subj_ent', 'subj_ent_label', 'obj_ent', 'obj_ent_label',
    'subj_terms', 'obj_terms',
    'subj_vector_norm', 'verb_vector_norm', 'obj_vector_norm',
    'subj_vector', 'verb_vector', 'obj_vector',
    'sentiment', 'sent_sentiment',
    'valence', 'sent_valence',
    'docid', 'sentid'
])

Token = namedtuple('Token', [
    'tense', 'mode', 'neg',
    'token', 'lead', 'lemma',
    'pos', 'dep',
    'ent', 'ent_label',
    'vector_norm', 'vector',
    'start', 'end',
    'sentiment', 'sent_sentiment',
    'valence', 'sent_valence',
    'docid', 'sentid'
])


def relation_to_record(r):
    """Convert relation to record.

    Parameters
    ----------
    r : tuple
        Relation tuple.
    """
    head = r.head
    sub = r.sub
    sent = r.head.sent
    doc = sent.doc
    docid = doc._.id
    sentid = sent._.id
    head_text = r.head.text.lower()
    sub_text = r.sub.text.lower()
    rel = doc[min(head.start, sub.start):max(head.end, sub.end)]
    sub_tense, sub_mode = sub._.lead._.tense
    head_pos, head_dep, sub_pos, sub_dep = \
        tuple(y for x in r.rel.split('=>') for y in x.split('.'))
    head_neg = any(t._.is_neg_dep for t in head)
    sub_neg = any(t._.is_neg_dep for t in sub)
    return Record(
        head_tense=r.tense,
        head_mode=r.mode,
        sub_tense=sub_tense,
        sub_mode=sub_mode,
        rtype=r.rtype,
        head=head_text,
        sub=sub_text,
        head_lead=head._.lead.text.lower(),
        sub_lead=sub._.lead.text.lower(),
        head_lemma=head._.lemma,
        sub_lemma=sub._.lemma,
        head_neg=head_neg,
        sub_neg=sub_neg,
        head_pos=head_pos,
        head_dep=head_dep,
        sub_pos=sub_pos,
        sub_dep=sub_dep,
        head_ent=head._.is_ent,
        head_ent_label=head.label_,
        sub_ent=sub._.is_ent,
        sub_ent_label=sub.label_,
        head_vector_norm=head.vector_norm,
        sub_vector_norm=sub.vector_norm,
        head_vector=head.vector,
        sub_vector=sub.vector,
        head_start=head.start,
        head_end=head.end,
        sub_start=sub.start,
        sub_end=sub.end,
        sentiment=rel._.sentiment,
        sent_sentiment=sent._.sentiment,
        valence=rel._.valence,
        sent_valence=sent._.valence,
        docid=docid,
        sentid=sentid
    )


def _reduce_left_adposition(r):
    if any(c._.is_adp for c in r.head._.drive.children):
        return None
    head = r.head.root
    while head._.is_adp and head.dep_ != 'ROOT':
        head = head.head
    return get_relation(head._.compound, r.sub)


def reduce_relations(relations):
    """Transform relations into relation reducts.

    Parameters
    ----------
    relations : iterable
        Iterable of relations.
    """
    for r in relations:
        if r.rtype == 'misc':
            continue
        elif r.rtype == 'right_adposition':
            continue
        elif r.rtype == 'left_adposition':
            r = _reduce_left_adposition(r)
        if r and r.rtype != 'misc':
            yield r


def relations_to_df(relations, columns=None, **kwds):
    """Convert relations to a data frame.

    Parameters
    ----------
    relations : iterable
        Iterable of relations.
    columns : iterable or None
        If ``None``, then ``Record`` field names are used.
    kwds :
        Additional keyword arguments passed to
        :py:meth:`pandas.DataFrame.from_records`.
    """
    records = map(relation_to_record, relations)
    columns = Record._fields if not columns else columns
    df = pd.DataFrame \
        .from_records(records, columns=columns, **kwds) \
        .drop_duplicates(subset=[
            'rtype', 'head_start', 'head_end',
            'sub_start', 'sub_end', 'docid', 'sentid'
        ])
    return df

def doc_to_relations_df(doc, reduced=True, **kwds):
    """Dump document to a relations data frame.

    Parameters
    ----------
    doc : spacy.tokens.Doc
        Document object.
    reduced : bool
        Should relations reducts be used.
    **kwds :
        Other keyword arguments passed to :py:func:`relations_to_df`.
    """
    relations = doc._.relations
    if reduced:
        relations = reduce_relations(relations)
    return relations_to_df(relations, **kwds)

def get_svos(relations):
    """Get subject-verb-object triplets from a relations.

    Relation types
    ==============

    svo
        Subject-verb-object triplet.

    svc
        Subject-verb-complement triplet.
    """
    for relation in relations:
        if relation.rtype != 'subject-verb':
            continue
        verb, subj = relation.sub, relation.head
        if not subj._.drive._.is_semantic:
            continue
        tense, mode = verb._.tense
        neg = verb._.is_neg
        for obj in verb._.vobjects:
            if obj._.drive._.is_obj_dep or obj._.drive._.is_noun:
                rtype = 'svo'
            else:
                rtype = 'svc'
            subj_terms = tuple(takewhile(lambda t: t != verb, subj._.drive._.subterms))
            obj_terms = tuple(st for st in obj._.drive._.subterms if st != verb)
            start = min(
                subj.start, verb.start, obj.start,
                *[ t.start for t in subj_terms ],
                *[ t.start for t in obj_terms ]
            )
            end = max(
                subj.end, verb.end, obj.end,
                *[ t.end for t in subj_terms ],
                *[ t.end for t in obj_terms ]
            )
            sent = subj.sent
            doc = sent.doc
            rel = doc[start:end]
            yield SVO(
                tense=tense,
                mode=mode,
                neg=neg,
                rtype=rtype,
                subj=subj,
                subj_terms=subj_terms,
                verb=verb,
                obj=obj,
                obj_terms=obj_terms,
                sentiment=rel._.sentiment,
                sent_sentiment=sent._.sentiment,
                valence=rel._.valence,
                sent_valence=sent._.valence
            )

def svo_to_record(svo):
    """Convert *SVO* object to *SVO* record.

    Parameters
    ----------
    svo : tuple
        SVO tuple.
    """
    return SVORecord(
        tense=svo.tense,
        mode=svo.mode,
        neg=svo.neg,
        rtype=svo.rtype,
        subj=svo.subj.text.lower(),
        verb=svo.verb.text.lower(),
        obj=svo.obj.text.lower(),
        subj_lead=svo.subj._.lead.text.lower(),
        verb_lead=svo.verb._.lead.text.lower(),
        obj_lead=svo.obj._.lead.text.lower(),
        subj_lemma=svo.subj._.lemma,
        verb_lemma=svo.verb._.lemma,
        obj_lemma=svo.obj._.lemma,
        subj_ent=svo.subj._.is_ent,
        subj_ent_label=svo.subj.label_,
        obj_ent=svo.obj._.is_ent,
        obj_ent_label=svo.obj.label_,
        subj_terms=tuple(t._.lemma for t in svo.subj_terms),
        obj_terms=tuple(t._.lemma for t in svo.obj_terms),
        subj_vector_norm=svo.subj.vector_norm,
        verb_vector_norm=svo.verb.vector_norm,
        obj_vector_norm=svo.obj.vector_norm,
        subj_vector=svo.subj.vector,
        verb_vector=svo.verb.vector,
        obj_vector=svo.obj.vector,
        sentiment=svo.sentiment,
        sent_sentiment=svo.sent_sentiment,
        valence=svo.valence,
        sent_valence=svo.sent_valence,
        docid=svo.verb.doc._.id,
        sentid=svo.verb.sent._.id
    )

def doc_to_svos_df(doc, columns=None):
    """Dump document to a *SVOs* data frame.

    Parameters
    ----------
    doc : spacy.tokens.Doc
        Document object.
    columns : iterable or None
        If ``None``, then ``SVORecord`` field names are used.
    """
    records = map(svo_to_record, get_svos(doc._.relations))
    columns = SVORecord._fields if not columns else columns
    df = pd.DataFrame.from_records(records, columns=columns)
    return df

def get_tokens(doc):
    """Get tokens from a document.

    Parameters
    ----------
    doc : spacy.tokens.Doc
        Document object.
    """
    for token in doc._.tokens:
        tense, mode = token._.tense
        yield Token(
            tense=tense,
            mode=mode,
            neg=token._.is_neg,
            token=token.text.lower(),
            lead=token.text.lower(),
            lemma=token.lemma_,
            pos=token._.drive.pos_,
            dep=token._.drive.dep_,
            ent=token._.is_ent,
            ent_label=token.label_,
            vector_norm=token.vector_norm,
            vector=token.vector,
            start=token.start,
            end=token.end,
            sentiment=token._.sentiment,
            sent_sentiment=token.sent._.sentiment,
            valence=token._.valence,
            sent_valence=token.sent._.valence,
            docid=token.doc._.id,
            sentid=token.sent._.id
        )

def doc_to_tokens_df(doc, columns=None):
    """Dump document to a tokens data frame.

    Parameters
    ----------
    doc : spacy.tokens.Doc
        Document object.
    columns : iterable or None
        If ``None``, then ``Token`` field names are used.
    """
    columns = Token._fields if not columns else columns
    df = pd.DataFrame.from_records(get_tokens(doc), columns=columns)
    return df
