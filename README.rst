===============================================================
*Narcy:* NLP relations extractor and narrative analysis library
===============================================================

.. image:: https://badge.fury.io/py/narcy.png
    :target: http://badge.fury.io/py/narcy

.. image:: https://travis-ci.org/sztal/narcy.png?branch=master
    :target: https://travis-ci.org/sztal/narcy

*Narcy* is a *Python* package for extraction of linguistic relations
from natural language and narrative analysis. It is based on the powerful
Spacy_ NLP library.

Installation
============

.. code-block:: bash

    pip install git+https://github.com/sztal/narcy.git
    python -m spacy download en_core_web_sm


Narrative analysis
==================

*Narrative analysis* (NA) is a technique for analysis of natural language
that focuses on extraction of relations between entities and
descriptions of those entities including associated sentiment scores.
The key part of narrative analysis is detection of tense and mode
so all relations can be associated with past, present and future tense
as well as normal or modal mode. This enables study of so-called
narrative arcs.

NA combines several standard NLP techniques such as named entity recognition,
POS and syntactic dependency tagging, vector space representations
and sentiment analysis.


Features
========

* Extract relations and relation reducts from documents.
* Extract subject-verb-object triplets from documents.
* Detect tense and mode.

Usage
=====

Currently there are two workhorse functions that converts documents
into tidy data frames that describe all relations extracted from the text.

``doc_to_relations``
    This function dumps a *spacy* ``Doc`` objects to *pandas* data frames
    describing all relation or relation reducts (see following sections).

``doc_to_svos``
    This function dumps a *spacy* ``Doc`` objects to *pandas* data frames
    describing all subject-verb-object triplets.


Example
-------

First, load *Spacy* and extend it with *Narcy* extension attributes.
Also import other functions that will be used later.

.. code-block:: python

    import spacy
    from narcy.nlp import spacy_ext
    from narcy.processors import doc_to_relations_df, doc_to_svos_df
    from narcy.nlp.utils import make_doc

Next, load text of some document and load *Spacy* NLP object with a language model.

Better models - like ``'en_core_web_md'`` - may be necessary for proper word vectors.

.. code-block:: python

    text = load_text()
    # Load NLP object and proper language model.
    nlp = spacy.load('en_core_web_sm')
    # Make document object with automatic normalization
    doc = make_doc(nlp, text)

Now, tidy data may be extracted from the document.

.. code-block:: python

    relations_df = doc_to_relations_df(doc)
    relation_reducts_df = doc_to_relations_df(doc, reduced=True)
    svos_df = doc_to_svos_df(doc)

Voila!


Data specification
==================

Tokens
------
For better capturing of the true semantics of texts, *Narcy* operates over
compound tokens. Compound tokens may correspond to single words
(they actually do in most of cases) and to spans composed of multiple words.
Multi-word tokens are mostly compound verbs and nouns.

Compound verbs are expressions like *have seen* or *going at*.
This approach allows capturing phrasal verbs and determining proper
tense of (sub)sentences.

Compound nouns honor *compoundedness* relations between nouns as detected
by the *Spacy* syntactic dependency tagger. This allows for treating expressions
like *climate change* as homogenous nouns related to specific entities
with clear semantics.

Heads and subs
--------------
Relations consists of **heads** and **subs** (subordinate parts). Heads
are objects that are superordinate in regard to subs in a *semantic-like* fashion.
This means here, that in most of cases a *head-sub* relationship follows
the syntactic structure of the sentence at hand. However, there is one important
exception: subjects are considered *heads* in regard to verbs.
This is important, since one the ideas behind the narrative analysis
is to try to determine the action flow in texts. Hence, verbs are associated
with actions and subjects with actants that perform those actions.

Leads and lemmas
----------------
Lead in the case of non-verbs is just a (compound) token itself.
However, lead of a verb is its semantic part (the driving, that is, the final
verb token in a compound verb + optional particle ending).

Lematization in *Narcy* always operates on leads.


Relations
---------
Rows in relation data frames describe atomic relations between various tokens.
They are described by the following features:

``head_tense``
    Tense (``PAST``, ``PRESENT`` or ``FUTURE``) of the relation head.
    Usually this is the tense that should be used in analyses.

``head_mode``
    Mode (``NORMAL`` or ``MODAL``) of the relation head.
    This is the mode of interest in most of cases.

``sub_tense``
    Tense of the relation sub.

``sub_mode``
    Mode of the relation sub.

``rtype``
    Relation type. There are the following types:

    ``verb-verb``
        Relation between two verbs.
        Head is superordinate and sub is subordinate
        in the parse tree of a sentence.

    ``subject-verb``
        Subject and verb.
        This is to be interpreted in terms of an action performed by an actant.

    ``verb-object``
        Object of a performed action.
        This the right side of a subject-object-triple.

    ``complement-verb``
        Action connected to a complement.

    ``verb-complement``
        Complement of a verb (action).

    ``left_adposition``
        Adposition. It may be connected to tokens of any type.
        Adposition introduce additional contextual information
        concerning things like time and/or space locations of events etc.
        They also link related subsentences.

        Left adposition designates the subordinate of the head of the
        corresponding right adposition.

    ``right_adposition``
        See ``left_adposition``.

        Right adposition designates the head of the corresponding left adposition.

    ``compound``
        Two noun-tokens constituting a compound noun token.

    ``noun-noun``
        Two nouns in a descriptive relation.
        For instance, "John Smith, school president".

    ``description``
        Description relation.
        The head is described by the sub.

    ``misc``
        Other types of relations.
        They can be safely discarded in most of cases.

``head``
    Raw text of the relation head.

``sub``
    Raw text of the relation sub.

``head_lead``
    Text of the lead of the relation head.

``sub_lead``
    Text of the lead of the relation sub.

``head_lemma``
    Lematized text of the lead of the relation head.

``sub_lemma``
    Lematized text of the lead of the relation sub.

``head_neg``
    Head negation flag.

``sub_neg``
    Sub negation flag.

``head_pos``
    Head POS tag.

``head_dep``
    Head syntactic dependency tag.

``sub_pos``
    Sub POS tag.

``sub_dep``
    Sub syntactic dependency tag.

``head_ent``
    Flag that indicates whether the head is part of a named entity.

``head_ent_label``
    Entity label for the head.

``sub_ent``
    Flag that indicates whether the sub is part of a named entity.

``sub_ent_label``
    Entity label for the sub.

``head_vector_norm``
    L2 norm of a word vector associated with the head.

``sub_vector_norm``
    L2 norm of a word vector associated with the sub.

``head_vector``
    Word vector associated with the head (about 300 dimensions).

``sub_vector``
    Word vector associated with the sub (about 300 dimensions).

``head_start``
    Index of the beginning of the head token-span in the document.

``head_end``
    Index of the end of the head token-span in the document.

``sub_start``
    Index of the beginning of the sub token-span in the document.

``sub_ent``
    Index of the end of the sub token-span in the document.

``docid``
    Document id based on MD5 hash of its content.
    Computed only once per document.

``sentid``
    Document id appended with start and end indexes of the sentence.
    It uniqualy identifies each sentence within a corpus of documents.


Relation reducts
----------------
They work the same as relations. The only difference is that ``misc`` relations
are discarded whatsoever and *adpositions* are removed and their subs are
transfered to their heads.


Subject-verb-object triplets
----------------------------
Rows in *SVO* data frames describe unique *subject-verb-object* triplets.
They use the following features:

``tense``
    Tense of the verb.

``mode``
    Mode of the verb.

``neg``
    Verb negation flag.

``rtype``
    Relation type. It is either ``svo`` (*subject-verb-object* triplet)
    or ``svc`` (*subject-verb-complement* triplet).
    Some verbs are not associated with a specific object but only a complement.

``subj``
    Raw text of the subject token.

``verb``
    Raw text of the verb token.

``obj``
    Raw text of the object/complement token.

``subj_lead``
    Text of the lead of the subject token.

``verb_lead``
    Text of the lead of the verb token.

``obj_lead``
    Text of the lead of the object token.

``subj_lemma``
    Lematized text of the lead of the subject token.

``verb_lemma``
    Lematized text of the lead of the verb token.

``obj_lemma``
    Lematized text of the lead of the object token.

``subj_ent``
    Flag indicating if the subject token is a part of a named entity.

``subj_ent_label``
    Entity label for the subject token.

``obj_ent``
    Flag inidicating if the object token is a part of a named entity.

``obj_ent_label``
    Entity label for the object token.

``subj_terms``
    Terms describing the subject.
    Terms are all semantic tokens that are subordinate in the parse tree
    in regards to some head token.

``obj_terms``
    Terms describing the object.

``subj_vector_norm``
    L2 norm of a word vector associated with the subject token.

``verb_vector_norm``
    L2 norm of a word vector associated with the verb token.

``obj_vector_norm``
    L2 norm of a word vector associated with the object token.

``subj_vector``
    Word vector associated with the subject token.

``verb_vector``
    Word vector associated with the verb token.

``obj_vector``
    Word vector associated with the object token.

``docid``
    Document id.

``sentid``
    Sentence id.



.. _Spacy: https://spacy.io/
