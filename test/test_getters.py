"""Unit tests for custom _Spacy_ getters."""
# pylint: disable=unused-import,redefined-outer-name
import pytest
from pytest import approx
import en_core_web_sm
from narcy import spacy_ext, make_doc
from narcy import doc_to_relations_df, doc_to_svos_df


data = [
    ("I recon he's very angry on you.", 5, -0.2325334),
    ("This is not a spider's web.", 3, 0),
    ("You're a fucking douchebag", 3, -0.4417996),
    ("This is a great new development.", 4, 0.3161994)
]

@pytest.mark.parametrize('text,nrow,sentiment', data)
def test_lemma(text, nrow, sentiment, nlp):
    doc = make_doc(nlp, text)
    df = doc_to_relations_df(doc)
    assert df.shape[0] == nrow
    assert doc._.sentiment == approx(sentiment)
