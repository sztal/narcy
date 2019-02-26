"""Unit tests for custom _Spacy_ getters."""
# pylint: disable=unused-import,redefined-outer-name
import pytest
import en_core_web_sm
from narcy import spacy_ext, make_doc
from narcy import doc_to_relations_df, doc_to_svos_df


@pytest.fixture(scope='session')
def nlp():
    return en_core_web_sm.load()

@pytest.fixture(scope='session')
def small_docs(nlp):
    return (
        (make_doc(nlp, "I recon he's very angry on you."), 5),
        (make_doc(nlp, "This is not a spider's web."), 3)
    )

def test_lemma(small_docs):
    for doc, nrow in small_docs:
        df = doc_to_relations_df(doc)
        assert df.shape[0] == nrow
