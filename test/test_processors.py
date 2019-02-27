"""Unit tests for processors."""
import pytest
from . import get_docs
from . import _test_relations, _test_doc_to_relations_df
from . import _test_doc_to_svos_df, _test_doc_to_tokens_df


docs = get_docs()

@pytest.mark.parametrize('doc', docs)
@pytest.mark.parametrize('reduced', [True, False])
def test_relations(doc, reduced):
    _test_relations(doc, reduced)

@pytest.mark.parametrize('doc', docs)
@pytest.mark.parametrize('reduced', [True, False])
def test_doc_to_relations_df(doc, reduced):
    _test_doc_to_relations_df(doc, reduced)


@pytest.mark.parametrize('doc', docs)
def test_doc_to_svos_df(doc):
    _test_doc_to_svos_df(doc)

@pytest.mark.parametrize('doc', docs)
def test_doc_to_tokens_df(doc):
    _test_doc_to_tokens_df(doc)
