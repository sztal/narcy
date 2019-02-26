"""Unit tests for processors."""
import pytest
from . import _test_relations, _test_doc_to_relations_df, _test_doc_to_svos_df


@pytest.mark.parametrize('reduced', [True, False])
def test_relations(docs, reduced):
    _test_relations(docs, reduced)

@pytest.mark.parametrize('reduced', [True, False])
def test_doc_to_relations_df(docs, reduced):
    _test_doc_to_relations_df(docs, reduced)

def test_doc_to_svos_df(docs):
    _test_doc_to_svos_df(docs)
