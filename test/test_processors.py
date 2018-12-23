"""Unit tests for processors."""
import pytest
import pandas as pd
from narcy.processors import reduce_relations
from narcy.processors import doc_to_relations_df, doc_to_svos_df
from narcy.nlp.utils import Relation


@pytest.mark.parametrize('reduced', [True, False])
def test_relations(doc, reduced):
    relations = doc._.relations
    if reduced:
        relations = reduce_relations(relations)
    for relation in relations:
        assert isinstance(relation, Relation)

@pytest.mark.parametrize('reduced', [True, False])
def test_doc_to_relations_df(doc, reduced):
    df = doc_to_relations_df(doc, reduced=reduced)
    expected_shape = (867, 31) if reduced else (1233, 31)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == expected_shape

def test_doc_to_svos_df(doc):
    df = doc_to_svos_df(doc)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (75, 27)
