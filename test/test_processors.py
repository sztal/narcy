"""Unit tests for processors."""
import pytest
import pandas as pd
from narcy.processors import reduce_relations
from narcy.processors import doc_to_relations_df, doc_to_svos_df
from narcy.nlp.utils import Relation


@pytest.mark.parametrize('reduced', [True, False])
def test_relations(docs, reduced):
    for doc in docs:
        relations = doc._.relations
        if reduced:
            relations = reduce_relations(relations)
        for relation in relations:
            assert isinstance(relation, Relation)

@pytest.mark.parametrize('reduced', [True, False])
def test_doc_to_relations_df(docs, reduced):
    for doc in docs:
        df = doc_to_relations_df(doc, reduced=reduced)
        assert isinstance(df, pd.DataFrame)
        assert df.shape != (0, 0)

def test_doc_to_svos_df(docs):
    for doc in docs:
        df = doc_to_svos_df(doc)
        assert isinstance(df, pd.DataFrame)
        assert df.shape != (0, 0)
