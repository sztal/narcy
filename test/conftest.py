"""*PyTest* configuration and general purpose fixtures."""
# pylint: disable=W0611
import os
import pytest
import en_core_web_sm
from narcy.nlp import spacy_ext
from narcy.nlp.utils import make_doc

_dirpath = os.path.join(os.path.split(__file__)[0], 'data')

def pytest_addoption(parser):
    """Custom `pytest` command-line options."""
    parser.addoption(
        '--benchmarks', action='store_true', default=False,
        help="Run benchmarks (instead of tests)."
    )
    parser.addoption(
        '--slow', action='store_true', default=False,
        help="Run slow tests / benchmarks."""
    )

def pytest_collection_modifyitems(config, items):
    """Modify test runner behaviour based on `pytest` settings."""
    run_benchmarks = config.getoption('--benchmarks')
    run_slow = config.getoption('--slow')
    if run_benchmarks:
        skip_test = \
            pytest.mark.skip(reason="Only benchmarks are run with --benchmarks")
        for item in items:
            if 'benchmark' not in item.keywords:
                item.add_marker(skip_test)
    else:
        skip_benchmark = \
            pytest.mark.skip(reason="Benchmarks are run only with --run-benchmark")
        for item in items:
            if 'benchmark' in item.keywords:
                item.add_marker(skip_benchmark)
    if not run_slow:
        skip_slow = pytest.mark.skip(reason="Slow tests are run only with --slow")
        for item in items:
            if 'slow' in item.keywords:
                item.add_marker(skip_slow)


# Fixtures --------------------------------------------------------------------

@pytest.fixture(scope='session')
def doc():
    with open(os.path.join(_dirpath, 'cop24.txt')) as stream:
        text = stream.read()
    nlp = en_core_web_sm.load()
    document = make_doc(nlp, text)
    return document
