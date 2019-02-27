"""Extensions attributes defined on the *Spacy* classes."""
# pylint: disable=E0611
import re
from collections import defaultdict
from spacy.tokens import Token, Span, Doc
from . import getters
from ..utils import make_hash

Doc.set_extension('_id', default=None)
Doc.set_extension('_polarity', default=None)
Span.set_extension('_polarity', default=None)

extensions = defaultdict(lambda: defaultdict(dict))
for name in dir(getters):
    if re.search(r"_[gsm]$", name):
        ext_name = name[-4:]
        attr_name = name[:-4]
        func = getattr(getters, name)
        if ext_name.startswith('_t'):
            extcls = extensions['token']
        elif ext_name.startswith('_s'):
            extcls = extensions['span']
        elif ext_name.startswith('_d'):
            extcls = extensions['doc']
        if ext_name.endswith('_g'):
            extcls[attr_name]['getter'] = func
        elif ext_name.endswith('_s'):
            extcls[attr_name]['setter'] = func
        elif ext_name.endswith('_m'):
            extcls[attr_name]['method'] = func

for attr_name, kwds in extensions['token'].items():
    Token.set_extension(attr_name, **kwds)
for attr_name, kwds in extensions['span'].items():
    Span.set_extension(attr_name, **kwds)
for attr_name, kwds in extensions['doc'].items():
    Doc.set_extension(attr_name, **kwds)
