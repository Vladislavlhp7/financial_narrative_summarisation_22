import os.path
import regex
from typing import Tuple

import stanza


def clean_company_name(line: str):
    line = line.strip()
    # Clear of irrelevant strings
    reg_to_drop = r'''(?x) # flag to allow comments and multi-line regex
            Annual | Report | Accounts | Financial | Statements | Chairman | Executive
    '''
    pattern = regex.compile(reg_to_drop, regex.IGNORECASE)
    line = pattern.sub("", line)
    # Extract the name of the company
    name = line.split('plc')[0] + ' plc '
    # Try to match the year on the line and add to the identifier
    year = regex.findall(r'\d{4}', line)
    if year:
        name += year[0]
    # Ensure unnecessary spaces are removed
    name = " ".join(name.split())
    return name


def tokenized_sent_to_str(tokenized_sent):
    return ' '.join(t.text for t in tokenized_sent.words)


def clean(doc: str):
    doc = doc.replace('\n', ' ')
    # remove duplicated spaces
    doc = " ".join(doc.split())
    # reconnect words split by end-of-line hyphenation with lookbehind
    doc = regex.sub(r"(?<=[a-z])-\s", '', doc)
    # Remove non-alphanumeric and non-special financial characters
    reg_to_drop = r'''(?x) # flag to allow comments and multi-line regex
            [^\w_ |        # alpha-numeric
            \p{Sc} |       # currencies
            \%\&\'\"\(\)\.\,\?\!\-\;\\\/ ]
        '''
    pattern = regex.compile(reg_to_drop, regex.UNICODE)
    doc = pattern.sub("", doc)
    # remove duplicated spaces after dropping special symbols
    doc = " ".join(doc.split())
    return doc


def preprocess(doc: str) -> Tuple[str, stanza.Document]:
    MODELS_DIR = 'resources'
    os.makedirs(MODELS_DIR, exist_ok=True)
    MODELS_PATH = 'resources/en_ewt_models'
    if not os.path.exists(MODELS_PATH):
        stanza.download('en', MODELS_DIR)
    # Clean the data
    doc = clean(doc)
    # Split content document into sentences
    nlp = stanza.Pipeline(processors='tokenize,ner,mwt,pos', lang='en', models_dir=MODELS_DIR, treebank='en_ewt')
    doc_tokenized = nlp(doc)
    doc_preprocessed_str = ""
    for i, sentence in enumerate(doc_tokenized.sentences):
        doc_preprocessed_str += tokenized_sent_to_str(sentence) + ' '
    return doc_preprocessed_str, doc_tokenized
