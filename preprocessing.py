import os.path
import re
from typing import Tuple

import stanfordnlp


def clean_company_name(line: str):
    line = line.strip()
    # Clear of irrelevant strings
    reg_to_drop = r'''(?x) # flag to allow comments and multi-line regex
            Annual | Report | Accounts | Financial | Statements | Chairman | Executive
    '''
    pattern = re.compile(reg_to_drop, re.IGNORECASE)
    line = pattern.sub("", line)
    # Extract the name of the company
    name = line.split('plc')[0] + ' plc '
    # Try to match the year on the line and add to the identifier
    year = re.findall(r'\d{4}', line)
    if year:
        name += year[0]
    # Ensure unnecessary spaces are removed
    name = " ".join(name.split())
    return name


def tokenized_sent_to_str(tokenized_sent):
    return ' '.join(t.text for t in tokenized_sent.words)


def preprocess(doc: str) -> Tuple[str, stanfordnlp.Document]:
    MODELS_DIR = 'resources'
    os.makedirs(MODELS_DIR, exist_ok=True)
    MODELS_PATH = 'resources/en_ewt_models'
    if not os.path.exists(MODELS_PATH):
        stanfordnlp.download('en', MODELS_DIR)
    # Split content document into sentences
    nlp = stanfordnlp.Pipeline(processors='tokenize', lang='en', models_dir=MODELS_DIR, treebank='en_ewt')
    doc_tokenized = nlp(doc)
    doc_preprocessed_str = ""
    for i, sentence in enumerate(doc_tokenized.sentences):
        doc_preprocessed_str += tokenized_sent_to_str(sentence) + ' '
    return doc_preprocessed_str, doc_tokenized
