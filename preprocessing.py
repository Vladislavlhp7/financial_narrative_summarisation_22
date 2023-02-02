import os.path
import re

import regex
from typing import Tuple
from nltk import sent_tokenize

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


def tokenized_sent_to_str(tokenized_sent, use_stanza: bool = False) -> str:
    # connect tokens into sentences
    if use_stanza:
        txt = str(' '.join(t.text for t in tokenized_sent.words))
    else:
        txt = tokenized_sent
    # remove redundant spaces around sentence delimiters
    for p in ".,?!:;":
        txt = txt.replace(f' {p}', f'{p}')
    return txt


def clean(doc: str):
    doc = doc.replace('\n', ' ')
    # remove duplicated spaces
    doc = " ".join(doc.split())
    # reconnect words split by end-of-line hyphenation with lookbehind
    doc = regex.sub(r"(?<=[a-z])-\s", '', doc)
    # remove emails, urls, hours, UK phones, dates
    doc = doc.replace('WWW.', 'www.')  # ensure url starts lowercase to be caught by regex below
    reg_to_drop = r"""(?x)
        (?:[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*|"(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21\x23-\x5b\x5d-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])*")@(?:(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?|\[(?:(?:(2(5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9]))\.){3}(?:(2(5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9])|[a-z0-9-]*[a-z0-9]:(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21-\x5a\x53-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])+)\])
        | (https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})
        | ^([0-1]?[0-9]|2[0-3])(:[0-5][0-9])+
        | ^(((\+44\s?\d{4}|\(?0\d{4}\)?)\s?\d{3}\s?\d{3})|((\+44\s?\d{3}|\(?0\d{3}\)?)\s?\d{3}\s?\d{4})|((\+44\s?\d{2}|\(?0\d{2}\)?)\s?\d{4}\s?\d{4}))(\s?\#(\d{4}|\d{3}))?$
        | ^(0?[1-9]|[12][0-9]|3[01])[\/\-](0?[1-9]|1[012])[\/\-]\d{4}$
    """
    pattern = regex.compile(reg_to_drop, regex.UNICODE)
    doc = pattern.sub("", doc)
    # Remove non-alphanumeric and non-special financial characters
    reg_to_drop = r'''(?x) # flag to allow comments and multi-line regex
            [^\w_ |        # alpha-numeric
            \p{Sc} |       # currencies
            \%\&\"\\'\’\(\)\.\,\?\!\-\;\\\/ ]  # preserve apostrophe `’`
        '''
    pattern = regex.compile(reg_to_drop, regex.UNICODE)
    doc = pattern.sub("", doc)
    doc = doc.replace("’", "'")  # replace special unicode apostrophe with normal one
    # Deal with unmerged apostrophes
    apostrophes = r"\s*\'s"
    pattern = regex.compile(apostrophes, regex.UNICODE)
    doc = pattern.sub("'s", doc)
    # remove duplicated spaces after dropping special symbols
    doc = " ".join(doc.split())
    # remove redundant spaces around sentence delimiters
    for p in ".,?!:;":
        doc = doc.replace(f' {p}', f'{p}')
    # normalise accents and umlauts
    # doc = unidecode(doc)  # unfortunately normalizes currencies as well
    return doc


def merge_characters(doc: str):
    """
        Try to merge split characters into words
    :param doc:
    :return:
    """
    # Rule-based merging
    doc_merged = ""
    doc_split = doc.split('\n')
    max_lines = len(doc_split)
    # Line-level operation is safer and more flexible
    # as in some reports only a few lines require character merging
    for i, line in enumerate(doc_split):
        line_new = str(line)
        # If there is a `\ \t \ ` assume tabs are spaces and delete spaces
        if regex.findall(pattern=r'(?<=\w)\ \t\ ', string=line):
            pattern = regex.compile(r'\ ')
            line_new = pattern.sub("", line)
        if i != max_lines - 1:  # control final end-line
            line_new += '\n'
        doc_merged += line_new
    return doc_merged


class Tokenize:
    def __init__(self, doc):
        self.sentences = sent_tokenize(doc)


def preprocess(doc: str, use_stanza: bool = False,
               is_lower: bool = True,
               models_path: str = 'resources/en_ewt_models') -> Tuple[str, stanza.Document]:
    # Remove lines with less than 2 non-digit words
    doc = "\n".join([l for l in doc.split('\n') if len(re.findall(r"[a-zA-Z]+'?[a-zA-Z]+", l)) > 1]).strip()
    # Remove upper-cased lines
    doc = "\n".join([l for l in doc.split('\n') if not l.isupper()]).strip()
    # Clean the data
    doc = clean(doc)
    # Remove super short sentences with less than 3 words
    doc = " ".join([s for s in sent_tokenize(doc) if len(re.findall(r'\w+', s)) >= 3]).strip()
    # Split content document into sentences
    if not use_stanza:
        doc_tokenized = Tokenize(doc)
    else:
        models_dir = "/".join(models_path.split('/')[:-1])
        os.makedirs(models_dir, exist_ok=True)
        if not os.path.exists(models_path):
            stanza.download('en', models_dir)
        nlp = stanza.Pipeline(processors='tokenize', lang='en', models_dir=models_dir, treebank='en_ewt',
                              download_method=None)
        doc_tokenized = nlp(doc)
    doc_preprocessed_str = ""
    for i, sentence in enumerate(doc_tokenized.sentences):
        doc_preprocessed_str += tokenized_sent_to_str(sentence, use_stanza=use_stanza) + ' '
    doc_preprocessed_str = doc_preprocessed_str.strip()
    # Lowercase document and sentences
    if is_lower:
        doc_preprocessed_str = doc_preprocessed_str.lower()
        for i, s in enumerate(doc_tokenized.sentences):
            doc_tokenized.sentences[i] = s.lower()
    return doc_preprocessed_str, doc_tokenized


def main():
    doc = """
    I n 	 a d d i t i o n , 	 X T A Q ’ s 	 s a l e s , 	 d e v e l o p m e n t 	 a n d 	 i m p l e m e n t a t i o n 	 s t a f f	
w i l l 	 s t r e n g t h e n 	 t h e 	 C o m p a n y’ s 	 e x i s t i n g 	 t e a m 	 w i t h 	 a 	 n u m b e r 	 o f	
b u d g e t e d 	 n e w 	 p o s i t i o n s 	 n o w 	 b e i n g 	 f i l l e d 	 b y 	 X T A Q ’ s 	 s t a f f . 	 I n 	 t h i s	
w a y 	 t h e 	 e n l a r g e d 	 G r o u p 	 w i l l 	 b e n e f i t 	 f r o m 	 t h e 	 e c o n o m i e s 	 o f 	 s c a l e	
r e s u l t i n g 	 f r o m 	 t h e 	 m e r g e r 	 o f 	 t h e 	 t w o 	 companies.
    """
    print(merge_characters(doc))
    print(clean(merge_characters(doc)))
    doc = """
    Operational highlights
• Continued primary focus, with ConocoPhillips, on our three western 
concessions in the Polish Baltic Basin, which we believe represent 
some of the most prospective shale acreage in Poland"""
    print(merge_characters(doc))
    print(clean(merge_characters(doc)))

# main()
