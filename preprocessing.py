import re
from typing import Tuple

import regex
from nltk import sent_tokenize


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


def tokenized_sent_to_str(tokenized_sent) -> str:
    # remove redundant spaces around sentence delimiters
    for p in ".,?!:;":
        tokenized_sent = tokenized_sent.replace(f' {p}', f'{p}')
    return tokenized_sent


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
    # Swap tabs for single spaces
    doc_merged = doc_merged.replace('\t', ' ')
    return doc_merged


class Tokenize:
    def __init__(self, doc):
        self.sentences = sent_tokenize(doc)

    def remove_short_sentences(self, num_words: int = 3):
        self.sentences = [s for s in self.sentences if len(re.findall(r'\w+', s)) >= num_words]

    def lowercase_sentences(self):
        for i, s in enumerate(self.sentences):
            self.sentences[i] = s.lower()

    def remove_ultra_long_sentences(self, tokens: int = 50):
        pass

    def __str__(self):
        d = ""
        for i, sentence in enumerate(self.sentences):
            d += tokenized_sent_to_str(sentence) + ' '
        d = d.strip()
        return d


def preprocess(doc: str, is_lower: bool = True) -> Tuple[str, Tokenize]:
    # Remove upper-cased lines
    doc = "\n".join([l for l in doc.split('\n') if not l.isupper()]).strip()
    # Clean the data
    doc = clean(doc)
    # Split content document into sentences
    doc_tokenized = Tokenize(doc)
    # Lowercase document and sentences
    if is_lower:
        doc_tokenized.lowercase_sentences()
    # Remove super short sentences with less than 3 words
    doc_tokenized.remove_short_sentences(num_words=3)
    return str(doc_tokenized), doc_tokenized


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

main()
