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
            \%\&\"\\'\’\(\)\.\,\?\!\-\;\\\/ ]  # preserve apostrophe `’`
        '''
    pattern = regex.compile(reg_to_drop, regex.UNICODE)
    doc = pattern.sub("", doc)
    doc = doc.replace("’", "'")  # replace special unicode apostrophe with normal one
    # remove duplicated spaces after dropping special symbols
    doc = " ".join(doc.split())
    return doc


def merge(doc: str):
    """
        Try to merge split characters into words
    :param doc:
    :return:
    """
    # Remove end-line spacing universal in split words cases
    reg_to_drop = r'(?<=\w)(\t\n)(?=\w)'
    pattern = regex.compile(reg_to_drop)
    doc = pattern.sub(" \t ", doc)
    reg_to_merge =r'''        (?x)   # flag to allow comments and multi-line regex
                (?<=\w)\ (?=\w\ )    # Case 1 - 'T h e 	 o b j e c t i v e'
                | (?<=\’)\s          # Case 2 - apostrophes ’s
                | \s(?=\’)           # Case 2 - apostrophes <ENT>’s
    '''
    pattern = regex.compile(reg_to_merge)
    doc = pattern.sub("", doc)
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


def main():
    doc = """
    I n 	 a d d i t i o n , 	 X T A Q ’ s 	 s a l e s , 	 d e v e l o p m e n t 	 a n d 	 i m p l e m e n t a t i o n 	 s t a f f	
w i l l 	 s t r e n g t h e n 	 t h e 	 C o m p a n y’ s 	 e x i s t i n g 	 t e a m 	 w i t h 	 a 	 n u m b e r 	 o f	
b u d g e t e d 	 n e w 	 p o s i t i o n s 	 n o w 	 b e i n g 	 f i l l e d 	 b y 	 X T A Q ’ s 	 s t a f f . 	 I n 	 t h i s	
w a y 	 t h e 	 e n l a r g e d 	 G r o u p 	 w i l l 	 b e n e f i t 	 f r o m 	 t h e 	 e c o n o m i e s 	 o f 	 s c a l e	
r e s u l t i n g 	 f r o m 	 t h e 	 m e r g e r 	 o f 	 t h e 	 t w o 	 companies.
    """
    print(merge(doc))
    print(clean(merge(doc)))

# main()
