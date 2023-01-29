import unittest

from preprocessing import clean, merge_characters, preprocess


class TestPreprocessing(unittest.TestCase):

    def test_currency(self):
        doc = '¥160m (£680k). 9.0¢ 10£ million'
        self.assertEqual(doc, clean(doc))

    def test_end_of_line_hyphenation(self):
        doc = '''
            enter-
            tainment
        '''
        self.assertEqual('entertainment', clean(doc))
        doc = 'self-assessment'
        self.assertEqual('self-assessment', clean(doc))

    def test_unicode_symbols(self):
        doc = 'Our  business'
        self.assertEqual('Our business', clean(doc))
        doc = '67,000 Oz®•‡'
        self.assertEqual('67,000 Oz', clean(doc))
        # doc = 'Ë'
        # self.assertEqual('E', clean(doc))

    def test_merging_characters(self):
        doc = "T h e \t G r o u p ’ s "
        self.assertEqual("The\tGroup’s", merge_characters(doc))
        doc = 'pa yment \t r eserv e'
        self.assertEqual("payment\treserve", merge_characters(doc))
        doc = 'He	is	currently	the	Chairman'
        self.assertEqual(doc, merge_characters(doc))

    def test_apostrophes(self):
        doc = "The Group’s"
        self.assertEqual("The Group's", clean(doc))
        doc = "The Group 's"
        self.assertEqual("The Group's", clean(doc))

    def test_spaces(self):
        doc = " The   Group "
        self.assertEqual("The Group", clean(doc))

    def test_emails(self):
        doc = "info@acalplc.co.uk"
        self.assertEqual("", clean(doc))

    def test_urls(self):
        doc = "WWW.ACALPLC.CO.UK"
        self.assertEqual("", clean(doc))
        doc = 'www.acalplc.co.uk'
        self.assertEqual("", clean(doc))
        doc = 'WWW.3LEGSRESOURCES.COM'
        self.assertEqual("", clean(doc))

    def test_hours(self):
        doc = '12:26'
        self.assertEqual("", clean(doc))

    def test_uk_phones(self):
        doc = '+44 1624 811 611'
        self.assertEqual("", clean(doc))
        doc = '+44 121 415 7047'
        self.assertEqual("", clean(doc))
        doc = '(01582) 723131'
        self.assertEqual("", clean(doc))
        # doc = '+44 (0) 1785 715772'  # does not work
        # self.assertEqual("", clean(doc))

    def test_dates(self):
        doc = '23/04/2010'
        self.assertEqual("", clean(doc))

    def test_short_sentences(self):
        doc = "Next sentence will be removed. This one."
        doc_preprocessed_str, _ = preprocess(doc, is_lower=False, models_path='../resources/en_ewt_models', use_stanza=True)
        self.assertEqual(doc_preprocessed_str, "Next sentence will be removed.")
        doc = "Next sentence will be removed. a."
        doc_preprocessed_str, _ = preprocess(doc, is_lower=False, models_path='../resources/en_ewt_models', use_stanza=True)
        self.assertEqual(doc_preprocessed_str, "Next sentence will be removed.")

    def test_single_words_per_line(self):
        doc = "Sentence to stay.\nTOBEREMOVED\n"
        doc_preprocessed_str, _ = preprocess(doc, is_lower=False)
        self.assertEqual(doc_preprocessed_str, "Sentence to stay.")
        doc = "To stay.\nTOBEREMOVED\n"  # too short sentences
        doc_preprocessed_str, _ = preprocess(doc, is_lower=False)
        self.assertEqual(doc_preprocessed_str, "")


    def test_uppercased_sent(self):
        doc = "NETWORKING INTERNET SERVICE AGGREGATION (NOT IP) (NOT IP)\n"
        doc_preprocessed_str, _ = preprocess(doc, is_lower=False)
        self.assertEqual(doc_preprocessed_str, "")
