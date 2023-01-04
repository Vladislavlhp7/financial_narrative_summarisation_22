import unittest

from thesis.preprocessing import clean, merge


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

    def test_unicode_symbols(self):
        doc = 'Our  business'
        self.assertEqual('Our business', clean(doc))
        doc = '67,000 Oz®•‡'
        self.assertEqual('67,000 Oz', clean(doc))

    def test_merging(self):
        doc = "T h e  G r o u p ’ s "
        self.assertEqual("The  Group’s ", merge(doc))

    def test_apostrophes(self):
        doc = "The Group’s"
        self.assertEqual("The Group's", clean(doc))

    def test_spaces(self):
        doc = " The   Group "
        self.assertEqual("The Group", clean(doc))
