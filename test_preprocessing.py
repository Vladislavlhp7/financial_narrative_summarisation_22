import unittest
import wordninja

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

    def test_wordninja(self):
        doc = "T h e  G r o u p ' s".replace(' ', '')
        self.assertEqual("The Group's", " ".join(wordninja.split(doc)))
        doc = 'strongfoundation'
        self.assertEqual("strong foundation", " ".join(wordninja.split(doc)))
        doc = 'T h e 	 o b j e c t i v e 	 o f 	 A C S'.replace(' ', '')
        self.assertEqual("The objective of ACS", " ".join(wordninja.split(doc)))

    def test_merging(self):
        doc = "T h e  G r o u p "
        self.assertEqual("The  Group ", merge(doc))
