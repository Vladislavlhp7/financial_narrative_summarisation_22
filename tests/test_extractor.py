from unittest import TestCase

import numpy as np
import torch

from src.extractor import pad_batch


class TestExtractor(TestCase):
    def test_pad_batch(self):
        # sent 1
        word_tensor_1 = torch.randn((1, 10), dtype=torch.float)
        word_tensor_2 = torch.randn((1, 10), dtype=torch.float)
        sent1 = torch.cat([word_tensor_1, word_tensor_2])
        # sent 2
        word_tensor_3 = torch.randn((1, 10), dtype=torch.float)
        sent2 = torch.FloatTensor(word_tensor_3)

        # pad the batch
        sent_list = pad_batch([sent1, sent2])
        sent_tensor = torch.stack(sent_list)

        # expected sent tensor
        sent_tensor_exp = torch.stack(pad_batch([
            sent1,  # sent 1
            torch.cat([word_tensor_3, torch.zeros((1,10), dtype=torch.float)])]  # sent 2
        ))
        self.assertTrue(torch.all(torch.eq(sent_tensor_exp,sent_tensor)))
