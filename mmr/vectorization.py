from typing import List

import numpy as np

from cubert_wrapper import CuBertWrapper


class AvgLineVectorization:
    def __init__(self, bert_wrapper: CuBertWrapper):
        self.bert_wrapper = bert_wrapper
        
    def __call__(self, tokens: List[List[str]]) -> np.ndarray:
        _, line_vectors = self.bert_wrapper(tokens)
        return np.array(line_vectors).mean(axis=0)
