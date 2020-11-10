from pathlib import Path
from typing import List

import numpy as np
import torch
from transformers import BertModel


TOKEN_CLS = '[CLS]_'
TOKEN_PAD = '<pad>_'
MAX_INPUT_LENGTH = 512


class CuBertWrapper:
    def __init__(self, model_path: Path, vocab_path: Path, batch_size: int = 64, device: str = 'cuda'):
        self.batch_size = batch_size
        self.model = BertModel.from_pretrained(model_path).to(device)
        with vocab_path.open() as vocab_file:
            self.token_ids = {token[1:-2]: token_id for token_id, token in enumerate(vocab_file)}
        self.cls_ti = self.token_ids[TOKEN_CLS]
        self.pad_ti = self.token_ids[TOKEN_PAD]
        self.device = device

    def _pad_batch(self, batch: List[List[str]]) -> (torch.BoolTensor, torch.LongTensor):
        batch_width = min(MAX_INPUT_LENGTH, max(map(len, batch)))
        cropped_batch = [line[:batch_width] for line in batch]
        attention_mask = torch.zeros(len(batch), batch_width, dtype=torch.bool)
        padded_batch = torch.zeros(len(batch), batch_width, dtype=torch.long)
        for i, line in enumerate(cropped_batch):
            attention_mask[i, :len(line)] = True
            padded_batch[i, :len(line)] = torch.LongTensor(line)
            padded_batch[i, len(line):] = self.pad_ti
        return attention_mask, padded_batch

    def _process_batch(self, batch: List[List[str]]) -> (List[np.ndarray], np.ndarray):
        batch_tis = [
            [self.cls_ti, *map(self.token_ids.get, line)]
            for line in batch
        ]
        attention_mask, padded_batch = self._pad_batch(batch_tis)
        attention_mask = attention_mask.to(self.device)
        padded_batch = padded_batch.to(self.device)
        with torch.no_grad():
            token_vectors, line_vectors = self.model(padded_batch, attention_mask=attention_mask)
            token_vectors, line_vectors = token_vectors.cpu().numpy(), line_vectors.cpu().numpy()
        token_vectors = [tv[:am.sum()] for tv, am in zip(token_vectors, attention_mask)]
        return token_vectors, line_vectors

    def __call__(self, tokens: List[List[str]]) -> (List[np.ndarray], List[np.ndarray]):
        batches = [tokens[batch_start:batch_start + self.batch_size] for batch_start in
                   range(0, len(tokens), self.batch_size)]
        token_vectors, line_vectors = [], []
        for batch in batches:
            batch_token_vectors, batch_line_vectors = self._process_batch(batch)
            token_vectors.extend(batch_token_vectors)
            line_vectors.extend(batch_line_vectors)
        return token_vectors, line_vectors
