from pathlib import Path

import numpy as np
import torch
from transformers import BertModel


TOKEN_CLS = '[CLS]_'
TOKEN_PAD = '<pad>_'
MAX_INPUT_LENGTH = 512


class BertWrapper:
    def __init__(self, model_path: Path, vocab_path: Path, batch_size: int = 64, device: str = 'cuda'):
        self.batch_size = batch_size
        self.model = BertModel.from_pretrained(model_path).to(device)
        with vocab_path.open() as vocab_file:
            self.token_ids = {token[1:-2]: token_id for token_id, token in enumerate(vocab_file)}
        self.cls_ti = self.token_ids[TOKEN_CLS]
        self.pad_ti = self.token_ids[TOKEN_PAD]
        self.device = device

    def _pad_batch(self, batch):
        batch_width = min(MAX_INPUT_LENGTH, max(map(len, batch)))
        cropped_batch = [line[:batch_width] for line in batch]
        attention_mask = np.zeros((len(batch), batch_width), np.bool)
        padded_batch = []
        for i, line in enumerate(cropped_batch):
            attention_mask[i, :len(line)] = True
            padded_batch.append(line + [self.pad_ti] * (batch_width - len(line)))
        return attention_mask, padded_batch

    def _process_batch(self, batch):
        batch_tis = [
            [self.cls_ti, *map(self.token_ids.get, line)]
            for line in batch
        ]
        attention_mask, padded_batch = self._pad_batch(batch_tis)
        attention_mask = torch.tensor(attention_mask, device=self.device)
        padded_batch = torch.tensor(padded_batch, device=self.device)
        with torch.no_grad():
            token_vectors, line_vectors = self.model(padded_batch)
        token_vectors = [tv[:am.sum()] for tv, am in zip(token_vectors, attention_mask)]
        line_vectors = list(line_vectors)
        return token_vectors, line_vectors

    def __call__(self, tokens):
        batches = [tokens[batch_start:batch_start + self.batch_size] for batch_start in
                   range(0, len(tokens), self.batch_size)]
        token_vectors, line_vectors = [], []
        for batch in batches:
            batch_token_vectors, batch_line_vectors = self._process_batch(batch)
            token_vectors.extend(batch_token_vectors)
            line_vectors.extend(batch_line_vectors)
        return token_vectors, line_vectors
