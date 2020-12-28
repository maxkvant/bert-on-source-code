from torch.utils.data import Dataset, DataLoader
from torch import Tensor, LongTensor
from torch.nn.utils.rnn import pad_sequence
from typing import Callable, Iterable, Optional


class SequenceToSequenceDataset(Dataset):

    def __init__(
        self,
        src_stream: Iterable['T'],
        src_encoder: Callable[['T'], Tensor],
        ref_stream: Iterable['T'],
        ref_encoder: Callable[['T'], Tensor],
        src_pad_index: int,
        ref_pad_index: Optional[int] = None
    ):
        self.src = [src_encoder(s) for s in src_stream]
        self.ref = [ref_encoder(s) for s in ref_stream]
        assert len(self.src) == len(self.ref)
        self.src_pad_index = src_pad_index
        self.ref_pad_index = ref_pad_index if ref_pad_index is not None else src_pad_index

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return self.src[idx], self.ref[idx]

    def collate_fn(self, data):
        src_batch, ref_batch = zip(*data)
        input_ids = pad_sequence(
            src_batch,
            padding_value=self.src_pad_index,
            batch_first=True
        )
        attention_mask = input_ids != self.src_pad_index
        decoder_input_ids = pad_sequence(
            ref_batch,
            padding_value=self.ref_pad_index,
            batch_first=True
        )
        labels = decoder_input_ids[:,1:]
        decoder_input_ids = decoder_input_ids[:,:-1]
        decoder_attention_mask = decoder_input_ids != self.ref_pad_index
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'decoder_input_ids': decoder_input_ids,
            'decoder_attention_mask': decoder_attention_mask,
            'labels': labels
        }

    def make_loader(self, *args, **kwargs):
        return DataLoader(self, *args, collate_fn=self.collate_fn, **kwargs)


def get_method_name_dataset(data, token_to_index, pad_index, max_length):

    def truncated_encoder(encoder, max_length):
        def wrapper(*args, **kwargs):
            return encoder(*args, **kwargs)[:max_length]
        return wrapper

    def to_torch_encoder(encoder):
        def wrapper(*args, **kwargs):
            return LongTensor(encoder(*args, **kwargs))
        return wrapper

    return SequenceToSequenceDataset(
        src_stream = (e['function_body_tokenized'] for e in data),
        src_encoder = to_torch_encoder(
            truncated_encoder(token_to_index.encode_code, max_length)
        ),
        ref_stream = (e['function_name_tokenized'] for e in data),
        ref_encoder = to_torch_encoder(
            truncated_encoder(token_to_index.encode_code, max_length)
        ),
        src_pad_index = pad_index
    )
