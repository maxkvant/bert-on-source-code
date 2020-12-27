from typing import Union, Iterable, List
from pathlib import Path
from tensor2tensor.data_generators import text_encoder
import re


class TokenToIndexConverter:

    def __init__(
        self,
        vocab_path: Union[str, Path],
        unk_token = '[UNK]_',
        bos_token = '\\u\\u\\uNL\\u\\u\\u_',
        eos_token = '\\u\\u\\uNEWLINE\\u\\u\\u_',
        pad_token = '<pad>_'
    ):
        vocab_path = Path(vocab_path)
        self.subword_tokenizer = text_encoder.SubwordTextEncoder(vocab_path.as_posix())
        self.token_to_index_map = {
            tok: i
            for i, tok in enumerate(self.subword_tokenizer.all_subtoken_strings)
        }
        self.index_to_token_map = {v: k for k, v in self.token_to_index_map.items()}
        self.unk_token = unk_token
        self.unk_index = self[unk_token]
        self.bos_token = bos_token
        self.bos_index = self[bos_token]
        self.eos_token = eos_token
        self.eos_index = self[eos_token]
        self.pad_token = pad_token
        self.pad_index = self[pad_token]
        self.util_tokens = set(self.subword_tokenizer.all_subtoken_strings[:5])

        self.bos_exp = re.compile(r"^(___NL___)*")  # Remove any number of repeating newline characters
        self.eos_exp = re.compile(r"___NEWLINE___.*$")  # Remove evrythin after first eod

    def __getitem__(self, key):
        return (self.token_to_index_map[key] 
                if key in self.token_to_index_map 
                else self.unk_index)

    def encode(self, tokens: str) -> List[int]:
        return [self[tok] for tok in tokens]
    
    def encode_code(self, code: List[List[str]]) -> List[int]:
        return [self.bos_index] + [
            tok for line in code 
            for tok in self.encode(line)
        ]
    
    def decode(self, tokens: List[int]) -> str:
        text = self.subword_tokenizer.decode(tokens, strip_extraneous=True)
        text = self.bos_exp.sub("", text)
        text = self.eos_exp.sub("", text)
        return text

    def decode_list(self, tokens: List[int]) -> List[str]:
        tokens = self.subword_tokenizer.decode_list(tokens)
        while tokens[0] == self.bos_token and len(tokens) != 0:
            tokens = tokens[1:]
        end_position = tokens.index(self.eos_token) if self.eos_token in tokens else -1 
        tokens = tokens[:end_position]
        return [tok for tok in tokens if tok not in self.util_tokens]

    @property
    def vocab_size(self):
        return self.subword_tokenizer.vocab_size
