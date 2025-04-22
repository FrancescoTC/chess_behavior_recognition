import torch
import numpy as np
import os
from typing import Optional, Tuple
import json

from transformers import PreTrainedTokenizer
from globals import Globals


class ChessTokenizer(PreTrainedTokenizer):
    def __init__(self, **kwargs):

        # Initialize vocabulary
        self.vocab: dict = Globals.meta["stoi"]
        self.ids_to_tokens: dict = Globals.meta["itos"]

        super().__init__(**kwargs)

        # Set up special tokens
        self._unk_token = " "
        self._pad_token = " "
        self.bos_token = ";"
        self.eos_token = ";"

        # Automatically set special tokens ids
        self._unk_token_id = self.vocab.get(self._unk_token)
        self._pad_token_id = self.vocab.get(self._pad_token)
        self.bos_token_id = self.vocab.get(self.bos_token)
        self.eos_token_id = self.vocab.get(self.eos_token)

    @property
    def vocab_size(self):
        return len(self.vocab)

    def get_vocab(self):
        return dict(self.vocab)

    def _tokenize(self, text):
        """ Tokenize a string: split in characters. """
        return list(text)

    def _convert_token_to_id(self, token):
        """ Converts a token (str) in an id using the vocab. """
        return self.vocab.get(token)

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.ids_to_tokens.get(index)

    def encode(self, text: str) -> np.ndarray:
        """Encodes a text into a numpy array of ids."""
        return np.array([self._convert_token_to_id(token) for token in self._tokenize(text)], dtype=np.int64)

    def decode(self, ids: np.ndarray) -> str:
        """Decodes a numpy array of ids into a text."""
        return "".join([self._convert_id_to_token(idx) for idx in ids])

    def _convert_ids_to_tokens(self, ids):
        return self.decode(ids)

    def __call__(
        self,
        text,
        return_tensors=None,
        **kwargs,
    ):
        if isinstance(text, str):
            # Single string input
            input_ids = self.encode(text)
        else:
            # List of strings input
            input_ids = np.array([self.encode(t) for t in text], dtype=np.int64)

        # Output dictionary
        batch_encoding = {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
        }

        if return_tensors == "pt":
            batch_encoding = {k: torch.as_tensor(v) for k, v in batch_encoding.items()}

        return batch_encoding

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        if not os.path.isdir(save_directory):
            os.makedirs(save_directory)

        # Set file names
        vocab_filename = (filename_prefix + "-" if filename_prefix else "") + "vocab.json"
        vocab_path = os.path.join(save_directory, vocab_filename)

        # Save vocab
        with open(vocab_path, 'w', encoding='utf-8') as vocab_file:
            json.dump(self.vocab, vocab_file, ensure_ascii=False, indent=2)

        return (vocab_path,)



# Test the ChessTokenizer
if __name__ == "__main__":
    tokenizer = ChessTokenizer(clean_up_tokenization_spaces=False)
    moves = "1. e4 e5"

    print("original moves:", moves)
    encoded = tokenizer(moves, padding=True, truncation=True, max_length=2048)
    print(f"Encoded: {encoded}")
    decoded = tokenizer.decode(encoded["input_ids"])
    print(f"Decoded: {decoded}")


