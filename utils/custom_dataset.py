from torch.utils.data import Dataset
import torch
import pandas as pd
from utils.text import tokenize_sentence
from utils.file import load_from_local_file

# Custom Text Dataset
class TextDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, max_len: int, embedding_matrix_vocab_to_index: dict):
        self.data = dataframe
        self.max_len = max_len
        self.embedding_matrix_vocab_to_index = embedding_matrix_vocab_to_index

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get the sentence and label
        sentence = self.data.iloc[idx, 0]
        label = self.data.iloc[idx, 1]

        # Tokenize sentence
        sentence_tokens: list[str] = tokenize_sentence(sentence)

        # Convert Tokens into indexes used in embeddings layer
        sentence_tokens_indexes = []
        for token in sentence_tokens:
            if token in self.embedding_matrix_vocab_to_index.keys():
                sentence_tokens_indexes.append(self.embedding_matrix_vocab_to_index[token])
            else:
                # For OOV words in val and test set
                sentence_tokens_indexes.append(self.embedding_matrix_vocab_to_index[""])

        # Pad the sentence if it's shorter than max_len, or truncate if it's longer
        if len(sentence_tokens_indexes) < self.max_len:
            sentence_tokens_indexes = sentence_tokens_indexes + [0] * (self.max_len - len(sentence_tokens_indexes)) # Padding with 0
        elif len(sentence_tokens_indexes) > self.max_len:
            sentence_tokens_indexes = sentence_tokens_indexes[:self.max_len] # Truncate to max_len

        # Convert to PyTorch tensors
        sentence = torch.tensor(sentence_tokens_indexes, dtype=torch.long)
        label = torch.tensor(label, dtype=torch.long)

        return sentence, label
