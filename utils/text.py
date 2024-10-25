import re

import nltk
import pandas as pd

from datasets import Dataset

nltk.download("punkt")
nltk.download("treebank")
nltk.download("punkt_tab")


def tokenize(dataset: Dataset, text_column: str = "text") -> pd.Series:
    def _tokenize(examples):
        text = examples[text_column].lower()
        # Remove numbers, non-alphabetical symbols and trailing white spaces.
        cleaned_text = re.sub("[^a-z]", " ", text).strip()
        tokens = nltk.tokenize.word_tokenize(cleaned_text)
        return {"tokens": tokens}

    filter_na = lambda example: example[text_column] is not None

    return dataset.filter(filter_na).map(_tokenize)
