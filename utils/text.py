import re

import nltk
import numpy as np
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


def get_context_average_embedding(
    sentence_tokens: list[str], oov_token: str, w2v_model: any
) -> np.ndarray:
    """Generates an approximate embedding for an out-of-vocabulary (OOV) word by averaging the
    embeddings of surrounding context words within the sentence.

    Args:
        sentence_tokens (list[str]): A list of tokens from a sentence, each as a lowercase word.
        oov_token (str): The OOV word for which an approximate embedding is needed.
        w2v_model (gensim.models.KeyedVectors): A pretrained Word2Vec model where word embeddings
            can be accessed by `w2v_model[word]`.

    Returns:
        np.ndarray: The averaged embedding vector for the OOV word based on its context.
        Returns a zero vector with the same dimension as the Word2Vec embeddings if no
        context words are found.

    Example:
        >>> sentence_tokens = ["this", "is", "an", "example", "with", "oovword"]
        >>> oov_token = "oovword"
        >>> get_context_average_embedding(sentence_tokens, oov_token, w2v_model)
        array([...])  # The average embedding of context words

    Notes:
        This function assumes that `sentence_tokens` is already tokenized and in lowercase.
    """

    context_embeddings = [
        w2v_model[word]
        for word in sentence_tokens
        if word != oov_token and word in w2v_model.key_to_index
    ]
    if context_embeddings:
        return np.mean(context_embeddings, axis=0)  # Average embedding of context
    return np.zeros(w2v_model.vector_size)  # Return zero vector if no context
