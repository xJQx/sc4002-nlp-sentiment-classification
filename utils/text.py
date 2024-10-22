import re
import nltk
import pandas as pd

nltk.download("punkt")
nltk.download("treebank")
nltk.download("punkt_tab")


def preprocess_text(dataframe: pd.DataFrame, text_column: str = "text") -> pd.Series:
    if text_column not in dataframe.columns:
        raise KeyError(f"'{text_column}' column not found in the dataframe.")

    valid_texts = dataframe[text_column].dropna().astype(str)

    return valid_texts.apply(tokenize_sentence).tolist()


def tokenize_sentence(text: str) -> list[str]:
    text = re.sub("[^a-zA-Z]", " ", text)  # remove numbers and non-alphabetical symbols
    text = text.lower()  # lower case
    text = text.strip()

    if isinstance(text, str):
        tokens = nltk.tokenize.word_tokenize(text)
    else:
        raise Exception("Input is not a valid string.")

    return tokens
