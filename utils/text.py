import re
import nltk
import pandas as pd

nltk.download('punkt')
nltk.download('treebank')
nltk.download('punkt_tab')

def preprocessText(dataframe: pd.DataFrame):
  texts_tokens = []

  for i in range(0, len(dataframe)):
    tokens = tokenize_sentence(dataframe['text'][i])
    
    texts_tokens.append(tokens)
      
  return texts_tokens

def tokenize_sentence(text: str) -> list[str]:
  text = re.sub('[^a-zA-Z]', ' ', text) # remove numbers and non-alphabetical symbols
  text = text.lower() # lower case
  text = text.strip()

  if isinstance(text, str):    
    tokens = nltk.tokenize.word_tokenize(text) 
  else:
    raise Exception("Input is not a valid string.")

  return tokens