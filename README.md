# SC4002 Natural Language Processing Group Assignment - Sentiment Classification

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Introduction](#introduction)
- [Setup Instructions](#setup-instructions)
- [Project Structure](#project-structure)
  - [Jupyter Notebooks](#jupyter-notebooks)
  - [Utilities](#utilities)
  - [Models](#models)
  - [Additional Scripts](#additional-scripts)
- [Code Used for Each Part](#code-used-for-each-part)
- [Directory Layout](#directory-layout)

## Introduction

This project is a group assignment for the SC4002 Natural Language Processing course, focusing on sentiment classification using various machine learning models. The implementation includes RNNs, LSTMs, GRUs, CNNs, and Transformers, along with techniques for handling out-of-vocabulary (OOV) words. The dataset used is the rotten tomato dataset.

## Setup Instructions

1. **Create a Python Virtual Environment and Activate It**

   Navigate to the root directory of the project and execute the following commands:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Linux/macOS
   ```

   *Note for Windows users:*

   ```bash
   .\.venv\Scripts\activate
   ```

2. **Install Required Packages**

   Install all necessary packages using the `requirements.txt` file:

   ```bash
   pip install -r requirements.txt
   ```

3. **Run Jupyter Notebook**

   We utilized Jupyter Notebooks to explain and display our outputs interactively. To learn how to run a Jupyter Notebook, refer to the [official documentation](https://docs.jupyter.org/en/latest/running.html).

## Project Structure

### Jupyter Notebooks

- `part0.ipynb`: Downloading the dataset.
- `part1.ipynb`: Answers Part 1 questions (preparing word embeddings and mitigating OOV).
- `part2.ipynb`: Answers Part 2 questions (RNN model).
- `part3a.ipynb`: Answers Part 3a questions (RNN model with trainable embeddings).
- `part3b_generate_embedding.ipynb`: Generates a new embedding matrix with mitigated OOV.
- `part3b.ipynb`: Answers Part 3b questions (RNN model with trainable embeddings and mitigated OOV).
- `part3c_biLSTM.ipynb`: Answers Part 3c question on biLSTM.
- `part3c_biGRU.ipynb`: Answers Part 3c question on biGRU.
- `part3d.ipynb`: Answers Part 3d questions (CNN model).
- `part3e.ipynb`: Answers Part 3e questions (Transformer models).
- `part3f.ipynb`: Answers Part 3f questions (model comparison).

### Utilities

- `utils/analytics.py`: Contains code for loading TensorBoard log files and uploading data to Weights & Biases (WandB).
- `utils/text.py`: Provides functions for tokenizing and preprocessing text data, as well as computing average context embeddings.
- `utils/train.py`: Includes training routines for the models, handling logic such as early stopping and logging.

### Models

- `models/RNN.py`: Implementation of the RNN model with PyTorch Lightning wrappers for training, validation, and testing.
- `models/CNN.py`: Implementation of the CNN model with PyTorch Lightning wrappers.
- `models/biLSTM.py`: Implementation of the bidirectional LSTM model.
- `models/biGRU.py`: Implementation of the bidirectional GRU model.
- `models/MetaModel.py`: Implementation of the ensemble meta model.
- `models/embedding_matrix.npy`: Embedding matrix based on GoogleNews300 Word2Vec.
- `models/index_from_word.json`: A mapping from words to their corresponding indices in the embedding matrix.
- `models/embedding_matrix_oov.npy`: Embedding matrix based on GoogleNews300 Word2Vec with OOV words filled with contextual average.
- `models/index_from_word_oov.json`: A mapping from words to their corresponding indices in the embedding matrix with OOV words filled with contextual average.

### Additional Scripts

- `part3e_transformers.py`: A pipeline script to train, evaluate, and test Transformer models for Part 3e.
  - **Example Usage**: `python part3e_transformers.py --model roberta`
- `part3e_ensemble.py`: A script to train and evaluate ensemble models for Part 3e.
- `scripts/xxx.py`: A script to train certain models using the terminal instead of jupyter notebook (optional, you may run the jupyter notebook directly to train the model instead).
  - **Example Usage**: `python scripts/train_cnn.py`

## Code Used for Each Part

| **Part** | **Files and Scripts Used** |
|----------|----------------------------|
| **Part 1** | - `utils/text.py`<br>- `part1.ipynb` |
| **Part 2** | - `utils/text.py`<br>- `utils/train.py`<br>- `utils/analytics.py`<br>- `models/RNN.py`<br>- `part2.ipynb` |
| **Part 3a** | - `utils/text.py`<br>- `utils/train.py`<br>- `utils/analytics.py`<br>- `models/RNN.py`<br>- `part3a.ipynb` |
| **Part 3b** | - `utils/text.py`<br>- `utils/train.py`<br>- `utils/analytics.py`<br>- `models/RNN.py`<br>- `part3b.ipynb` |
| **Part 3c** | - `utils/text.py`<br>- `utils/train.py`<br>- `utils/analytics.py`<br>- `models/RNN.py`<br>- `part3c_biLSTM.ipynb`<br>- `part3c_biGRU.ipynb` |
| **Part 3d** | - `utils/text.py`<br>- `utils/train.py`<br>- `utils/analytics.py`<br>- `models/CNN.py`<br>- `part3d.ipynb` |
| **Part 3e** | - `utils/text.py`<br>- `utils/train.py`<br>- `utils/analytics.py`<br>- `part3e_transformers.py`<br>- `part3e_ensemble.py`<br>- `models/MetaModel.py`<br>- `part3e.ipynb` |
| **Part 3f** | - `part3f.ipynb` |

## Directory Layout

```
root_dir/
├── .venv/                          
├── best_model_predictions/         
├── models/                         
│   ├── CNN.py                      
│   ├── RNN.py                      
│   ├── MetaModel.py                
│   ├── embedding_matrix.npy        
│   ├── embedding_matrix_oov.npy    
│   ├── index_from_word.json        
│   ├── index_from_word_oov.json    
│   ├── word2vec-google-news-300    
│   └── word2vec-google-news-300.vectors.npy
├── scripts/                        
│   ├── train_bigru.py              
│   ├── train_cnn.py                
│   ├── train_lstm.py               
│   ├── train_rnn_part_2.py         
│   ├── train_rnn_part_3a.py        
│   └── ...        
├── tb_logs/                        
│   ├── rnn/                        
│   ├── rnn_trainable_embeddings/                        
│   ├── rnn_trainable_embeddings_and_contextual_oov/                        
│   ├── cnn/                        
│   ├── bigru/                        
│   ├── bilstm/                        
│   ├── transformers/                        
│   └── ...                        
├── utils/                          
│   ├── analytics.py                
│   ├── text.py                     
│   └── train.py                    
├── .gitignore                      
├── requirements.txt                
├── README.md                       
├── part0.ipynb                 
├── part1.ipynb                 
├── part2.ipynb                 
├── part3a.ipynb                
├── part3b_generate_embedding.ipynb
├── part3b.ipynb                
├── part3c_biGRU.ipynb          
├── part3c_biLSTM.ipynb         
├── part3d.ipynb                
├── part3e.ipynb                
├── part3e_ensemble.py          
├── part3e_transformers.py      
└── part3f.ipynb                
```
