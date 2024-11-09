import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[1]))

import numpy as np
import optuna
from datasets import load_dataset

from utils.text import token_to_index, tokenize
from utils.train import (
    CNNArgs,
    DataArgs,
    OptimizerArgs,
    train_cnn_model_with_parameters,
)

# Load the embedding matrix that handled OOV words
embedding_path = Path("models/embedding_matrix_oov.npy")
index_from_word_path = Path("models/index_from_word_oov.json")

embedding_matrix = np.load(embedding_path)
with index_from_word_path.open() as f:
    index_from_word = json.load(f)


dataset = load_dataset("rotten_tomatoes")
train_dataset = tokenize(dataset["train"])
val_dataset = tokenize(dataset["validation"])
test_dataset = tokenize(dataset["test"])


train_dataset = token_to_index(dataset=train_dataset, index_from_word=index_from_word)
val_dataset = token_to_index(dataset=val_dataset, index_from_word=index_from_word)
test_dataset = token_to_index(dataset=test_dataset, index_from_word=index_from_word)

train_dataset = train_dataset.select_columns(["label", "original_len", "indexes"])
val_dataset = val_dataset.select_columns(["label", "original_len", "indexes"])
test_dataset = test_dataset.select_columns(["label", "original_len", "indexes"])

train_dataset.set_format(type="torch")
val_dataset.set_format(type="torch")
test_dataset.set_format(type="torch")

_N_TRIALS = 500
SEARCH_SPACE = {
    "batch_size": [512, 1024, 2048],
    "learning_rate": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
    "optimizer_name": ["Adam"],
    # CNN Model Parameters
    "dropout": [0.1, 0.3, 0.5, 0.7, 0.9],
    "hidden_dim": [600, 500, 400, 300],
    "n_grams": [
        [2],
        [3],
        [4],
        [5],
        [6],
        [2, 3],
        [2, 3, 4],
        [2, 3, 4, 5],
        [2, 3, 4, 6],
        [3, 4],
        [3, 4, 5],
        [3, 4, 6],
        [4, 5],
        [4, 5, 6],
        [5, 6],
    ],
}


def objective(trial: optuna.Trial):
    batch_size = trial.suggest_categorical("batch_size", SEARCH_SPACE["batch_size"])
    learning_rate = trial.suggest_categorical(
        "learning_rate", SEARCH_SPACE["learning_rate"]
    )
    optimizer_name = trial.suggest_categorical(
        "optimizer_name", SEARCH_SPACE["optimizer_name"]
    )
    # CNN Model Parameters
    dropout = trial.suggest_categorical("dropout", SEARCH_SPACE["dropout"])
    hidden_dim = trial.suggest_categorical("hidden_dim", SEARCH_SPACE["hidden_dim"])
    n_grams = trial.suggest_categorical("n_grams", SEARCH_SPACE["n_grams"])

    log_message = f"---------- batch_size_{batch_size}; lr_{learning_rate}; optimizer_{optimizer_name}; hidden_dim_{hidden_dim}; n_grams_{"_".join(map(str, n_grams))}; dropout_{dropout}  ----------"
    print(log_message)

    cnn_args = CNNArgs(
        embedding_matrix=embedding_matrix,
        freeze_embedding=False,
        hidden_dim=hidden_dim,
        dropout=dropout,
        n_grams=n_grams,
    )

    optimizer_args = OptimizerArgs(
        optimizer_name=optimizer_name,
        learning_rate=learning_rate,
    )

    data_args = DataArgs(
        batch_size=batch_size,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )

    val_acc = train_cnn_model_with_parameters(
        data_args=data_args,
        cnn_args=cnn_args,
        optimizer_args=optimizer_args,
    )

    return val_acc


# Set up the Optuna study
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=_N_TRIALS)


print("Finish searching!")
print(study.best_params)
