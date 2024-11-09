import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[1]))

import numpy as np

embedding_path = Path("models/embedding_matrix.npy")
index_from_word_path = Path("models/index_from_word.json")

embedding_matrix = np.load(embedding_path)
with index_from_word_path.open() as f:
    index_from_word = json.load(f)


from utils.text import tokenize
from datasets import load_dataset

dataset = load_dataset("rotten_tomatoes")
train_dataset = tokenize(dataset["train"])
val_dataset = tokenize(dataset["validation"])
test_dataset = tokenize(dataset["test"])


from utils.text import token_to_index

train_dataset = token_to_index(dataset=train_dataset, index_from_word=index_from_word)
val_dataset = token_to_index(dataset=val_dataset, index_from_word=index_from_word)
test_dataset = token_to_index(dataset=test_dataset, index_from_word=index_from_word)

train_dataset = train_dataset.select_columns(["label", "original_len", "indexes"])
val_dataset = val_dataset.select_columns(["label", "original_len", "indexes"])
test_dataset = test_dataset.select_columns(["label", "original_len", "indexes"])
train_dataset.set_format(type="torch")
val_dataset.set_format(type="torch")
test_dataset.set_format(type="torch")

import optuna
from utils.train import train_rnn_model_with_parameters

SEARCH_SPACE = {
    "batch_size": [32, 64, 128, 256, 512, 1024, 2048],
    "learning_rate": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
    "optimizer_name": ["SGD", "Adagrad", "RMSprop", "Adam"],
    # RNN Model Parameters
    "hidden_dim": [256, 128, 64, 32],
    "num_layers": [1, 2, 4],
    "sentence_representation_type": ["last", "average", "max"],
}


def objective(trial):
    hidden_dim = trial.suggest_categorical("hidden_dim", SEARCH_SPACE["hidden_dim"])
    num_layers = trial.suggest_int(
        "num_layers", min(SEARCH_SPACE["num_layers"]), max(SEARCH_SPACE["num_layers"])
    )
    optimizer_name = trial.suggest_categorical(
        "optimizer_name", SEARCH_SPACE["optimizer_name"]
    )
    batch_size = trial.suggest_categorical("batch_size", SEARCH_SPACE["batch_size"])
    learning_rate = trial.suggest_categorical(
        "learning_rate", SEARCH_SPACE["learning_rate"]
    )
    sentence_representation_type = trial.suggest_categorical(
        "sentence_representation_type", SEARCH_SPACE["sentence_representation_type"]
    )

    log_message = f"---------- batch_size_{batch_size}; lr_{learning_rate}; optimizer_{optimizer_name}; hidden_dim_{hidden_dim}; num_layers_{num_layers}; sentence_representation_{sentence_representation_type} ----------"
    print(log_message)

    val_acc = train_rnn_model_with_parameters(
        embedding_matrix=embedding_matrix,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=batch_size,
        learning_rate=learning_rate,
        optimizer_name=optimizer_name,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        sentence_representation_type=sentence_representation_type,
        show_progress=True,
        freeze_embedding=False,
        log_dir="rnn_trainable_embeddings",
    )

    return val_acc


# Set up the Optuna study
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=500)

# Get the best hyperparameters
best_params = study.best_params

print("Best hyperparameters:", best_params)

from utils.analytics import load_tensorboard_logs

train_results_df = load_tensorboard_logs(log_dir="tb_logs/rnn_trainable_embeddings")

train_results_df = train_results_df.sort_values(
    by=["val_acc"], ascending=False
).reset_index(drop=True)
print(train_results_df.head())
