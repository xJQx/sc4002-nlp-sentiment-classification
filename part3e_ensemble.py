import json
from pathlib import Path

import lightning as L
import optuna
import torch
from datasets import load_dataset
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, TensorDataset

from models.MetaModel import MetaModel
from utils.analytics import get_result_from_file


def load_predictions(file_list):
    """Loads and combines predictions from a list of JSON files."""
    predictions = []
    for file_path in file_list:
        with file_path.open() as f:
            data = json.load(f)
            predictions.append(torch.tensor(data))

    # Stack along a new dimension to have shape (num_samples, num_models * num_classes)
    return torch.cat(predictions, dim=1)


_N_TRIALS = 1000
SEARCH_SPACE = {
    "batch_size": [8, 16, 32, 64, 128, 256, 512, 1024, 2048],
    "learning_rate": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
    "optimizer_name": ["Adagrad", "RMSprop", "Adam", "SGD"],
    # Model Parameters
    "dropout": [0, 0.1, 0.3, 0.5, 0.7, 0.9],
    "hidden_dim": [600, 500, 400, 300],
}


def train(
    train_data: TensorDataset,
    val_data: TensorDataset,
    test_data: TensorDataset,
    meta_input_dim: int,
    batch_size: int,
    learning_rate: float,
    optimizer_name: str,
    dropout: float,
    hidden_dim: int,
    log_dir: str,
):
    min_epochs = 0
    max_epochs = 10_000

    L.seed_everything(42)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    # test_loader = DataLoader(test_data, batch_size=batch_size)

    log_file_name = f"{log_dir}/batch_size_{batch_size}-lr_{learning_rate}-optimizer_{optimizer_name}-hidden_dim_{hidden_dim}-dropout-{dropout}"

    # Skip if run before
    if list(Path().rglob(log_file_name)):
        print(f"[Skipping] {log_file_name}")
        result = get_result_from_file(f"tb_logs/{log_file_name}")

        return result["val_acc"]  # for optuna

    # Initialize the meta-model
    meta_model = MetaModel(
        input_dim=meta_input_dim,
        hidden_dim=hidden_dim,
        optimizer_name=optimizer_name,
        learning_rate=learning_rate,
        dropout=dropout,
        num_classes=2,
    )

    # Train
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=3,
            min_delta=1e-4,
        ),
        EarlyStopping(
            monitor="val_acc",
            mode="max",
            patience=3 * 5,
            min_delta=1e-4,
        ),
        ModelCheckpoint(
            monitor="val_loss",
            save_top_k=1,
            mode="min",
        ),
    ]

    logger = TensorBoardLogger("tb_logs", name=log_file_name)

    trainer = L.Trainer(
        callbacks=callbacks,
        min_epochs=min_epochs,
        max_epochs=max_epochs,
        logger=logger,
        accelerator="cpu",
        log_every_n_steps=1,
    )
    trainer.fit(meta_model, train_loader, val_loader)
    # trainer.test(meta_model, test_loader)

    result = get_result_from_file(f"tb_logs/{log_file_name}")

    return result["val_acc"]  # for optuna


def main():
    pred_path = Path("best_model_predictions")

    train_files = list(pred_path.rglob("train*"))
    val_files = list(pred_path.rglob("val*"))
    test_files = list(pred_path.rglob("test*"))

    # Load and combine predictions
    train_predictions = load_predictions(train_files)
    val_predictions = load_predictions(val_files)
    test_predictions = load_predictions(test_files)

    dataset = load_dataset("rotten_tomatoes")
    dataset.set_format(type="torch")
    train_labels = dataset["train"]["label"]
    val_labels = dataset["validation"]["label"]
    test_labels = dataset["test"]["label"]

    train_data = TensorDataset(train_predictions, train_labels)
    val_data = TensorDataset(val_predictions, val_labels)
    test_data = TensorDataset(test_predictions, test_labels)

    def objective(trial):
        batch_size = trial.suggest_categorical("batch_size", SEARCH_SPACE["batch_size"])
        learning_rate = trial.suggest_categorical(
            "learning_rate", SEARCH_SPACE["learning_rate"]
        )
        optimizer_name = trial.suggest_categorical(
            "optimizer_name", SEARCH_SPACE["optimizer_name"]
        )
        dropout = trial.suggest_categorical("dropout", SEARCH_SPACE["dropout"])
        hidden_dim = trial.suggest_categorical("hidden_dim", SEARCH_SPACE["hidden_dim"])

        log_message = f"---------- batch_size_{batch_size}; lr_{learning_rate}; optimizer_{optimizer_name}; dropout_{dropout}; hidden_dim_{hidden_dim}; ----------"
        print(log_message)

        val_acc = train(
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            meta_input_dim=train_predictions.shape[1],
            batch_size=batch_size,
            learning_rate=learning_rate,
            optimizer_name=optimizer_name,
            dropout=dropout,
            hidden_dim=hidden_dim,
            log_dir="ensemble_small",
        )

        return val_acc

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=_N_TRIALS)

    print("Finish searching!")
    print(study.best_params)


if __name__ == "__main__":
    main()
