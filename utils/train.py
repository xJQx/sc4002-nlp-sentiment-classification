import os
from dataclasses import dataclass
from pathlib import Path

import lightning as L
import numpy as np
from datasets import Dataset
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from models.CNN import CNN, CNNArgs, CNNClassifier
from models.RNN import RNN, RNNClassifier
from utils.analytics import get_result_from_file


def train_rnn_model_with_parameters(
    embedding_matrix: np.ndarray,
    train_dataset: Dataset,
    val_dataset: Dataset,
    batch_size: int,
    learning_rate: float,
    optimizer_name: str,
    hidden_dim: int,
    num_layers: int,
    sentence_representation_type: str = "last",
    show_progress: bool = True,
    seed: int = 42,
    log_dir: str = "rnn/test",
    early_stopping_patience: int = 3,
    freeze_embedding: bool = True,
    rnn_type: str = "RNN",
    bidirectional: bool = False,
):

    min_epochs = 0
    max_epochs = 10_000
    num_workers = os.cpu_count() // 2

    L.seed_everything(seed)

    _rnn_model = RNN(
        embedding_matrix=embedding_matrix,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        output_dim=2,
        sentence_representation_type=sentence_representation_type,
        freeze_embedding=freeze_embedding,
        rnn_type=rnn_type,
        bidirectional=bidirectional,
    )

    model = RNNClassifier(
        rnn_model=_rnn_model,
        optimizer_name=optimizer_name,
        lr=learning_rate,
        show_progress=show_progress,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    # Train model.
    log_file_name = f"{log_dir}/batch_size_{batch_size}-lr_{learning_rate}-optimizer_{optimizer_name}-hidden_dim_{hidden_dim}-num_layers_{num_layers}-sr_type_{sentence_representation_type}-freeze_{freeze_embedding}-rnn_type_{rnn_type}-bidirectional_{bidirectional}"

    # Skip if run before
    if list(Path().rglob(log_file_name)):
        print(f"[Skipping] {log_file_name}")
        result = get_result_from_file(f"tb_logs/{log_file_name}")

        return result["val_loss"]  # for optuna

    logger = TensorBoardLogger("tb_logs", name=log_file_name)

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=early_stopping_patience,
            min_delta=1e-4,
        ),
        EarlyStopping(
            monitor="val_acc",
            mode="max",
            patience=early_stopping_patience * 5,
            min_delta=1e-4,
        ),
        ModelCheckpoint(
            monitor="val_loss",
            save_top_k=1,
            mode="min",
        ),
    ]
    trainer = L.Trainer(
        default_root_dir="models/",
        callbacks=callbacks,
        min_epochs=min_epochs,
        max_epochs=max_epochs,
        logger=logger,
        accelerator="cpu",
        log_every_n_steps=5,
    )

    trainer.fit(
        model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
    )

    result = get_result_from_file(f"tb_logs/{log_file_name}")

    return result["val_loss"]  # for optuna


@dataclass
class OptimizerArgs:
    optimizer_name: str = "Adam"
    learning_rate: float = 1e-3

    def __post_init__(self):
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be a positive float")


@dataclass
class DataArgs:
    batch_size: int
    train_dataset: Dataset
    val_dataset: Dataset
    shuffle_train: bool = True
    shuffle_val: bool = False


# TODO: refactor with above function
def train_cnn_model_with_parameters(
    data_args: DataArgs,
    cnn_args: CNNArgs,
    optimizer_args: OptimizerArgs,
    seed: int = 42,
    log_dir: str = "cnn/",
    early_stopping_patience: int = 3,
    show_progress: bool = True,
):
    min_epochs = 0
    max_epochs = 10_000
    num_workers = os.cpu_count() // 2

    L.seed_everything(seed)

    _cnn_model = CNN(cnn_args)
    model = CNNClassifier(
        cnn_model=_cnn_model,
        show_progress=show_progress,
        **optimizer_args.__dict__,
    )

    train_dataloader = DataLoader(
        data_args.train_dataset,
        batch_size=data_args.batch_size,
        shuffle=data_args.shuffle_train,
        num_workers=num_workers,
    )
    val_dataloader = DataLoader(
        data_args.val_dataset,
        batch_size=data_args.batch_size,
        shuffle=data_args.shuffle_val,
        num_workers=num_workers,
    )

    # Train model.
    log_file_name = f"{log_dir}/batch_size_{data_args.batch_size}-lr_{optimizer_args.learning_rate}-optimizer_{optimizer_args.optimizer_name}-hidden_dim_{cnn_args.hidden_dim}-n_grams_{'_'.join(map(str, cnn_args.n_grams))}-dropout_{cnn_args.dropout}"

    # Skip if run before
    if list(Path().rglob(log_file_name)):
        print(f"[Skipping] {log_file_name}")
        result = get_result_from_file(f"tb_logs/{log_file_name}")
        return result["val_loss"]  # for optuna
    logger = TensorBoardLogger("tb_logs", name=log_file_name)

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=early_stopping_patience,
            min_delta=1e-4,
        ),
        EarlyStopping(
            monitor="val_acc",
            mode="max",
            patience=early_stopping_patience * 5,
            min_delta=1e-4,
        ),
        ModelCheckpoint(
            monitor="val_loss",
            save_top_k=1,
            mode="min",
        ),
    ]
    trainer = L.Trainer(
        default_root_dir="models/",
        callbacks=callbacks,
        min_epochs=min_epochs,
        max_epochs=max_epochs,
        logger=logger,
        accelerator="cpu",
    )

    trainer.fit(
        model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
    )

    result = get_result_from_file(f"tb_logs/{log_file_name}")

    return result["val_loss"]  # for optuna
