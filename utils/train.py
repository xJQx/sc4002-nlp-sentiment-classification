from pathlib import Path

import lightning as L
import numpy as np
from datasets import Dataset
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from models.RNN import RNN, RNNClassifier


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
):
    min_epochs = 20
    max_epochs = 10_000

    L.seed_everything(seed)

    _rnn_model = RNN(
        embedding_matrix=embedding_matrix,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        output_dim=2,
        sentence_representation_type=sentence_representation_type,
    )

    model = RNNClassifier(
        rnn_model=_rnn_model,
        optimizer_name=optimizer_name,
        lr=learning_rate,
        show_progress=show_progress,
    )

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Train model.
    log_file_name = f"{log_dir}/batch_size_{batch_size}-lr_{learning_rate}-optimizer_{optimizer_name}-hidden_dim_{hidden_dim}-num_layers_{num_layers}-sentence_representation_type_{sentence_representation_type}"

    # Skip if run before
    if list(Path().rglob(log_file_name)):
        print(f"[Skipping] {log_file_name}")
        return
    logger = TensorBoardLogger("tb_logs", name=log_file_name)

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=early_stopping_patience,
        )
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
