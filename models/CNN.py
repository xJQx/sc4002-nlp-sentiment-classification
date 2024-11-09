from dataclasses import dataclass, field

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import MulticlassAccuracy


@dataclass
class CNNArgs:
    embedding_matrix: np.ndarray
    freeze_embedding: bool = True
    hidden_dim: int = 128
    n_grams: list = field(default_factory=lambda: [3, 4, 5])
    dropout: float = 0.3
    output_dim: int = 2
    padding_idx: int = 0

    def __post_init__(self):
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be a positive integer")


class CNN(nn.Module):

    def __init__(self, args: CNNArgs):
        super().__init__()

        # Embedding layer
        _, embedding_dim = args.embedding_matrix.shape
        self.embedding = nn.Embedding.from_pretrained(
            embeddings=torch.FloatTensor(args.embedding_matrix),
            padding_idx=args.padding_idx,
            freeze=args.freeze_embedding,
        )

        hidden_dim = args.hidden_dim - args.hidden_dim % len(args.n_grams)
        num_filters = int(hidden_dim / len(args.n_grams))

        self.conv_list = []
        for n_gram in args.n_grams:
            self.conv_list.append(nn.Conv2d(1, num_filters, (n_gram, embedding_dim)))

        self.dropout = nn.Dropout(args.dropout)

        # fc layer
        self.fc = nn.Linear(hidden_dim, hidden_dim // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim // 2, args.output_dim)

    def forward(self, sequences):
        embeddings = self.embedding(sequences)  # [batch_size, seq_len, embed_dim]

        # add channel [batch_size, 1, seq_len, embed_dim]
        embeddings = embeddings.unsqueeze(1)

        # [batch_size, out_channels, seq_len - kernel_size + 1]
        pool_list = []
        for conv in self.conv_list:
            conv_out = torch.relu(conv(embeddings)).squeeze(3)
            pool = nn.functional.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
            pool_list.append(pool)

        sentence_representation = torch.cat(pool_list, dim=1)
        sentence_representation = self.dropout(sentence_representation)

        logits = self.fc2(self.relu(self.fc(sentence_representation)))
        return logits


class CNNClassifier(L.LightningModule):
    """CNN Classifier for binary classification tasks using PyTorch Lightning.

    Args:
        cnn_model (torch.nn.Module): The CNN model used for generating logits from inputs.
        optimizer_name (str): Name of the optimizer to use. Options include 'SGD', 'Adagrad', 'Adam', and 'RMSprop'.
        lr (float): Learning rate for the optimizer.
        show_progress (bool, optional): If True, logs additional progress information to the progress bar. Default is False.
    """

    def __init__(
        self,
        cnn_model: CNN,
        optimizer_name: str,
        learning_rate: float,
        show_progress: bool = False,
    ):
        super().__init__()
        self.model = cnn_model
        self.optimizer_name = optimizer_name
        self.lr = learning_rate
        self.show_progress = show_progress
        self.metric = MulticlassAccuracy(num_classes=2)
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        indexes = batch["indexes"]
        labels = batch["label"]

        logits = self.model(indexes)

        loss = F.cross_entropy(logits, labels)
        acc = self.metric(logits, labels)

        self.log("train_loss", loss, prog_bar=self.show_progress)
        self.log("train_acc", acc, prog_bar=self.show_progress)

        return loss

    def validation_step(self, batch, batch_idx):
        indexes = batch["indexes"]
        labels = batch["label"]

        logits = self.model(indexes)

        loss = F.cross_entropy(logits, labels)
        acc = self.metric(logits, labels)

        self.log("val_loss", loss, prog_bar=self.show_progress)
        self.log("val_acc", acc, prog_bar=self.show_progress)

    def test_step(self, batch, batch_idx):
        indexes = batch["indexes"]
        labels = batch["label"]

        logits = self.model(indexes)

        loss = F.cross_entropy(logits, labels)
        acc = self.metric(logits, labels)

        self.log("test_loss", loss, prog_bar=self.show_progress)
        self.log("test_acc", acc, prog_bar=self.show_progress)

    def predict_step(self, batch, batch_idx):
        indexes = batch["indexes"]

        logits = self.model(indexes)
        predicted_labels = torch.argmax(logits, dim=1)
        return predicted_labels

    def configure_optimizers(self):
        if self.optimizer_name == "SGD":
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        elif self.optimizer_name == "Adagrad":
            optimizer = torch.optim.Adagrad(self.model.parameters(), lr=self.lr)
        elif self.optimizer_name == "Adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        elif self.optimizer_name == "RMSprop":
            optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.lr)
        else:
            raise Exception("Invalid optimizer name!")

        return optimizer
