import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import MulticlassAccuracy

from utils.text import replace_oov_with_mean


class CNN(nn.Module):

    def __init__(
        self,
        embedding_matrix: np.ndarray,
        output_dim: int,
        freeze_embedding: bool = False,
        handle_oov: bool = True,
        padding_idx: int = 0,
        hidden_dim: int = 300,
        n_grams: list[int] = [3, 4, 5],
        dropout: int = 0.3,
    ):
        super().__init__()

        self.handle_oov = handle_oov

        # Embedding layer
        _, embedding_dim = embedding_matrix.shape
        self.embedding = nn.Embedding.from_pretrained(
            embeddings=torch.FloatTensor(embedding_matrix),
            padding_idx=padding_idx,
            freeze=freeze_embedding,
        )

        hidden_dim = hidden_dim - hidden_dim % len(n_grams)
        num_filters = hidden_dim / len(n_grams)

        self.conv_list = []
        for n_gram in n_grams:
            self.conv_list.append(nn.Conv2d(1, num_filters, (n_gram, embedding_dim)))

        self.dropout = nn.Dropout(dropout)

        # fc layer
        self.fc = nn.Linear(hidden_dim, hidden_dim // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)

    def forward(self, sequences):
        embeddings = self.embedding(sequences)  # [batch_size, seq_len, embed_dim]

        if self.handle_oov:
            embeddings_list = []
            for i in range(sequences.size(0)):
                embeddings_list.append(
                    replace_oov_with_mean(embeddings[i], ids=sequences[i])
                )
            embeddings = torch.stack(embeddings_list)

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
        lr: float,
        show_progress: bool = False,
    ):
        super().__init__()
        self.model = cnn_model
        self.optimizer_name = optimizer_name
        self.lr = lr
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
