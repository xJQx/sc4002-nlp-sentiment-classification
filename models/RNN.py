import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchmetrics.classification import MulticlassAccuracy


class RNN(nn.Module):
    def __init__(
        self,
        embedding_matrix: np.ndarray,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sentence_representation_type: str,
        freeze_embedding: bool = True,
    ):
        """
        RNN model with pretrained embeddings for text classification tasks.

        Args:
            embedding_matrix (Union[torch.Tensor, list]): Pretrained word embeddings matrix.
            hidden_dim (int): Dimensionality of the hidden layer in the RNN.
            output_dim (int): Dimensionality of the output layer.
            num_layers (int): Number of RNN layers.
            sentence_representation_type (str): Type of sentence representation to use
                                                ('last', 'max', 'average').
            freeze_embedding (Optional[bool]): Whether to freeze the embedding layer
                                               (default: True).
        """
        super(RNN, self).__init__()

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        if sentence_representation_type not in ["last", "max", "average"]:
            raise Exception(
                "Invalid `sentence_representation_type`. Choose from 'last', 'max', or 'average'."
            )
        self.sentence_representation_type = sentence_representation_type

        # embedding layer
        _, embedding_dim = embedding_matrix.shape
        self.embedding = nn.Embedding.from_pretrained(
            torch.FloatTensor(embedding_matrix), freeze=freeze_embedding
        )

        # rnn layer
        self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers, batch_first=True)

        # fc layer
        self.fc = nn.Linear(hidden_dim, hidden_dim // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)

    def forward(self, sequences, original_len):
        embeddings = self.embedding(sequences)

        # Handle variable length sequences
        packed_input = pack_padded_sequence(
            embeddings,
            lengths=original_len,
            enforce_sorted=False,
            batch_first=True,
        )

        packed_output, hidden = self.rnn(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        # extract sentence representation
        if self.sentence_representation_type == "last":
            sentence_representation = hidden[-1]

        elif self.sentence_representation_type == "max":
            sentence_representation, _ = torch.max(output, dim=1)
        elif self.sentence_representation_type == "average":
            sentence_representation = torch.mean(output, dim=1)

        # output layer
        logits = self.fc2(self.relu(self.fc(sentence_representation)))

        return logits

    def get_embeddings(self):
        return self.embedding.weight.data


class RNNClassifier(L.LightningModule):
    """RNN Classifier for binary classification tasks using PyTorch Lightning.

    Args:
        rnn_model (torch.nn.Module): The RNN model used for generating logits from inputs.
        optimizer_name (str): Name of the optimizer to use. Options include 'SGD', 'Adagrad', 'Adam', and 'RMSprop'.
        lr (float): Learning rate for the optimizer.
        show_progress (bool, optional): If True, logs additional progress information to the progress bar. Default is False.
    """

    def __init__(
        self,
        rnn_model: any,
        optimizer_name: str,
        lr: float,
        show_progress: bool = False,
    ):
        super().__init__()
        self.model = rnn_model
        self.optimizer_name = optimizer_name
        self.lr = lr
        self.show_progress = show_progress
        self.metric = MulticlassAccuracy(num_classes=2)
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        indexes = batch["indexes"]
        labels = batch["label"]
        original_lens = batch["original_len"]

        logits = self.model(indexes, original_lens)

        loss = F.cross_entropy(logits, labels)
        acc = self.metric(logits, labels)

        self.log("train_loss", loss, prog_bar=self.show_progress)
        self.log("train_acc", acc, prog_bar=self.show_progress)

        return loss

    def validation_step(self, batch, batch_idx):
        indexes = batch["indexes"]
        labels = batch["label"]
        original_lens = batch["original_len"]

        logits = self.model(indexes, original_lens)

        loss = F.cross_entropy(logits, labels)
        acc = self.metric(logits, labels)

        self.log("val_loss", loss, prog_bar=self.show_progress)
        self.log("val_acc", acc, prog_bar=self.show_progress)

    def test_step(self, batch, batch_idx):
        indexes = batch["indexes"]
        labels = batch["label"]
        original_lens = batch["original_len"]

        logits = self.model(indexes, original_lens)

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
