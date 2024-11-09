import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import MulticlassAccuracy


class MetaModel(L.LightningModule):

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        optimizer_name: str,
        learning_rate: float,
        dropout: float,
        num_classes: int = 2,
        show_progress: bool = True,
    ):
        super().__init__()
        self.optimizer_name = optimizer_name
        self.lr = learning_rate
        self.show_progress = show_progress
        self.metric = MulticlassAccuracy(num_classes=2)
        self.save_hyperparameters()

        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.fc(x)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch

        logits = self(inputs)

        loss = F.cross_entropy(logits, labels)
        acc = self.metric(logits, labels)

        self.log("train_loss", loss, prog_bar=self.show_progress)
        self.log("train_acc", acc, prog_bar=self.show_progress)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch

        logits = self(inputs)

        loss = F.cross_entropy(logits, labels)
        acc = self.metric(logits, labels)

        self.log("val_loss", loss, prog_bar=self.show_progress)
        self.log("val_acc", acc, prog_bar=self.show_progress)

    def test_step(self, batch, batch_idx):
        inputs, labels = batch

        logits = self(inputs)

        loss = F.cross_entropy(logits, labels)
        acc = self.metric(logits, labels)

        self.log("test_loss", loss, prog_bar=self.show_progress)
        self.log("test_acc", acc, prog_bar=self.show_progress)

    def configure_optimizers(self):
        if self.optimizer_name == "SGD":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        elif self.optimizer_name == "Adagrad":
            optimizer = torch.optim.Adagrad(self.parameters(), lr=self.lr)
        elif self.optimizer_name == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        elif self.optimizer_name == "RMSprop":
            optimizer = torch.optim.RMSprop(self.parameters(), lr=self.lr)
        else:
            raise Exception("Invalid optimizer name!")

        return optimizer
