import numpy as np
import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(
        self,
        embedding_matrix: torch.tensor | np.ndarray,
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
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)

        output, hidden = self.rnn(x)

        # extract sentence representation
        if self.sentence_representation_type == "last":
            sentence_representation = hidden[-1]
        elif self.sentence_representation_type == "max":
            sentence_representation, _ = torch.max(output, dim=1)
        elif self.sentence_representation_type == "average":
            sentence_representation = torch.mean(output, dim=1)

        # output layer
        logits = self.fc(sentence_representation)

        return logits
